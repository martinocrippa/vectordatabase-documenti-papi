#!/usr/bin/env python3
"""vdb.py - vector database dei documenti dei Papi: indice + ricerca ibrida (LanceDB).

Un solo file, poche primitive:
    documenti -> _segmenta/pezzi -> Embedder -> tabella LanceDB (vettori + full-text)
La ricerca e' IBRIDA e nativa: vettori (significato) + full-text/BM25 (parole),
fusi con RRF da LanceDB. Lo store e' on-disk in indice/, incrementale: niente piu'
vettori.npy/bm25.pkl/RRF fatti a mano.

In piu':
  - chunking strutturale (_segmenta): toglie intestazione e coda saluti italiani;
  - ogni chunk ha la lingua; i saluti tradotti sono 'escludibili' (fuori ricerca
    di default; --lingua / --tutto per controllarlo);
  - risultati deduplicati per documento (un solo chunk, il migliore);
  - pesi del modello in models/ (HF_HOME), non nella cache utente.

Il build sull'intero corpus e' lungo su CPU (l'embedding e' il collo di bottiglia)
ma resumabile: i chunk preparati sono in indice/meta.prepared.jsonl e LanceDB
persiste ogni blocco aggiunto, quindi al rilancio si riparte dalle righe gia'
presenti (niente re-embedding).

Uso:
    python vdb.py build                      # costruisce/aggiorna l'indice da data/
    python vdb.py build --per-papa 50        # campione fresco (prova veloce)
    python vdb.py search "cosa dice sulla pace?"
    python vdb.py search "..." --papa francesco --tipo angelus --lingua it -k 8
    python vdb.py search "..." --tutto       # includi i saluti tradotti
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys

import numpy as np
import py3langid as langid

# Lingue attese nel corpus (corpi + saluti). Restringere migliora l'accuratezza.
langid.set_languages(["it", "en", "fr", "es", "de", "pt", "pl"])

ROOT = pathlib.Path(__file__).resolve().parent   # data/, indice/, models/ stanno qui

# I pesi del modello vivono in models/ dentro il repo (gitignored), non nella
# cache utente. HF_HOME va impostato PRIMA di importare sentence_transformers.
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from sentence_transformers import SentenceTransformer  # noqa: E402
import lancedb  # noqa: E402
from lancedb.rerankers import RRFReranker  # noqa: E402

MODELLO = "intfloat/multilingual-e5-base"   # modello da retrieval (vedi review)
PREF_Q, PREF_P = "query: ", "passage: "     # e5 vuole questi prefissi
PAROLE_CHUNK = 180
BATCH = 2000                                # chunk per blocco di embedding/add
CAMPI = ("papa", "tipologia", "data", "titolo", "url")


# --- primitive ---------------------------------------------------------------

def _meta(testo: str) -> dict:
    """Estrae i campi dal frontmatter YAML (tra i primi due '---')."""
    m = {}
    for campo in CAMPI:
        # [ \t] e non \s: \s includerebbe il newline e, con un valore vuoto
        # (es. "data:" senza data), il match sconfinerebbe nella riga dopo.
        r = re.search(rf'^{campo}:[ \t]*"?(.*?)"?[ \t]*$', testo, re.M)
        m[campo] = r.group(1) if r else ""
    return m


def _corpo(testo: str) -> str:
    """Testo senza il frontmatter."""
    parti = testo.split("---", 2)
    return parti[2] if len(parti) == 3 else testo


def documenti(data_dir="data", per_papa=None):
    """Genera un dict per documento: i campi del frontmatter + il corpo.

    Con per_papa si campiona (sparso su tutte le tipologie) per fare prove
    veloci senza indicizzare tutto il corpus.
    """
    root = pathlib.Path(data_dir)
    for papa_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(papa_dir.rglob("*.md"))
        if per_papa:
            files = files[:: max(1, len(files) // per_papa)][:per_papa]
        for f in files:
            t = f.read_text(encoding="utf-8", errors="ignore")
            yield _meta(t) | {"corpo": _corpo(t)}


def pezzi(corpo: str, n: int = PAROLE_CHUNK) -> list[str]:
    """Spezza il corpo in finestre da ~n parole."""
    p = corpo.split()
    return [" ".join(p[i:i + n]) for i in range(0, len(p), n)] or [""]


_SALUTO = re.compile(r"^(Cari|Care|Caro|Fratelli|Sorelle|Eccellenz|Signor|Venerat)", re.I)
_CODA = re.compile(
    r"\n\s*(Dopo l['’]Angelus|Dopo la recita|Dopo il Regina|Dopo la preghiera)\b", re.I)


def _segmenta(corpo: str) -> tuple[str, str]:
    """Pulisce il corpo dalle parti non-discorso. Ritorna (corpo, coda_saluti).

    Toglie il blocco intestazione (titolo + bullet di metadati + riga di luogo/
    data in maiuscolo) e stacca la coda dei saluti italiani ("Dopo l'Angelus").
    KISS: solo marcatori e poche euristiche (vedi doc/mappa-tipologie.md).
    """
    # 1. via il blocco metadati: tutto fino al primo '---' su riga propria
    parti = re.split(r"\n-{3,}\n", corpo, maxsplit=1)
    testo = parti[1] if len(parti) == 2 else corpo
    # 2. stacca la coda saluti (marcatori italiani espliciti)
    m = _CODA.search(testo)
    coda = testo[m.start():].strip() if m else ""
    if m:
        testo = testo[:m.start()]
    # 3. salta l'intestazione: parti dal saluto se c'e' nelle prime righe,
    #    altrimenti dalla prima riga di prosa (minuscole >> maiuscole, lunga)
    righe = [r.strip() for r in testo.split("\n") if r.strip()]
    inizio = 0
    for j in range(min(len(righe), 8)):
        if _SALUTO.match(righe[j]):
            inizio = j
            break
    else:
        while inizio < len(righe):
            r = righe[inizio]
            minusc = sum(c.islower() for c in r)
            maiusc = sum(c.isupper() for c in r)
            if minusc > 2 * maiusc and len(r.split()) >= 8:
                break
            inizio += 1
    return "\n".join(righe[inizio:]), coda


def _lingua(testo: str) -> str:
    """Lingua del testo (codice ISO) o 'und' se troppo corto per decidere."""
    t = testo.strip()
    return langid.classify(t)[0] if len(t) >= 20 else "und"


def _prevalente(chunks: list[str], lingue: list[str]) -> str:
    """Lingua prevalente di un documento: quella con più parole (priorità al
    grosso del testo → di norma l'italiano del corpo, non i saluti)."""
    peso: collections.Counter = collections.Counter()
    for testo, lg in zip(chunks, lingue):
        peso[lg] += len(testo.split())
    peso.pop("und", None)
    return peso.most_common(1)[0][0] if peso else "und"


def _prepara_doc(doc: dict) -> list[dict]:
    """Prepara i chunk di UN documento (testo + sezione + lingua), senza embedding."""
    corpo, coda = _segmenta(doc["corpo"])
    segmenti = [("corpo", c) for c in pezzi(corpo)]
    if coda:
        segmenti += [("saluti", c) for c in pezzi(coda)]
    lingue = [_lingua(t) for _, t in segmenti]
    cc = [t for (s, t), _ in zip(segmenti, lingue) if s == "corpo"]
    cl = [lg for (s, _), lg in zip(segmenti, lingue) if s == "corpo"]
    lingua_doc = _prevalente(cc or [t for _, t in segmenti], cl or lingue)
    out = []
    for (sez, pz), lg in zip(segmenti, lingue):     # riusa lingue (no doppio langid)
        escl = sez == "saluti" or (lg != lingua_doc and lg != "und")
        out.append({c: doc[c] for c in CAMPI} | {
            "testo": pz, "sezione": sez, "lingua": lg, "escludibile": escl})
    return out


class Embedder:
    """Modello locale e5: testi -> vettori normalizzati.

    Niente cache degli embedding: LanceDB persiste i vettori, e il build riprende
    saltando i chunk già nella tabella. Si embedda solo ciò che non c'è ancora.
    """

    def __init__(self, modello=MODELLO):
        self.model = SentenceTransformer(modello)

    def passaggi(self, testi: list[str]) -> np.ndarray:
        return self.model.encode([PREF_P + t for t in testi],
                                 normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=False).astype("float32")

    def query(self, q: str) -> np.ndarray:
        return self.model.encode(PREF_Q + q, normalize_embeddings=True)


# --- costruzione e ricerca (LanceDB) -----------------------------------------

def _riga(m: dict, vec) -> dict:
    """Una riga della tabella LanceDB: metadati + testo + vettore."""
    return {**{c: m[c] for c in CAMPI}, "testo": m["testo"], "sezione": m["sezione"],
            "lingua": m["lingua"], "escludibile": bool(m["escludibile"]),
            "vector": np.asarray(vec, dtype="float32")}


def _prepara(data_dir, out_p, per_papa):
    """Prepara tutti i chunk. Per il build completo usa/aggiorna il checkpoint
    indice/meta.prepared.jsonl (salta lettura+lingua dei documenti già fatti)."""
    incrementale = per_papa is None
    prep = out_p / "meta.prepared.jsonl"
    meta, fatti = [], set()
    if incrementale and prep.exists():
        for r in prep.read_text(encoding="utf-8").splitlines():
            try:
                meta.append(json.loads(r))
            except ValueError:
                pass
        fatti = {m["url"] for m in meta}
        print(f"Riprendo preparazione: {len(meta)} chunk da {len(fatti)} documenti.")
    f = open(prep, "a", encoding="utf-8") if incrementale else None
    nuovi = 0
    for doc in documenti(data_dir, per_papa):
        if doc["url"] in fatti:
            continue
        chs = _prepara_doc(doc)
        meta.extend(chs)
        if f:
            for m in chs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
            nuovi += 1
            if nuovi % 1000 == 0:
                f.flush()
                print(f"  preparati altri {nuovi} documenti...")
    if f:
        f.flush()
        f.close()
    return meta


def costruisci(data_dir="data", out="indice", per_papa=None):
    """Prepara i chunk, li embedda a blocchi e li scrive in una tabella LanceDB.

    Resumabile: LanceDB persiste ogni blocco; al rilancio si riparte dal numero
    di righe già presenti. Un campione (--per-papa) ricrea la tabella da zero.
    """
    emb = Embedder()
    out_p = pathlib.Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    meta = _prepara(data_dir, out_p, per_papa)

    db = lancedb.connect(out)
    try:
        tab = db.open_table("chunk")
    except Exception:
        tab = None
    if per_papa is not None and tab is not None:
        db.drop_table("chunk")                       # campione: tabella fresca
        tab = None
    start = tab.count_rows() if tab is not None else 0
    print(f"{len(meta)} chunk; già nell'indice: {start}. Embedding + scrittura...")
    for i in range(start, len(meta), BATCH):
        blocco = meta[i:i + BATCH]
        vecs = emb.passaggi([m["testo"] for m in blocco])
        righe = [_riga(m, v) for m, v in zip(blocco, vecs)]
        if tab is None:
            tab = db.create_table("chunk", data=righe)
        else:
            tab.add(righe)
        print(f"  indicizzati {min(i + BATCH, len(meta))}/{len(meta)}")
    if tab is not None:
        tab.create_fts_index("testo", replace=True)   # full-text/BM25 per l'ibrido
        print(f"Indice LanceDB pronto: {tab.count_rows()} chunk in {out}/")
    return tab


# Query apposita: i primi saluti/benedizioni "Urbi et Orbi" dalla loggia subito
# dopo l'elezione (uno per Papa). Identificati per titolo (più affidabile della
# ricerca semantica: sono testi corti e generici). Esclude i "primi saluti" di
# viaggio (es. all'arrivo in un Paese).
_FILTRO_PRIMO_SALUTO = (
    "lower(titolo) LIKE '%prima benedizione%'"                       # Leone XIV
    " OR (lower(titolo) LIKE '%primo saluto%' AND ("                 # gli altri,
    "lower(titolo) LIKE '%urbi et orbi%'"                            # ma non i
    " OR lower(titolo) LIKE '%ai fedeli%'))")                        # saluti di viaggio


def primi_saluti(out="indice") -> list[dict]:
    """I primi saluti dalla loggia dopo l'elezione (un documento per Papa)."""
    tab = lancedb.connect(out).open_table("chunk")
    righe = tab.search().where(_FILTRO_PRIMO_SALUTO).limit(500).to_list()
    visti, risultati = set(), []
    for r in sorted(righe, key=lambda r: r.get("data", "")):
        u = r.get("url") or r["titolo"]
        if u not in visti:
            visti.add(u)
            risultati.append(r)
    return risultati


def cerca(query: str, k=5, papa=None, tipo=None, lingua=None, tutto=False,
          out="indice", n=60) -> list[dict]:
    """Ricerca ibrida nativa LanceDB: vettori + full-text fusi con RRF.

    Di default esclude i chunk "escludibili" (saluti/tradotti); `tutto=True` li
    reinclude. Filtri su papa, tipologia, lingua. Dedup per documento.
    """
    emb = Embedder()
    tab = lancedb.connect(out).open_table("chunk")
    conds = [] if tutto else ["escludibile = false"]
    if lingua:
        conds.append(f"lingua = '{lingua}'")
    if tipo:
        conds.append(f"tipologia = '{tipo}'")
    q = tab.search(query_type="hybrid").vector(emb.query(query)).text(query).limit(n)
    if conds:
        q = q.where(" AND ".join(conds), prefilter=True)
    risultati, visti = [], set()
    for r in q.rerank(RRFReranker()).to_list():
        if papa and papa.lower() not in r["papa"].lower():
            continue
        u = r.get("url") or r["titolo"]
        if u in visti:
            continue
        visti.add(u)
        risultati.append(r)
        if len(risultati) >= k:
            break
    return risultati


# --- CLI ---------------------------------------------------------------------

def main() -> int:
    # La console Windows (cp1252) non stampa certi caratteri: forziamo UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="vector database dei documenti dei Papi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="costruisce/aggiorna l'indice LanceDB da data/")
    b.add_argument("--data", default=str(ROOT / "data"))
    b.add_argument("--out", default=str(ROOT / "indice"))
    b.add_argument("--per-papa", type=int, default=None,
                   help="campiona N documenti per Papa (tabella fresca, per prove)")

    s = sub.add_parser("search", help="ricerca ibrida sull'indice")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--papa", default=None)
    s.add_argument("--tipo", default=None)
    s.add_argument("--lingua", default=None, help="filtra i chunk per lingua (es. it)")
    s.add_argument("--tutto", action="store_true",
                   help="includi anche i chunk escludibili (saluti/tradotti)")
    s.add_argument("--out", default=str(ROOT / "indice"))

    ps = sub.add_parser("primo-saluto",
                        help="i primi saluti 'Urbi et Orbi' dopo l'elezione (uno per Papa)")
    ps.add_argument("--out", default=str(ROOT / "indice"))

    a = ap.parse_args()
    if a.cmd == "build":
        costruisci(a.data, a.out, a.per_papa)
    elif a.cmd == "search":
        for r, m in enumerate(cerca(a.query, a.k, a.papa, a.tipo,
                                    a.lingua, a.tutto, a.out), 1):
            print(f"{r}. {m['papa']} · {m['tipologia']} · {m['data']} · [{m.get('lingua', '?')}]")
            print(f"   {m['titolo']}")
            print(f"   …{m['testo'][:160].strip()}…")
            print(f"   {m['url']}")
    elif a.cmd == "primo-saluto":
        for m in primi_saluti(a.out):
            print(f"{m['papa']} · {m['data']}")
            print(f"   {m['titolo']}")
            print(f"   {m['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
