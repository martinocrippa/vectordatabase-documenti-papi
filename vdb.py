#!/usr/bin/env python3
"""vdb.py - vector database dei documenti dei Papi: indice + ricerca ibrida.

Un solo file, poche primitive (vedi doc/architettura.md):
    documenti -> pezzi -> Embedder -> Indice (vettori + BM25)
La ricerca e' IBRIDA: vettori (significato) + BM25 (parole) fusi con RRF,
senza soglie. L'indice si costruisce una volta e si salva in indice/.

In piu':
  - ogni chunk e' etichettato con la lingua; i saluti tradotti in altra lingua
    (i "doppioni") sono marcati escludibili e tenuti fuori dalla ricerca di
    default (vedi --lingua / --tutto);
  - i risultati sono deduplicati per documento (un solo chunk, il migliore);
  - i pesi del modello stanno in models/ dentro il repo (HF_HOME), non nella
    cache utente; la cache embedding si carica solo in build (non in search).

Il build sull'intero corpus e' lungo su CPU ma **resumabile**: cache embedding
(models/) + checkpoint dei chunk (indice/meta.prepared.jsonl). Questi puntelli
sono interim: la direzione e' migrare lo store a LanceDB (doc/scelta-store.md).

Uso:
    python vdb.py build                         # costruisce indice/ da data/
    python vdb.py build --per-papa 50           # campione (per provare)
    python vdb.py search "cosa dice sulla pace?"
    python vdb.py search "..." --papa francesco --tipo angelus -k 8
    python vdb.py search "..." --lingua it       # solo chunk italiani
    python vdb.py search "..." --tutto           # includi i saluti tradotti
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import pickle
import re
import sys

import numpy as np
import py3langid as langid
from rank_bm25 import BM25Okapi

# Lingue attese nel corpus (corpi + saluti). Restringere migliora l'accuratezza
# del riconoscimento su testi brevi.
langid.set_languages(["it", "en", "fr", "es", "de", "pt", "pl"])

ROOT = pathlib.Path(__file__).resolve().parent   # data/, indice/, models/ stanno qui

# I pesi del modello vivono in models/ dentro il repo (gitignored), non nella
# cache utente. HF_HOME va impostato PRIMA di importare sentence_transformers.
# setdefault: se l'utente ha gia' un HF_HOME suo, lo rispettiamo.
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from sentence_transformers import SentenceTransformer  # noqa: E402

MODELLO = "intfloat/multilingual-e5-base"   # modello da retrieval (vedi review)
PREF_Q, PREF_P = "query: ", "passage: "     # e5 vuole questi prefissi
PAROLE_CHUNK = 180
K_RRF = 60                                  # costante della Reciprocal Rank Fusion
N_CANDIDATI = 50                            # candidati per retriever prima della fusione
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
    """Spezza il corpo in finestre da ~n parole.

    Chunking semplice (un primo passo). Quello strutturale per tipologia
    e' descritto in doc/mappa-tipologie.md ed e' il refinement successivo.
    """
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


def _tok(s: str) -> list[str]:
    """Tokenizzazione semplice per BM25 (parole di >2 lettere, minuscole)."""
    return [w for w in re.findall(r"[a-zàèéìòóù]+", s.lower()) if len(w) > 2]


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


class Embedder:
    """Incapsula il modello locale: testi -> vettori normalizzati.

    I vettori dei passaggi sono messi in **cache su disco** (in models/, keyed da
    hash del testo): ogni chunk si embedda una volta sola, i run successivi lo
    riusano. Niente piu' re-embedding dello stesso testo a ogni build/esperimento.
    """

    def __init__(self, modello=MODELLO):
        self.model = SentenceTransformer(modello)
        self._cache_path = ROOT / "models" / f"emb_{modello.split('/')[-1]}.pkl"
        self._cache = None   # pigra: la cache serve solo in build, non in search

    def _carica_cache(self) -> dict:
        if self._cache is None:
            self._cache = (pickle.loads(self._cache_path.read_bytes())
                           if self._cache_path.exists() else {})
        return self._cache

    def passaggi(self, testi: list[str], checkpoint: int = 8000) -> np.ndarray:
        self._carica_cache()
        chiavi = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in testi]
        manca = [(t, k) for t, k in zip(testi, chiavi) if k not in self._cache]
        # embedding a blocchi con salvataggio periodico: un build lungo che si
        # interrompe riprende dalla cache (i chunk gia' fatti sono saltati).
        for s in range(0, len(manca), checkpoint):
            blocco = manca[s:s + checkpoint]
            v = self.model.encode([PREF_P + t for t, _ in blocco],
                                  normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=False).astype("float32")
            for (_, k), riga in zip(blocco, v):
                self._cache[k] = riga
            self._salva_cache()
            if len(manca) > checkpoint:
                print(f"  embedding: {min(s + checkpoint, len(manca))}/{len(manca)} "
                      f"nuovi chunk (cache salvata)")
        return np.vstack([self._cache[k] for k in chiavi])

    def _salva_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")   # salvataggio atomico
        tmp.write_bytes(pickle.dumps(self._cache))
        os.replace(tmp, self._cache_path)

    def query(self, q: str) -> np.ndarray:
        return self.model.encode(PREF_Q + q, normalize_embeddings=True)


class Indice:
    """Vettori (matrice NumPy) + BM25 sui testi + metadati paralleli."""

    def __init__(self, vettori: np.ndarray, bm25: BM25Okapi, meta: list[dict]):
        self.vettori, self.bm25, self.meta = vettori, bm25, meta

    def salva(self, out="indice") -> None:
        p = pathlib.Path(out)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vettori.npy", self.vettori)
        (p / "bm25.pkl").write_bytes(pickle.dumps(self.bm25))
        with open(p / "meta.jsonl", "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def carica(cls, out="indice") -> "Indice":
        p = pathlib.Path(out)
        vettori = np.load(p / "vettori.npy")
        bm25 = pickle.loads((p / "bm25.pkl").read_bytes())
        meta = [json.loads(r) for r in
                (p / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
        return cls(vettori, bm25, meta)

    def per_vettore(self, qv: np.ndarray, k: int) -> list[int]:
        return list(np.argsort(-(self.vettori @ qv))[:k])

    def per_keyword(self, qtok: list[str], k: int) -> list[int]:
        return list(np.argsort(-self.bm25.get_scores(qtok))[:k])


def _rrf(*ranking: list[int]) -> dict[int, float]:
    """Reciprocal Rank Fusion: fonde liste ordinate usando solo le posizioni."""
    punti: dict[int, float] = {}
    for lista in ranking:
        for pos, idx in enumerate(lista):
            punti[int(idx)] = punti.get(int(idx), 0.0) + 1.0 / (K_RRF + pos)
    return punti


# --- orchestrazione ----------------------------------------------------------

def _prepara_doc(doc: dict) -> list[dict]:
    """Prepara i chunk di UN documento (testo + sezione + lingua), senza embedding."""
    corpo, coda = _segmenta(doc["corpo"])
    # chunk del corpo (sezione=corpo) + chunk della coda saluti (sezione=saluti)
    segmenti = [("corpo", c) for c in pezzi(corpo)]
    if coda:
        segmenti += [("saluti", c) for c in pezzi(coda)]
    lingue = [_lingua(t) for _, t in segmenti]
    cc = [t for (s, t), _ in zip(segmenti, lingue) if s == "corpo"]
    cl = [lg for (s, _), lg in zip(segmenti, lingue) if s == "corpo"]
    lingua_doc = _prevalente(cc or [t for _, t in segmenti], cl or lingue)
    out = []
    for i, ((sez, pz), lg) in enumerate(zip(segmenti, lingue)):
        # escludibile dalla ricerca: i saluti (coda) e i "doppioni" tradotti
        # (chunk in lingua diversa dal corpo del documento).
        escl = sez == "saluti" or (lg != lingua_doc and lg != "und")
        out.append({c: doc[c] for c in CAMPI} | {
            "testo": pz, "i": i, "sezione": sez, "lingua": lg,
            "lingua_doc": lingua_doc, "escludibile": escl})
    return out


def costruisci(data_dir="data", out="indice", per_papa=None) -> Indice:
    """documenti -> pezzi -> embedding + BM25 -> salva.

    **Resumabile.** La preparazione dei chunk (lettura + lingua, la parte lenta)
    è salvata documento per documento in <out>/meta.prepared.jsonl: se il build
    si interrompe, al rilancio salta i documenti già preparati. Con la cache
    embedding (Embedder) anche i vettori si riusano. Così bastano pochi rilanci
    per arrivare in fondo, anche su CPU.
    """
    emb = Embedder()
    out_p = pathlib.Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    incrementale = per_papa is None       # solo il build completo è resumabile
    prep = out_p / "meta.prepared.jsonl"
    meta, fatti = [], set()
    if incrementale and prep.exists():
        for r in prep.read_text(encoding="utf-8").splitlines():
            try:
                meta.append(json.loads(r))
            except ValueError:
                pass                       # tollera l'ultima riga troncata
        fatti = {m["url"] for m in meta}
        print(f"Riprendo: {len(meta)} chunk da {len(fatti)} documenti già preparati.")

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
    print(f"{len(meta)} chunk pronti. Calcolo embedding (modello {MODELLO})...")
    vettori = emb.passaggi([m["testo"] for m in meta]).astype("float32")
    bm25 = BM25Okapi([_tok(m["testo"]) for m in meta])
    idx = Indice(vettori, bm25, meta)
    idx.salva(out)
    print(f"Indice salvato in {out}/ ({len(meta)} chunk).")
    return idx


def cerca(query: str, k=5, papa=None, tipo=None, lingua=None, tutto=False,
          out="indice", n=N_CANDIDATI) -> list[dict]:
    """Ricerca ibrida: vettori + BM25 fusi con RRF, poi filtri sui metadati.

    Di default esclude i chunk "escludibili" (saluti tradotti in altra lingua):
    `tutto=True` li reinclude. `lingua` filtra su una lingua specifica.
    """
    emb = Embedder()
    idx = Indice.carica(out)
    fusi = _rrf(idx.per_vettore(emb.query(query), n),
                idx.per_keyword(_tok(query), n))
    risultati, visti = [], set()
    for i in sorted(fusi, key=lambda j: -fusi[j]):
        m = idx.meta[i]
        if not tutto and m.get("escludibile"):
            continue
        if lingua and m.get("lingua") != lingua:
            continue
        if papa and m["papa"].lower().find(papa.lower()) < 0:
            continue
        if tipo and m["tipologia"] != tipo:
            continue
        doc = m.get("url") or m.get("titolo")   # un solo (il migliore) chunk per documento
        if doc in visti:
            continue
        visti.add(doc)
        risultati.append(m)
        if len(risultati) >= k:
            break
    return risultati


# --- CLI ---------------------------------------------------------------------

def main() -> int:
    # La console Windows (cp1252) non stampa certi caratteri (titoli con accenti
    # o lettere straniere): forziamo UTF-8 per non far crashare la stampa.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="vector database dei documenti dei Papi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="costruisce l'indice da data/")
    b.add_argument("--data", default=str(ROOT / "data"))
    b.add_argument("--out", default=str(ROOT / "indice"))
    b.add_argument("--per-papa", type=int, default=None,
                   help="campiona N documenti per Papa (per prove veloci)")

    s = sub.add_parser("search", help="ricerca ibrida sull'indice")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--papa", default=None)
    s.add_argument("--tipo", default=None)
    s.add_argument("--lingua", default=None, help="filtra i chunk per lingua (es. it)")
    s.add_argument("--tutto", action="store_true",
                   help="includi anche i chunk escludibili (saluti tradotti)")
    s.add_argument("--out", default=str(ROOT / "indice"))

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
