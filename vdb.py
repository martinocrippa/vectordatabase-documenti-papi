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
    cache utente.

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
    """Incapsula il modello locale: testi -> vettori normalizzati."""

    def __init__(self, modello=MODELLO):
        self.model = SentenceTransformer(modello)

    def passaggi(self, testi: list[str]) -> np.ndarray:
        return self.model.encode([PREF_P + t for t in testi],
                                 normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=False)

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

def costruisci(data_dir="data", out="indice", per_papa=None) -> Indice:
    """documenti -> pezzi -> embedding + BM25 -> salva."""
    emb = Embedder()
    meta = []
    for doc in documenti(data_dir, per_papa):
        chs = pezzi(doc["corpo"])
        lingue = [_lingua(t) for t in chs]
        lingua_doc = _prevalente(chs, lingue)   # flag lingua prevalente del documento
        for i, (pz, lg) in enumerate(zip(chs, lingue)):
            # un chunk in lingua diversa dal documento e' un "doppione" (saluto
            # tradotto): lo teniamo ma marchiamo escludibile dalla ricerca.
            meta.append({c: doc[c] for c in CAMPI} | {
                "testo": pz, "i": i, "lingua": lg, "lingua_doc": lingua_doc,
                "escludibile": lg != lingua_doc and lg != "und"})
    print(f"{len(meta)} chunk. Calcolo embedding (modello {MODELLO})...")
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
