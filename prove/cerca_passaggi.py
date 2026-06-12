#!/usr/bin/env python3
"""Esperimento 2: ricerca IBRIDA per passaggi (vettori + BM25 + RRF).

E' il mini-prototipo dello Stadio 3 del piano, in un file: si costruisce un
indice su un campione del corpus, poi si interroga e si confrontano i tre
modi di recuperare i passaggi piu' pertinenti:

    BM25 (parole)   vs   vettori (significato)   vs   ibrido (RRF)

Mostra il punto chiave: la query "ecologia / pianeta" recupera, via vettori,
passaggi che dicono "casa comune", "madre terra", "creato" — che il solo BM25
non trova. L'ibrido tiene il meglio dei due. NESSUNA soglia: solo ranking.

Uso:
    python prove/cerca_passaggi.py
    python prove/cerca_passaggi.py "che cosa dice sulla guerra e la pace?"
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODELLO = "intfloat/multilingual-e5-base"
PREF_Q, PREF_P = "query: ", "passage: "
PAROLE_CHUNK = 180
PER_PAPA = 150
N = 50          # candidati per retriever
K_RRF = 60      # costante RRF
TOP = 5         # risultati mostrati
QUERY = " ".join(sys.argv[1:]) or "la salvaguardia del pianeta e dell'ecologia"


def corpo(t: str) -> str:
    p = t.split("---", 2)
    return p[2] if len(p) == 3 else t


def pezzi(t: str, n: int = PAROLE_CHUNK) -> list[str]:
    w = t.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [""]


def tok(s: str) -> list[str]:
    return [x for x in re.findall(r"[a-zàèéìòóù]+", s.lower()) if len(x) > 2]


def campiona(root: pathlib.Path, per_papa: int) -> list[dict]:
    """~per_papa documenti per Papa, sparsi su tutte le tipologie/anni."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        tutti = sorted(d.rglob("*.md"))
        files = tutti[:: max(1, len(tutti) // per_papa)][:per_papa]
        for f in files:
            t = f.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'^titolo:\s*"?(.+?)"?\s*$', t, re.M)
            dt = re.search(r"^data:\s*(\S+)", t, re.M)
            out.append({"papa": d.name, "data": dt.group(1) if dt else "?",
                        "titolo": m.group(1) if m else f.stem, "corpo": corpo(t)})
    return out


def rrf(*ranking: list[int]) -> dict[int, float]:
    """Reciprocal Rank Fusion: fonde liste di id ordinate per posizione."""
    punti: dict[int, float] = {}
    for lista in ranking:
        for pos, idx in enumerate(lista):
            punti[idx] = punti.get(idx, 0.0) + 1.0 / (K_RRF + pos)
    return punti


def mostra(titolo: str, ids: list[int], chunks: list[dict], extra=None) -> None:
    print(f"\n— {titolo} —")
    for r, i in enumerate(ids[:TOP], 1):
        c = chunks[i]
        tag = f"  [{extra[i]}]" if extra else ""
        print(f"  {r}. {c['papa']:14} {c['data']:10} {c['titolo'][:54]}{tag}")
        print(f"     …{c['testo'][:150].strip()}…")


def main() -> int:
    root = pathlib.Path("data")
    if not root.is_dir():
        print("Cartella data/ assente."); return 1

    model = SentenceTransformer(MODELLO)
    docs = campiona(root, PER_PAPA)

    # un chunk = unita' di ricerca, con i metadati del documento
    chunks = [{"papa": d["papa"], "data": d["data"], "titolo": d["titolo"],
               "testo": pz} for d in docs for pz in pezzi(d["corpo"])]
    print(f"Indice su {len(docs)} documenti -> {len(chunks)} chunk. Embedding...")

    M = model.encode([PREF_P + c["testo"] for c in chunks],
                     normalize_embeddings=True, batch_size=64,
                     show_progress_bar=False)
    bm25 = BM25Okapi([tok(c["testo"]) for c in chunks])

    # --- ricerca ---
    qv = model.encode(PREF_Q + QUERY, normalize_embeddings=True)
    vec_rank = list(np.argsort(-(M @ qv))[:N])
    bm_scores = bm25.get_scores(tok(QUERY))
    bm_rank = list(np.argsort(-bm_scores)[:N])

    fusi = rrf(vec_rank, bm_rank)
    hyb_rank = sorted(fusi, key=lambda i: -fusi[i])

    # da dove arriva ogni risultato ibrido (V=vettori, K=keyword)
    sv, sk = set(vec_rank), set(bm_rank)
    prov = {i: ("V+K" if i in sv and i in sk else "V" if i in sv else "K")
            for i in hyb_rank}

    print(f"\n### QUERY: «{QUERY}»")
    mostra("solo BM25 (parole)", bm_rank, chunks)
    mostra("solo vettori (significato)", vec_rank, chunks)
    mostra("IBRIDO (RRF)", hyb_rank, chunks, extra=prov)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
