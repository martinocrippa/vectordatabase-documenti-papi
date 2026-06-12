#!/usr/bin/env python3
"""Esperimento esplorativo: i Papi e lo SPORT, con i tre metodi di ricerca.

Confronta sullo stesso indice (costruito da vdb.py):
    BM25 (parole)  vs  vettori (significato)  vs  ibrido (RRF)

Riusa le primitive di vdb.py. Mostra, per alcune query a tema sport, i passaggi
trovati da ciascun metodo, e un quadro AGGREGATO (documenti distinti, per Papa,
sovrapposizione tra metodi).

⚠️ Esplorativo: l'indice e' un CAMPIONE del corpus, quindi i conteggi per Papa
non sono autorevoli (un'assenza puo' essere effetto del campionamento, non del
fatto che il Papa non ne abbia parlato). Il confronto FRA METODI invece e' equo.

Uso:  python prove/esperimento_sport.py
"""

from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from vdb import ROOT, Embedder, Indice, _rrf, _tok, N_CANDIDATI  # noqa: E402

# La console Windows (cp1252) non stampa certi caratteri (es. polacchi): UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = [
    "lo sport, gli atleti e le competizioni sportive",
    "il calcio e il campionato mondiale",
    "i giochi olimpici e le olimpiadi",
]
TOP = 8        # risultati mostrati per metodo
AGG = 20       # profondita' per l'aggregato


def _doc(m: dict) -> str:
    return m.get("url") or m.get("titolo", "")


def main() -> int:
    idx = Indice.carica(str(ROOT / "indice"))
    emb = Embedder()
    print(f"Indice: {len(idx.meta)} chunk.\n")

    agg = {"BM25": [], "vettori": [], "ibrido": []}

    for q in QUERIES:
        qv = emb.query(q)
        vec = [int(i) for i in idx.per_vettore(qv, N_CANDIDATI)]
        bm = [int(i) for i in idx.per_keyword(_tok(q), N_CANDIDATI)]
        fusi = _rrf(vec, bm)
        hyb = sorted(fusi, key=lambda i: -fusi[i])
        sv, sk = set(vec), set(bm)

        print("=" * 78)
        print(f"QUERY: «{q}»")
        for nome, rank in (("BM25 (parole)", bm), ("vettori (significato)", vec)):
            print(f"\n— {nome} —")
            for r, i in enumerate(rank[:TOP], 1):
                m = idx.meta[i]
                print(f"  {r}. {m['papa']:14} {m['data']:10} {m['titolo'][:56]}")
        print("\n— IBRIDO (RRF) —")
        for r, i in enumerate(hyb[:TOP], 1):
            m = idx.meta[i]
            tag = "V+K" if i in sv and i in sk else "V" if i in sv else "K"
            print(f"  {r}. [{tag}] {m['papa']:14} {m['data']:10} {m['titolo'][:52]}")
        print()

        for nome, rank in (("BM25", bm), ("vettori", vec), ("ibrido", hyb)):
            agg[nome] += [_doc(idx.meta[i]) for i in rank[:AGG]]

    # --- quadro aggregato (esplorativo) ---
    print("=" * 78)
    print("AGGREGATO (documenti DISTINTI nei top risultati, unione delle query)\n")
    docset = {k: set(v) for k, v in agg.items()}
    for nome in ("BM25", "vettori", "ibrido"):
        print(f"  {nome:8} {len(docset[nome])} documenti distinti")

    print("\nSovrapposizione BM25 vs vettori (sui documenti distinti):")
    soloK = docset["BM25"] - docset["vettori"]
    soloV = docset["vettori"] - docset["BM25"]
    ent = docset["BM25"] & docset["vettori"]
    print(f"  trovati da entrambi : {len(ent)}")
    print(f"  solo BM25           : {len(soloK)}  (parola esatta, niente senso vicino)")
    print(f"  solo vettori        : {len(soloV)}  (senso vicino, senza la parola)")

    # per-Papa, sui documenti dell'ibrido (con caveat campionamento)
    perpapa = collections.Counter()
    url2papa = {_doc(m): m["papa"] for m in idx.meta}
    for d in docset["ibrido"]:
        perpapa[url2papa.get(d, "?")] += 1
    print("\nDocumenti (ibrido) per Papa — ⚠️ campione, non autorevole:")
    for papa, n in perpapa.most_common():
        print(f"  {papa:18} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
