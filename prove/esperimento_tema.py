#!/usr/bin/env python3
"""Esperimento mirato su un TEMA: BM25 vs vettori vs ibrido, su materiale vero.

Per avere contenuto vero senza indicizzare tutto il corpus, costruisce un indice
in memoria su misura: i documenti che nominano il tema (per parola chiave, su
TUTTO il corpus) + un campione di sfondo per contrasto. Poi:
  - confronta i tre metodi sulla prima query del tema;
  - mostra la sovrapposizione (chi trova cosa);
  - dà il footprint per Papa (per rispondere a "ne parla solo un Papa?").

Generalizza gli esperimenti calcio/sport: un solo file, tanti temi. Riusa le
primitive di vdb.py.

Uso:
    python prove/esperimento_tema.py ambiente
    python prove/esperimento_tema.py calcio
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from vdb import (CAMPI, ROOT, Embedder, Indice, _corpo, _lingua,  # noqa: E402
                 _meta, _prevalente, _rrf, _segmenta, _tok, pezzi, N_CANDIDATI)
from rank_bm25 import BM25Okapi  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# tema -> (regex parole chiave per raccogliere i documenti, query semantiche)
TEMI = {
    "calcio": (
        r"calcio|football|soccer",
        ["il calcio, i mondiali di calcio e i campioni",
         "lo sport, gli atleti e le competizioni sportive"],
    ),
    "ambiente": (
        # termini ambientali specifici (no "creato" da solo: matcha "Dio ha creato")
        r"ambient|ecolog|inquinament|biodivers|casa comune|madre terra|"
        r"(custodia|cura|salvaguardia) del creato|cambiament[oi] climatic|"
        r"riscaldamento globale",
        ["la salvaguardia dell'ambiente e la cura del creato",
         "la custodia della casa comune e il rispetto della terra",
         "l'ecologia, il clima e la natura"],
    ),
}
SFONDO_PER_PAPA = 80
TOP = 6


def raccogli(data_dir: pathlib.Path, rx: re.Pattern):
    """{path: è_tema} + conteggio dei documenti-tema per Papa (su tutto il corpus)."""
    scelti, per_papa = {}, collections.Counter()
    for papa_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        tutti = sorted(papa_dir.rglob("*.md"))
        tema = [f for f in tutti
                if rx.search(f.read_text(encoding="utf-8", errors="ignore"))]
        per_papa[papa_dir.name] = len(tema)
        sfondo = tutti[:: max(1, len(tutti) // SFONDO_PER_PAPA)][:SFONDO_PER_PAPA]
        for f in tema:
            scelti[f] = True
        for f in sfondo:
            scelti.setdefault(f, False)
    return scelti, per_papa


def costruisci_mem(scelti: dict):
    """Indice in memoria (embedding + BM25) sui documenti scelti."""
    emb = Embedder()
    meta = []
    for f, is_tema in scelti.items():
        t = f.read_text(encoding="utf-8", errors="ignore")
        doc = _meta(t) | {"corpo": _corpo(t)}
        corpo, coda = _segmenta(doc["corpo"])
        segmenti = [("corpo", c) for c in pezzi(corpo)]
        if coda:
            segmenti += [("saluti", c) for c in pezzi(coda)]
        lingue = [_lingua(x) for _, x in segmenti]
        cc = [x for (s, x), _ in zip(segmenti, lingue) if s == "corpo"]
        cl = [lg for (s, _), lg in zip(segmenti, lingue) if s == "corpo"]
        ld = _prevalente(cc or [x for _, x in segmenti], cl or lingue)
        for (sez, pz), lg in zip(segmenti, lingue):
            escl = sez == "saluti" or (lg != ld and lg != "und")
            meta.append({c: doc[c] for c in CAMPI} | {
                "testo": pz, "lingua": lg, "escludibile": escl, "tema": is_tema})
    print(f"Indice in memoria: {len(scelti)} documenti -> {len(meta)} chunk. Embedding...")
    M = emb.passaggi([m["testo"] for m in meta]).astype("float32")
    bm = BM25Okapi([_tok(m["testo"]) for m in meta])
    return emb, Indice(M, bm, meta)


def top_docs(idx, ranking, k):
    """Da un ranking di chunk a documenti distinti (esclude i saluti)."""
    out, visti = [], set()
    for i in ranking:
        m = idx.meta[i]
        if m["escludibile"]:
            continue
        d = m.get("url") or m["titolo"]
        if d in visti:
            continue
        visti.add(d)
        out.append(m)
        if len(out) >= k:
            break
    return out


def main() -> int:
    tema = sys.argv[1] if len(sys.argv) > 1 else "ambiente"
    if tema not in TEMI:
        print(f"Tema sconosciuto: {tema}. Disponibili: {', '.join(TEMI)}"); return 1
    rx_str, queries = TEMI[tema]
    rx = re.compile(rx_str, re.I)
    data = ROOT / "data"
    if not data.is_dir():
        print("data/ assente."); return 1

    print(f"TEMA: {tema}  —  parole chiave: {rx_str}")
    print("Raccolgo i documenti del tema + sfondo (leggo tutto il corpus)...")
    scelti, per_papa = raccogli(data, rx)
    n = sum(scelti.values())
    print(f"  {n} documenti-tema + {len(scelti) - n} di sfondo")
    print("  documenti-tema per Papa (parola chiave, TUTTO il corpus):")
    for papa, c in per_papa.most_common():
        print(f"    {papa:22} {c}")
    emb, idx = costruisci_mem(scelti)

    # --- confronto dei tre metodi sulla prima query ---
    q = queries[0]
    vec = [int(i) for i in idx.per_vettore(emb.query(q), N_CANDIDATI)]
    bm = [int(i) for i in idx.per_keyword(_tok(q), N_CANDIDATI)]
    fusi = _rrf(vec, bm)
    hyb = sorted(fusi, key=lambda j: -fusi[j])
    print("\n" + "=" * 78 + f"\nCONFRONTO METODI — QUERY: «{q}»")
    for nome, rank in (("BM25 (parole)", bm), ("vettori (significato)", vec),
                       ("IBRIDO (RRF)", hyb)):
        print(f"\n— {nome} —")
        for m in top_docs(idx, rank, TOP):
            flag = tema.upper() if m["tema"] else "sfondo"
            print(f"  [{flag}] {m['papa'][:18]:18} {m['data']:10} {m['titolo'][:48]}")

    # --- sovrapposizione tra metodi (top-15, documenti distinti) ---
    dv = {(m.get("url") or m["titolo"]): m["tema"] for m in top_docs(idx, vec, 15)}
    db = {(m.get("url") or m["titolo"]): m["tema"] for m in top_docs(idx, bm, 15)}
    com, solo_v, solo_b = set(dv) & set(db), set(dv) - set(db), set(db) - set(dv)
    cal = lambda S, d: sum(d[x] for x in S)
    print("\n" + "=" * 78 + "\nSOVRAPPOSIZIONE (top-15 per metodo)")
    print(f"  entrambi    : {len(com):2}  (di cui tema: {cal(com, {**dv, **db})})")
    print(f"  solo vettori: {len(solo_v):2}  (di cui tema: {cal(solo_v, dv)})")
    print(f"  solo BM25   : {len(solo_b):2}  (di cui tema: {cal(solo_b, db)})")

    # --- footprint del tema per Papa nei risultati ibridi (unione query) ---
    perpapa = collections.Counter()
    for qq in queries:
        f2 = _rrf([int(i) for i in idx.per_vettore(emb.query(qq), N_CANDIDATI)],
                  [int(i) for i in idx.per_keyword(_tok(qq), N_CANDIDATI)])
        rk = sorted(f2, key=lambda j: -f2[j])
        for m in top_docs(idx, rk, 10):
            if m["tema"]:
                perpapa[m["papa"]] += 1
    print("\n" + "=" * 78 + "\nFOOTPRINT nei top-10 ibridi (unione query), per Papa:")
    for papa, c in perpapa.most_common():
        print(f"  {papa:22} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
