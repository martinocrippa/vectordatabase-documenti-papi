#!/usr/bin/env python3
"""Analisi dei temi per Papa: confronto REGEX vs EMBEDDING, sui temi soliti e
sui temi NUOVI (aborto, valore della vita, dignità dell'uomo, intelligenza
artificiale, cambiamento della Chiesa).

Per ogni tema:
  - regex   = % di documenti con un chunk che matcha le parole chiave;
  - semantico = % di documenti tra i top-N più vicini al concetto (N = numero di
    positivi regex, così i due metodi marcano lo stesso *volume* e si confronta
    la distribuzione fra Papi, senza scegliere una soglia arbitraria).
Più una riga di sovrapposizione (quanti documenti trovano entrambi).

⚠️ Alcuni temi non esistono per tutti nel tempo (l'IA è solo recente): il dato
per Papa va letto con questa cautela, segnalata nei risultati.

Gira sull'indice LanceDB (vettori dei chunk già pronti). Uso:
    python prove/analisi_temi.py
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from vdb import ROOT, Embedder, lancedb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# tema -> (regex parole chiave, concetto per l'embedding)
TEMI = {
    # --- soliti, per contesto ---
    "poveri": (r"pover|emarginat|bisognos",
               "i poveri, gli emarginati, gli ultimi, chi è nel bisogno"),
    "migranti": (r"migrant|rifugiat|profugh|immigrat",
                 "i migranti, i rifugiati, chi fugge dalla guerra e cerca accoglienza"),
    "ambiente": (r"ambient|ecolog|casa comune",
                 "la cura del creato, l'ambiente, l'ecologia, la casa comune"),
    # --- NUOVI ---
    "aborto": (r"aborto|abortiv",
               "l'aborto e la difesa della vita nascente e non ancora nata"),
    "valore della vita": (r"valore della vita|sacralità della vita|sacralit\w* della vita",
                          "il valore e la sacralità della vita umana, dal concepimento alla morte naturale"),
    "dignità dell'uomo": (r"dignità (dell'uomo|della persona|umana)",
                          "la dignità della persona umana e il valore inviolabile di ogni essere umano"),
    "intelligenza artificiale": (r"intelligenza artificiale|algoritm|robotic",
                                 "l'intelligenza artificiale, gli algoritmi e le nuove tecnologie digitali e i loro rischi"),
    "cambiamento della Chiesa": (r"riforma della (chiesa|curia)|sinodal|conversione (pastorale|missionaria)|chiesa in uscita",
                                 "il cambiamento e la riforma della Chiesa, la sinodalità, la conversione missionaria"),
}
PAPI = ["Papa Giovanni Paolo II", "Papa Benedetto XVI", "Papa Francesco", "Papa Leone XIV"]
AB = {"Papa Giovanni Paolo II": "GP2", "Papa Benedetto XVI": "BXVI",
      "Papa Francesco": "FRA", "Papa Leone XIV": "LEO"}


def main() -> int:
    print("Carico i chunk dall'indice (vettori + testo)...")
    tab = lancedb.connect(str(ROOT / "indice")).open_table("chunk")
    t = tab.search().limit(10 ** 9).to_arrow()
    V = np.stack(t.column("vector").to_pylist()).astype("float32")  # (N, dim) normalizzati
    testi = t.column("testo").to_pylist()
    urls = t.column("url").to_pylist()
    papi = t.column("papa").to_pylist()
    # raggruppa i chunk per documento
    doc_chunks: dict = collections.defaultdict(list)
    doc_papa: dict = {}
    for i, u in enumerate(urls):
        doc_chunks[u].append(i)
        doc_papa[u] = papi[i]
    docs = list(doc_chunks)
    tot = collections.Counter(doc_papa[u] for u in docs)
    print(f"{len(docs)} documenti, {len(testi)} chunk. Documenti per Papa: "
          + ", ".join(f"{AB[p]}={tot[p]}" for p in PAPI) + "\n")

    emb = Embedder()
    for tema, (rx_str, concetto) in TEMI.items():
        rx = re.compile(rx_str, re.I)
        # --- regex: doc positivo se un suo chunk matcha ---
        reg = {u for u in docs if any(rx.search(testi[i]) for i in doc_chunks[u])}
        N = len(reg)
        # --- semantico: max similarità dei chunk del doc al concetto ---
        q = emb.query(concetto).astype("float32")
        sims = V @ q
        doc_sim = {u: max(sims[i] for i in doc_chunks[u]) for u in docs}
        sem = set(sorted(docs, key=lambda u: -doc_sim[u])[:N]) if N else set()
        # --- per Papa ---
        creg = collections.Counter(doc_papa[u] for u in reg)
        csem = collections.Counter(doc_papa[u] for u in sem)
        print(f"### {tema}   (regex: {N} doc; sovrapposizione regex∩sem: {len(reg & sem)})")
        print(f"  {'':9}" + "".join(f"{AB[p]:>7}" for p in PAPI))
        print(f"  {'regex':9}" + "".join(f"{100*creg[p]/tot[p]:6.1f}%" for p in PAPI))
        print(f"  {'semantico':9}" + "".join(f"{100*csem[p]/tot[p]:6.1f}%" for p in PAPI))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
