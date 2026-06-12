#!/usr/bin/env python3
"""Esperimento: marcare i documenti che parlano di AMBIENTE con gli embedding,
invece che con le regex su radici di parola (come faceva check_dati.py).

Tesi da verificare: il significato batte le stringhe. Documenti che parlano di
ambiente come "casa comune" / "custodia del creato" NON contengono le radici
`ambient`/`ecolog`, quindi la regex li perde; gli embedding no.

Uso (con sentence-transformers gia' installato):
    python prove/ambiente_semantico.py            # campione di default
    python prove/ambiente_semantico.py 400         # 400 doc per Papa

NON e' la pipeline finale: e' un assaggio. Niente indice salvato, niente CLI.
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# --- parametri (poche costanti, KISS) ---------------------------------------
# e5 e' un modello da RETRIEVAL: vuole i prefissi "query:"/"passage:" e le sue
# similarita' coseno stanno molto piu' in alto (relevant ~0.84+, sfondo ~0.78).
MODELLO = "intfloat/multilingual-e5-base"
PREF_Q, PREF_P = "query: ", "passage: "
CONCETTO = ("La cura dell'ambiente e dell'ecologia, la custodia del creato, "
            "la salvaguardia della casa comune e della natura.")
REGEX_AMBIENTE = re.compile(r"ambient|creato|ecolog", re.I)   # come check_dati.py
PAROLE_CHUNK = 200
PER_PAPA = int(sys.argv[1]) if len(sys.argv) > 1 else 200      # campione/Papa
SOGLIA = 0.84                                                  # sim coseno (e5)


def corpo(testo: str) -> str:
    """Testo senza il frontmatter YAML (tra i primi due '---')."""
    parti = testo.split("---", 2)
    return parti[2] if len(parti) == 3 else testo


def chunk(testo: str, n: int = PAROLE_CHUNK) -> list[str]:
    """Spezza in finestre da ~n parole (esperimento: niente overlap)."""
    p = testo.split()
    return [" ".join(p[i:i + n]) for i in range(0, len(p), n)] or [""]


def campiona(root: pathlib.Path, per_papa: int) -> list[dict]:
    """Prende ~per_papa documenti per Papa, sparsi su TUTTE le tipologie/anni.

    Campionamento a passo costante sulla lista ordinata: cosi' non si pesca solo
    la prima cartella in ordine alfabetico (angelus), ma un po' di tutto.
    """
    docs = []
    for papa_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        tutti = sorted(papa_dir.rglob("*.md"))
        passo = max(1, len(tutti) // per_papa)
        files = tutti[::passo][:per_papa]
        for f in files:
            t = f.read_text(encoding="utf-8", errors="ignore")
            tit = re.search(r"^titolo:\s*\"?(.+?)\"?\s*$", t, re.M)
            docs.append({
                "papa": papa_dir.name,
                "titolo": tit.group(1) if tit else f.stem,
                "testo": corpo(t),
            })
    return docs


def main() -> int:
    root = pathlib.Path("data")
    if not root.is_dir():
        print("Cartella data/ assente: copiala qui dal repo di ingestion.")
        return 1

    print(f"Carico il modello {MODELLO} ...")
    model = SentenceTransformer(MODELLO)
    q = model.encode(PREF_Q + CONCETTO, normalize_embeddings=True)

    docs = campiona(root, PER_PAPA)
    print(f"Campione: {len(docs)} documenti. Calcolo gli embedding...")

    # un doc "parla di ambiente" (semantico) se UN suo chunk e' vicino al concetto
    for d in docs:
        pezzi = chunk(d["testo"])
        emb = model.encode([PREF_P + p for p in pezzi], normalize_embeddings=True,
                           batch_size=64, show_progress_bar=False)
        sim = emb @ q
        j = int(np.argmax(sim))
        d["sim"] = float(sim[j])
        d["snippet"] = pezzi[j][:180]
        d["regex"] = bool(REGEX_AMBIENTE.search(d["testo"]))
        d["sem"] = d["sim"] >= SOGLIA

    # --- confronto regex vs semantico, per Papa ---
    print(f"\n% documenti 'su ambiente' — regex vs semantico (soglia {SOGLIA})")
    papi = sorted({d["papa"] for d in docs})
    for p in papi:
        g = [d for d in docs if d["papa"] == p]
        r = 100 * sum(x["regex"] for x in g) / len(g)
        s = 100 * sum(x["sem"] for x in g) / len(g)
        print(f"  {p:18} regex {r:4.0f}%   semantico {s:4.0f}%   ({len(g)} doc)")

    # --- il guadagno di recall: presi dal semantico, persi dalla regex ---
    persi = sorted((d for d in docs if d["sem"] and not d["regex"]),
                   key=lambda d: -d["sim"])
    print(f"\nDocumenti che la REGEX PERDE e il semantico trova ({len(persi)}):")
    for d in persi[:12]:
        print(f"  [{d['sim']:.2f}] {d['papa']:14} {d['titolo'][:60]}")
        print(f"         …{d['snippet']}…")

    # --- falsi positivi della regex: match di stringa ma lontani dal tema ---
    falsi = sorted((d for d in docs if d["regex"] and not d["sem"]),
                   key=lambda d: d["sim"])
    print(f"\nMatch della REGEX ma lontani dal tema (es. 'creato' = part. passato) "
          f"({len(falsi)}):")
    for d in falsi[:6]:
        print(f"  [{d['sim']:.2f}] {d['papa']:14} {d['titolo'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
