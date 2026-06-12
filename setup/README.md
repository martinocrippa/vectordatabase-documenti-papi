# Setup dell'ambiente

Il vector database richiede **Python ≥ 3.9** (consigliato 3.12) e le librerie
`sentence-transformers` (porta con sé `torch`), `numpy`, `rank-bm25`.

> Parte di un progetto in tre repository indipendenti (ingestion → vector
> database → text mining): ognuno ha il proprio `setup/`, da installare a sé.

## Opzione A — conda (consigliata)

```bash
conda env create -f setup/environment.yml
conda activate vectordb-papi
```

Per aggiornarlo dopo una modifica al file:

```bash
conda env update -f setup/environment.yml --prune
```

## Opzione B — venv + pip

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux/macOS:    source .venv/bin/activate
pip install -r requirements.txt
```

## Verifica e uso

Con l'ambiente **attivo** (`python` = Python 3 dell'ambiente):

```bash
python -c "import sentence_transformers, numpy, rank_bm25; print('ok')"  # librerie
python vdb.py build --per-papa 50          # indice da un campione (prova veloce)
python vdb.py search "custodire il creato" --papa francesco
```

> Nota: al **primo avvio** `sentence-transformers` scarica il modello di
> embedding (`multilingual-e5-base`, qualche centinaio di MB) nella cache di
> Hugging Face; le esecuzioni successive sono offline.
>
> I dati (`data/`) e l'indice generato (`indice/`) restano locali e non
> versionati — vedi [`../note.txt`](../note.txt).
