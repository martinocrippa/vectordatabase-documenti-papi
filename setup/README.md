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

> I dati (`data/`) e l'indice generato (`indice/`) restano locali e non
> versionati — vedi [`../note.txt`](../note.txt).

## Dove finiscono i pesi del modello

Al **primo avvio** `sentence-transformers` scarica `multilingual-e5-base`
(qualche centinaio di MB) da Hugging Face. `vdb.py` imposta `HF_HOME` su
**`models/` dentro il repo** (in `.gitignore`), così i pesi stanno "con il
progetto". Lo scarica **una volta sola** e poi lavora offline; la libreria fa
già "se c'è usalo, altrimenti scaricalo", quindi **non serve gestire il download
a mano**.

Vuoi invece la **cache utente globale** (un solo download condiviso fra tutti i
progetti della macchina)? Imposta `HF_HOME` a mano **prima** di lanciare
`vdb.py` — `vdb.py` usa `setdefault`, quindi rispetta il tuo valore:

```bash
# Linux/macOS
export HF_HOME="$HOME/.cache/huggingface"
# Windows PowerShell
$env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
```

In entrambi i casi il meccanismo "check-se-c'è-altrimenti-scarica" è quello
della libreria: noi scegliamo solo *dove*.
