# vectordatabase-documenti-papi

Costruisce un **vector database arricchito** a partire dal corpus dei documenti
dei Papi. Legge i markdown grezzi prodotti da
[ingestion-documenti-papi](https://github.com/martinocrippa/ingestion-documenti-papi),
li **arricchisce** (embeddings, topic, entità, sentiment, frame morali/valoriali)
e li indicizza per **ricerca semantica e RAG**.

**Solo codice rilanciabile.** I dati raw non sono inclusi (rigenerabili e sotto
copyright): vedi [`note.txt`](note.txt).

## Posto nel progetto

```
ingestion-documenti-papi   →  scarica i documenti raw (markdown)
vectordatabase-...         →  arricchisce e indicizza (questo repo)
analisi & esposizione      →  trend, confronti, dashboard (futuro)
```

## Come iniziare

```bash
git clone https://github.com/martinocrippa/vectordatabase-documenti-papi.git
cd vectordatabase-documenti-papi
```

1. Procurati i dati raw con `ingestion-documenti-papi` (`python papi.py`).
2. Copia/sposta la cartella `data/` qui dentro (resta locale, è in `.gitignore`).
3. Esegui il codice di arricchimento/indicizzazione (in arrivo).

La struttura attesa di `data/` è mostrata nella cartella
[`esempio/`](esempio/) con un documento finto.

## Dati e risorse (locali)

- `data/` — corpus markdown (~25.000 file). Non versionato.
- `resources/` — materiali di ricerca (review tecniche, paper, risultati
  preliminari). Non versionato.

## Stato

Scaffold iniziale. Il codice di arricchimento (embeddings, topic modeling,
indicizzazione vettoriale) è in sviluppo.
