# vectordatabase-documenti-papi

Costruisce un **vector database** a partire dal corpus dei documenti dei Papi.
Legge i markdown grezzi prodotti da
[ingestion-documenti-papi](https://github.com/martinocrippa/ingestion-documenti-papi),
li spezza in pezzi, li trasforma in **embeddings** e li indicizza per **ricerca
semantica e RAG** (fare domande al corpus e avere risposte ancorate ai testi).

**Solo codice rilanciabile.** I dati raw non sono inclusi (rigenerabili e sotto
copyright): vedi [`note.txt`](note.txt).

## Da dove nasce

Stessa origine del repo di ingestion: una domanda grande tra amici che leggono i
giornali. Di che cosa parlano *davvero* i Papi? C'è continuità tra un
pontificato e l'altro? Quando un titolo riassume un Papa in una parola — i
migranti, l'ambiente — è il Papa o è il giornale? Discorsi che scivolano in
fretta sul senso della vita e sulla fede, e la voglia di **verificarli sui dati**
invece di tirare a indovinare.

Il repo di ingestion ha raccolto i testi (~25.000 documenti, decenni di
parole). Ma un conteggio di parole chiave non basta: non capisce che "casa
comune" parla di ambiente, o che due Papi dicono la stessa cosa con parole
diverse. Serve cercare **per significato**, non per stringhe. È quello che fa
questo repo: trasforma i testi in vettori e permette di interrogarli per senso —
il gradino che rende possibili i confronti veri tra pontificati.

> 📂 Il disegno completo, il piano di sviluppo e la sintesi del corpus sono in
> **[`doc/`](doc/)**.

## Posto nel progetto

```
ingestion-documenti-papi   →  scarica i documenti raw (markdown)
vectordatabase-...         →  spezza, fa gli embeddings, indicizza, cerca (questo repo)
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

## Documentazione

In [`doc/`](doc/):

- [`doc/sintesi-corpus.md`](doc/sintesi-corpus.md) — che cosa c'è nel corpus
  (conteggi, periodi, dimensioni) descritto **senza riprodurre i testi**, nel
  rispetto del copyright.
- [`doc/architettura.md`](doc/architettura.md) — il disegno tecnico: poche
  primitive, dipendenze minime, tutto locale.
- [`doc/mappa-tipologie.md`](doc/mappa-tipologie.md) — struttura dei documenti
  per tipologia e strategia di chunking ibrido (struttura + overlap).
- [`doc/piano-di-sviluppo.md`](doc/piano-di-sviluppo.md) — gli stadi di sviluppo,
  uno per volta.

## Stato

Prima versione funzionante in [`vdb.py`](vdb.py): `build` costruisce l'indice
(embeddings e5 + BM25, salvato in `indice/`) e `search` fa la **ricerca ibrida**
(vettori + BM25 fusi con RRF) con filtri per Papa/tipologia.

```bash
pip install -r requirements.txt
python vdb.py build --per-papa 50          # indice da un campione (prova veloce)
python vdb.py search "custodire il creato" --papa francesco
```

Da fare (vedi piano): chunking strutturale per tipologia, reranking, `build`
incrementale sull'intero corpus.
