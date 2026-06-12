# vectordatabase-documenti-papi

Costruisce un **vector database** a partire dal corpus dei documenti dei Papi.
Legge i markdown grezzi prodotti da
[ingestion-documenti-papi](https://github.com/martinocrippa/ingestion-documenti-papi),
li spezza in pezzi, li trasforma in **embeddings** + indice **BM25**, e li
interroga con **ricerca ibrida** (significato + parole, fusi con RRF) per fare
domande al corpus e avere passaggi pertinenti, ancorati ai testi.

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
(vettori + BM25 fusi con RRF). Inoltre: ogni chunk ha la **lingua**, i saluti
tradotti sono **esclusi di default**, i risultati sono **deduplicati per
documento**, e ci sono filtri per Papa, tipologia e lingua.

```bash
pip install -r requirements.txt            # oppure conda: vedi setup/README.md
python vdb.py build --per-papa 50          # indice da un campione (prova veloce)
python vdb.py search "custodire il creato" --papa francesco
python vdb.py search "lo sport e gli atleti" --lingua it   # solo italiano
```

Dettagli sull'ambiente (conda/venv) in [setup/README.md](setup/README.md).

### Il modello di embedding (Hugging Face) — gratis e locale

Al **primo** `build`/`search`, `sentence-transformers` scarica il modello
`multilingual-e5-base` (qualche centinaio di MB) da **Hugging Face Hub**. È solo
un download di file, **gratuito**: nessuna API, nessun account, nessun costo. Il
modello gira poi **in locale** sulla CPU e dalle volte successive è **offline**.

I pesi vanno in **`models/` dentro il repo** (gitignored): `vdb.py` imposta
`HF_HOME` lì, così i pesi stanno "con il progetto". Si scaricano **una volta
sola**; dalle volte dopo la libreria fa da sé "se c'è usalo, altrimenti
scaricalo", quindi non serve gestire il download a mano. Il warning
`Please set a HF_TOKEN…` è innocuo (solo rate-limit) e si può ignorare.

> Per usare invece la **cache utente globale** (condivisa fra progetti), imposta
> `HF_HOME` a mano prima di lanciare `vdb.py`. Vedi
> [setup/README.md](setup/README.md).

Da fare (vedi piano): chunking strutturale per tipologia, reranking, `build`
incrementale sull'intero corpus.
