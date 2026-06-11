# Architettura

Il disegno di questo repository segue lo stesso principio del repo di ingestion:
**poche primitive che si compongono, niente sovrastrutture**. Si aggiunge
complessità solo quando i dati reali la impongono, mai per principio.

## Obiettivo (e non-obiettivo)

**Obiettivo della v1:** dato il corpus markdown, poter **cercare per
significato** — fare una domanda in linguaggio naturale e riavere i passaggi
più pertinenti dei documenti, con i loro metadati (Papa, tipologia, data, url).
Questo è il mattone su cui poggiano tutti i confronti tra pontificati.

**Non-obiettivo della v1:** topic modeling, named-entity, sentiment, frame
valoriali, dashboard. Sono cose utili ma successive: tenerle fuori dalla prima
versione è una scelta, non una dimenticanza (vedi
[`piano-di-sviluppo.md`](piano-di-sviluppo.md)). La prima versione fa **una cosa
sola e bene**: embeddings + ricerca semantica.

## Le primitive

Tutto si regge su quattro mattoni, esattamente come `papi.py` nel repo di
ingestion si regge sui suoi:

| Primitiva | Cosa fa |
|---|---|
| `documenti(data_dir)` | generatore: scorre `data/**.md`, separa frontmatter e corpo, restituisce un dict per documento. |
| `pezzi(documento)` | spezza il corpo in **chunk** seguendo la struttura della tipologia (sezioni numerate/separatori/paragrafi) con overlap solo *dentro* la sezione, portandosi dietro i metadati. Ricetta per tipo in [`mappa-tipologie.md`](mappa-tipologie.md). |
| `Embedder` | incapsula il modello locale di embedding; `encode(testi) -> matrice di vettori` (normalizzati). |
| `Indice` | tiene i vettori (matrice NumPy) + i metadati paralleli; `aggiungi`, `salva`, `carica`, `cerca(vettore, k)`. |

E due funzioni di orchestrazione che le mettono in fila:

| Funzione | Cosa fa |
|---|---|
| `costruisci(data_dir)` | `documenti → pezzi → Embedder.encode → Indice.salva`. Costruisce l'indice da zero (o in modo incrementale, vedi sotto). |
| `cerca(query, k)` | carica l'indice, `Embedder.encode([query])`, `Indice.cerca`, restituisce i chunk migliori con i metadati. |

Si legge dall'alto in basso come una ricetta. Niente classi oltre alle due
necessarie (`Embedder`, `Indice`, che incapsulano stato reale: il modello e la
matrice); il resto sono funzioni pure su strutture dati elementari.

## Come è fatto l'indice (lo storage)

La scelta più semplice che regge il volume del corpus (vedi
[`sintesi-corpus.md`](sintesi-corpus.md)):

```
indice/
  vettori.npy     # matrice float32 (N_chunk × dim), vettori L2-normalizzati
  meta.jsonl      # una riga JSON per chunk, nello stesso ordine della matrice
```

- **Vettori** → un singolo array NumPy `float32`. Normalizzati a norma 1, così
  la similarità coseno è un semplice prodotto scalare.
- **Metadati** → `meta.jsonl`, una riga per chunk: `papa`, `tipologia`, `data`,
  `titolo`, `url`, indice del documento, e il **testo del chunk** (serve a
  mostrare il risultato e a costruire il contesto per il RAG).
- **Ricerca** → `vettori @ query` (un solo prodotto matrice-vettore), poi i `k`
  punteggi più alti con `numpy.argpartition`. Su qualche centinaio di migliaia
  di chunk a 384 dimensioni sono **decine di millisecondi** in RAM.

> ⚠️ **`meta.jsonl` contiene spezzoni dei testi originali**, quindi è
> **materiale sotto copyright**: l'intera cartella `indice/` è in `.gitignore`,
> esattamente come `data/`. Si rigenera in locale. Non si versiona, non si
> pubblica.

### Perché NON un database vettoriale esterno (per ora)

Niente FAISS, niente Chroma/Qdrant/pgvector, niente server. Il brute-force con
NumPy è sufficiente per questo volume e ha un costo cognitivo vicino a zero: un
file, una matrice, un prodotto. Un indice approssimato (es. `hnswlib` o `faiss`)
si aggiunge **solo quando la ricerca lineare diventa lenta davvero** — e a quel
punto sarà un cambiamento isolato dietro `Indice.cerca`, non una riscrittura.
"Aggiungi complessità quando ti fa male, non prima."

## Il modello di embedding

- **Locale, multilingue, nessuna API.** I testi non escono dalla macchina:
  questo evita sia i costi sia ogni dubbio sul mandare materiale protetto a
  servizi terzi. Tutto gira in casa.
- **Default proposto:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (384 dim, veloce, gestisce bene l'italiano, nessun prefisso da ricordare).
- **Upgrade qualità drop-in:** `intfloat/multilingual-e5-base` (768 dim, migliore
  nel retrieval; richiede i prefissi `query:` / `passage:`). Il modello è una
  **singola costante** in cima al modulo: cambiarlo è una riga.

## Interfaccia (CLI)

Una sola riga di comando, tre verbi, sullo stile di `python papi.py`:

```bash
python -m vdb build                  # costruisce indice/ da data/
python -m vdb search "cosa dice il Papa sulla pace?"      # ricerca semantica
python -m vdb search "..." --papa francesco --tipo angelus  # con filtri sui metadati
python -m vdb ask "..."              # (stadio successivo, opzionale) risposta RAG
```

I **filtri per metadato** (`--papa`, `--tipo`, intervallo di date) sono il modo
con cui si fanno i confronti tra pontificati: si cerca lo stesso significato e
si guarda chi lo dice, quando. Non serve altro per le prime domande del
progetto.

## Layout del codice

Un solo modulo, come il repo di ingestion. Si divide solo se e quando cresce.

```
vdb.py            # le primitive + la CLI (un file, struttura piatta)
indice/           # output rigenerabile, NON versionato (.gitignore)
test/
  smoke_test.py   # costruisce un mini-indice dall'esempio e fa una ricerca
requirements.txt
setup/environment.yml
```

## Dipendenze (minime)

| Pacchetto | Perché |
|---|---|
| `sentence-transformers` | il modello locale di embedding (porta con sé `torch`, `transformers`). È l'unica dipendenza pesante, ed è inevitabile per fare embedding in locale. |
| `numpy` | la matrice dei vettori e la ricerca. |

`build` e `search` non chiedono altro. Lo stadio opzionale `ask` (RAG) aggiunge
un client LLM (es. `anthropic`) usato con **chiave/account personale**; vedi le
note di copyright nel piano. Python ≥ 3.9 (consigliato 3.12), come l'ingestion.

## Stile del codice

- **PEP 8** per la formattazione, **PEP 257** per i docstring, **PEP 484** per i
  type hint dove aiutano la leggibilità (firme delle primitive).
- Funzioni piccole, una responsabilità ciascuna; nomi espliciti; commenti solo
  per i fatti non ovvi (come nel repo di ingestion: "l'header mente, è UTF-8").
- Niente configurazione prematura: le poche scelte (modello, dimensione del
  chunk, `k`) sono costanti in cima al file, cambiabili in una riga.

## Copyright e dati: la regola del repo

Vale qui come nell'ingestion ed è il vincolo che tiene tutto "senza problemi":

- **Si versiona solo il codice rilanciabile.** `data/`, `resources/` e `indice/`
  sono locali e in `.gitignore`.
- **Nessun testo reale nel repo.** Solo il documento finto in `esempio/`.
- I testi restano © Libreria Editrice Vaticana; l'uso è personale e di studio,
  con la fonte (`url`) sempre tracciata nei metadati.
