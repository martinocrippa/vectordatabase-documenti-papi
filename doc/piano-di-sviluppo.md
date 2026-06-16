# Piano di sviluppo

Uno stadio per volta. Ogni stadio è **piccolo, testabile e committabile** da
solo: alla fine di ciascuno c'è qualcosa che funziona davvero, non un pezzo a
metà. Si va avanti solo quando lo stadio precedente regge. Le primitive citate
sono quelle descritte in [`architettura.md`](architettura.md).

## Stadio 0 — Scaffolding ✅

- [x] `doc/` con sintesi del corpus, architettura e questo piano.
- [x] `.gitignore` che esclude `data/`, `resources/`, `indice/`.
- [x] `requirements.txt` (`sentence-transformers`, `numpy`, `rank-bm25`).
- [x] `setup/environment.yml`, sullo stile del repo di ingestion.

**Risultato:** l'ambiente si installa e la direzione è scritta.

## Stadio 1 — Leggere e spezzare (`documenti`, `pezzi`)

- [x] `documenti(data_dir)`: scorre `data/**.md`, separa frontmatter YAML e
  corpo, restituisce un dict per documento (parsing a mano, niente dipendenze).
- [~] `pezzi(documento)`: per ora chunking **semplice** a finestra di ~180
  parole. Come misura interim contro il rumore multilingue, ogni chunk è
  **etichettato con la lingua** e i saluti tradotti sono marcati escludibili.
  **Resta** il chunking **strutturale** per tipologia (sezioni numerate,
  separatori, coda saluti) della [mappa](mappa-tipologie.md), che toglierà anche
  l'intestazione e i saluti *italiani*.
- [ ] Test sull'`esempio/` (e, se presente, su `data/`).

**Risultato:** da `data/` si ottiene un flusso di chunk con metadati (compresa la
lingua). Il chunking strutturale è il prossimo affinamento di qualità.

## Stadio 2 — Embeddings e indice (`Embedder`, `Indice`, `costruisci`)

- [x] `Embedder`: carica il modello locale (e5, costante in cima al file),
  `passaggi`/`query` → matrice `float32` normalizzata.
- [x] `Indice`: `salva()` → `indice/vettori.npy` + `bm25.pkl` + `meta.jsonl`,
  `carica()`, `per_vettore`/`per_keyword`.
- [x] `costruisci(data_dir)`: `documenti → pezzi → encode + BM25 → salva`, con
  progress a video. **Resumabile**: cache embedding (`models/emb_*.pkl`) +
  checkpoint chunk (`meta.prepared.jsonl`), così un build interrotto riprende.

**Risultato:** `python vdb.py build` produce `indice/`; `--per-papa N` campiona
per le prove. Sull'intero corpus richiede **ore su CPU**: il build completo è
stato fatto a rilanci (resumabile); finora indicizzato **~83% del corpus**
(~147k chunk su ~175k). ⚠️ Lezione: la persistenza fatta a mano (cache da
~450 MB ricaricata ogni volta, checkpoint) è fragile a questa scala → è il caso
per cui si passa a **LanceDB** (vedi Stadio 6 e [`scelta-store.md`](scelta-store.md)).

## Stadio 3 — Ricerca ibrida (`cerca` + CLI) 🎯

Schema deciso dopo l'esperimento: **vettori + BM25 + RRF (+ rerank)**, niente soglie
(vedi [architettura](architettura.md#ricerca-ibrida-vettori--bm25--reranking)).

- [x] `Indice.per_vettore(q, k)`: coseno (prodotto matrice-vettore) + top-k.
- [x] `Indice.per_keyword(q, k)`: BM25 sui testi dei chunk (`rank-bm25`).
- [x] `cerca(query, k, filtri)`: fonde i due ranking con **RRF**, esclude i
  saluti tradotti, filtri `--papa`/`--tipo`/`--lingua`, **dedup per documento**,
  restituisce i chunk con titolo e `url`.
- [x] CLI `python vdb.py search "..."` con i filtri.
- [ ] **Rerank** con cross-encoder (il passo che alza di più la qualità).

**Risultato e traguardo della v1:** si fa una domanda al corpus e si ottengono i
passaggi più pertinenti, filtrabili per Papa/tipologia/periodo. **Le prime
domande del progetto trovano risposta sui dati** (es. "chi parla di 'casa
comune' e quando?" — che il conteggio di parole chiave non sapeva cogliere).

> ✅ **Implementato in [`../vdb.py`](../vdb.py)**: indice persistente (vettori +
> BM25 + meta) e ricerca **ibrida** con RRF, etichetta di **lingua** per chunk
> (saluti tradotti esclusi di default), **dedup per documento**, filtri
> Papa/tipologia/lingua, CLI `build`/`search`. **Restano:** chunking strutturale
> per tipologia ([mappa-tipologie](mappa-tipologie.md)) per togliere intestazioni
> e saluti italiani, **reranking** col cross-encoder, `build` incrementale.

## Stadio 4 — Risposta RAG (`ask`) — opzionale

- [ ] `ask(domanda)`: recupera i top-k chunk, costruisce un prompt con i
  passaggi e le fonti, chiede a un LLM una risposta **ancorata e citata**.
- [ ] CLI `python -m vdb ask "..."`.

> **Nota copyright/account.** A differenza degli stadi 1–3 (tutto locale), il RAG
> manda **spezzoni di testo** a un servizio LLM esterno. Per questo è opzionale e
> separato. Se attivato: usare **chiave/account personale**, mandare solo i
> passaggi necessari (non interi documenti), e ricordare che restano testi ©
> Libreria Editrice Vaticana — uso personale e di studio, con citazione della
> fonte. In alternativa si può usare un modello locale e non far uscire nulla.

## Stadio 5 — Arricchimento (topic, entità, sentiment, frame)

Dopo che la ricerca della v1 funziona, si arricchisce il corpus con campi
analitici. **Principio KISS che tiene tutto in riga:** ogni arricchimento è un
modulo indipendente che **legge i documenti/chunk e aggiunge campi a
`meta.jsonl`**, senza toccare né l'indice vettoriale né la ricerca di base. Si
fa uno alla volta, ognuno committabile da solo; se uno non serve, si salta
senza conseguenze sugli altri. La primitiva nuova è una sola:

```
arricchisci(nome, fn)   # applica fn a ogni documento e scrive i campi in meta.jsonl
```

I confronti tra Papi e nel tempo nascono poi **filtrando e raggruppando** su
questi campi, esattamente come i filtri `--papa/--tipo` della ricerca. Il primo
arricchimento — la **lingua** del chunk — è già in `vdb.py` (vedi Stadio 1).

- [ ] **5a — Topic modeling.** Raggruppa i documenti per tema *emergente* (non
  per parole chiave fissate a mano: era il limite del `check_dati.py`
  dell'ingestion). Approccio KISS e locale: clustering sugli embeddings già
  calcolati (es. BERTopic, o k-means + parole rappresentative per cluster).
  Campo aggiunto: `topic`. Riusa i vettori dell'indice, nessun ricalcolo.
- [ ] **5b — Entità (NER).** Persone, luoghi, organizzazioni citati. Locale con
  un modello italiano (es. spaCy `it_core_news_lg`). Campi: `entita` (liste).
  Utile per "chi/cosa nomina un Papa e con che frequenza".
- [ ] **5c — Sentiment / tono.** Tono del testo (es. consolatorio, esortativo,
  di denuncia) più che il semplice positivo/negativo, poco adatto a testi
  pastorali. Si parte da un classificatore italiano locale e si valuta sui dati.
  Campo: `sentiment` / `tono`.
- [ ] **5d — Frame morali/valoriali.** I valori richiamati (es. cura, giustizia,
  autorità, sacralità — sullo stile della Moral Foundations Theory). Più
  esplorativo: probabile uso di un LLM con prompt mirato. Campo: `frame`.

> **Copyright/account, di nuovo.** 5a–5c possono restare **interamente locali**
> (nessun testo esce). 5d, se usa un LLM esterno, segue le stesse regole dello
> Stadio 4: account/chiave **personale**, solo i passaggi necessari, testi ©
> LEV per uso personale e di studio. In alternativa, modello locale.

## Stadio 6 — Oltre

- [x] **Migrazione a LanceDB** (fatta, vedi [`scelta-store.md`](scelta-store.md)).
  `Indice` (vettori.npy/bm25.pkl/RRF) e la cache embedding **rimossi**: lo store
  è una tabella LanceDB on-disk, ibrida vettori+full-text+RRF nativa, `add()`
  incrementale, **ricerca flat** (niente ANN). ~Metà del plumbing in meno.
- [x] **Query apposita `primo-saluto`**: i primi saluti "Urbi et Orbi" dalla
  loggia dopo l'elezione (uno per Papa), trovati per titolo.
- [ ] **Reranking** col cross-encoder (vedi Stadio 3).
- [ ] **Indice approssimato** (l'IVF-PQ di LanceDB, o `hnswlib`/`faiss`) solo se
  la ricerca diventa lenta davvero: a 175k chunk il brute-force basta ancora, e
  l'ANN aggiunge tuning e recall approssimato in cambio di niente (vedi la scelta
  esplicita in [`scelta-store.md`](scelta-store.md)).
- [ ] **Esposizione** (confronti tra pontificati, trend nel tempo, dashboard):
  vive nel repo di analisi, alimentato da questo indice arricchito.

## Filo conduttore

Ogni stadio aggiunge **una** capacità e lascia il codice ancora leggibile in una
seduta. Se uno stadio comincia a richiedere classi, configurazioni o dipendenze
che non stavano nel piano, è il segnale di fermarsi e chiedersi se serve davvero
— prima di scriverlo.
