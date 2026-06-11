# Piano di sviluppo

Uno stadio per volta. Ogni stadio è **piccolo, testabile e committabile** da
solo: alla fine di ciascuno c'è qualcosa che funziona davvero, non un pezzo a
metà. Si va avanti solo quando lo stadio precedente regge. Le primitive citate
sono quelle descritte in [`architettura.md`](architettura.md).

## Stadio 0 — Scaffolding ✅

- [x] `doc/` con sintesi del corpus, architettura e questo piano.
- [x] `.gitignore` che esclude `data/`, `resources/`, `indice/`.
- [ ] `requirements.txt` (`sentence-transformers`, `numpy`) e
  `setup/environment.yml`, sullo stile del repo di ingestion.

**Risultato:** l'ambiente si installa e la direzione è scritta.

## Stadio 1 — Leggere e spezzare (`documenti`, `pezzi`)

- [ ] `documenti(data_dir)`: scorre `data/**.md`, separa frontmatter YAML e
  corpo, restituisce un dict per documento (parsing del frontmatter a mano, è
  banale: niente nuove dipendenze).
- [ ] `pezzi(documento)`: spezza il corpo in chunk di ~200–400 parole con piccola
  sovrapposizione, taglia preferibilmente sui confini di paragrafo, propaga i
  metadati a ogni chunk.
- [ ] Test sull'`esempio/` (e, se presente, su `data/`).

**Risultato:** da `data/` si ottiene un flusso di chunk con metadati. Nessun
modello ancora: si verifica solo che il testo entri e si spezzi bene.

## Stadio 2 — Embeddings e indice (`Embedder`, `Indice`, `costruisci`)

- [ ] `Embedder`: carica il modello locale (costante in cima al file),
  `encode(testi)` → matrice `float32` normalizzata.
- [ ] `Indice`: `aggiungi(vettori, meta)`, `salva()` → `indice/vettori.npy` +
  `indice/meta.jsonl`, `carica()`.
- [ ] `costruisci(data_dir)`: orchestrazione `documenti → pezzi → encode → salva`,
  a batch, con un minimo di progress a video.

**Risultato:** `python -m vdb build` produce `indice/` dall'intero corpus.
Da fare a batch per non saturare la RAM; misurare tempi e dimensioni reali.

## Stadio 3 — Ricerca semantica (`cerca` + CLI) 🎯

- [ ] `Indice.cerca(vettore, k)`: prodotto matrice-vettore + top-k con
  `numpy.argpartition`.
- [ ] `cerca(query, k, filtri)`: carica indice, embedda la query, applica i
  **filtri per metadato** (`--papa`, `--tipo`, intervallo di date), restituisce i
  chunk migliori con punteggio, titolo e `url`.
- [ ] CLI `python -m vdb search "..."` con i filtri.

**Risultato e traguardo della v1:** si fa una domanda al corpus e si ottengono i
passaggi più pertinenti, filtrabili per Papa/tipologia/periodo. **Le prime
domande del progetto trovano risposta sui dati** (es. "chi parla di 'casa
comune' e quando?" — che il conteggio di parole chiave non sapeva cogliere).

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

## Stadio 5 — Oltre la v1 (quando servirà davvero)

Da affrontare **solo se e quando un bisogno reale lo chiede**, non in anticipo:

- **Indice approssimato** (`hnswlib`/`faiss`) se la ricerca lineare diventa
  lenta — cambiamento isolato dietro `Indice.cerca`.
- **Arricchimento** (topic, entità, sentiment, frame valoriali) come stadio a
  parte che aggiunge campi ai metadati, senza toccare la ricerca di base.
- **Esposizione** (confronti tra pontificati, trend nel tempo, dashboard): vive
  nel repo di analisi, alimentato da questo indice.

## Filo conduttore

Ogni stadio aggiunge **una** capacità e lascia il codice ancora leggibile in una
seduta. Se uno stadio comincia a richiedere classi, configurazioni o dipendenze
che non stavano nel piano, è il segnale di fermarsi e chiedersi se serve davvero
— prima di scriverlo.
