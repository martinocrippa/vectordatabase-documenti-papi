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
- [ ] `pezzi(documento)`: chunking **ibrido struttura+overlap** secondo la
  [mappa delle tipologie](mappa-tipologie.md) — taglia prima sui confini
  strutturali (sezioni numerate, separatori, coda multilingua), impacchetta fino
  a ~200–400 parole, overlap solo dentro la stessa sezione; propaga i metadati e
  l'etichetta di sezione a ogni chunk.
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
questi campi, esattamente come i filtri `--papa/--tipo` della ricerca.

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

## Stadio 6 — Oltre (quando servirà davvero)

Da affrontare **solo se e quando un bisogno reale lo chiede**, non in anticipo:

- **Indice approssimato** (`hnswlib`/`faiss`) se la ricerca lineare diventa
  lenta — cambiamento isolato dietro `Indice.cerca`.
- **Esposizione** (confronti tra pontificati, trend nel tempo, dashboard): vive
  nel repo di analisi, alimentato da questo indice arricchito.

## Filo conduttore

Ogni stadio aggiunge **una** capacità e lascia il codice ancora leggibile in una
seduta. Se uno stadio comincia a richiedere classi, configurazioni o dipendenze
che non stavano nel piano, è il segnale di fermarsi e chiedersi se serve davvero
— prima di scriverlo.
