# Scelta dello store (decisione)

Breve nota di decisione (ADR) su *dove e come* tenere vettori + testo + metadati.

## Contesto

`vdb.py` oggi tiene l'indice a mano: `indice/vettori.npy` (matrice NumPy),
`indice/bm25.pkl` (BM25), `indice/meta.jsonl` (metadati) e, per reggere il build
su CPU, due puntelli aggiunti **sotto pressione**: una **cache embedding** in
`models/emb_*.pkl` e un **checkpoint dei chunk** in `indice/meta.prepared.jsonl`.

Costruire l'indice sull'**intero corpus** (~25.000 documenti → ~175.000 chunk)
ha messo a nudo i limiti di questo fai-da-te:
- l'embedding su CPU è lungo (ore) — e **nessuno store lo accelera** (è calcolo,
  non archiviazione);
- ma la **persistenza** sì: la cache da ~450 MB veniva ricaricata interamente a
  ogni run (sempre più lenta), tenuta in RAM, ri-salvata a blocchi; il resume era
  codice con stato. Tutta idraulica che un vero store fa per noi.

## Cosa deve davvero fare lo store (il punto che orienta tutto)

Prima di guardare il mercato, vale la pena dire **che problema** stiamo
risolvendo, perché non è quello che di solito vende un "database vettoriale".

- **La ricerca non è il collo di bottiglia.** 175k vettori × 768 dim in
  `float32` ≈ 540 MB: un prodotto scalare brute-force su questa matrice è
  **millisecondi** (lo dice già [`architettura.md`](architettura.md#persistenza-stato-e-direzione--lancedb)).
  Quindi **non ci serve un indice ANN** (HNSW, IVF-PQ): è la feature di punta di
  metà dei vector DB, e per noi è irrilevante — anzi, è tuning e recall
  approssimato in cambio di niente.
- **Il dolore vero è la persistenza del build**: cache da 450 MB ricaricata ogni
  run, checkpoint con stato, ri-salvataggi. Idraulica scritta a mano e fragile.
- **Il cuore funzionale è l'ibrido vettori + BM25 + RRF**, che oggi teniamo a
  mano (`_rrf`, `per_vettore`, `per_keyword`).

Riformulato così, **non cerchiamo un motore ANN veloce**. Cerchiamo uno store
embedded che ci dia **persistenza incrementale + FTS/BM25 nativo + RRF nativo**,
così cancelliamo ~metà del plumbing. È questo il filtro con cui leggere la
tabella qui sotto, non i benchmark di velocità.

## Valutazione (vector DB, giugno 2026)

Criteri nostri: **locale/embedded** (niente server), **ricerca ibrida**
vettori + BM25 + RRF, scala ~175k e in crescita, build **incrementale**, **poca
RAM**, dati **in locale** (copyright). La velocità di ricerca **non è** un
criterio: a questa scala la risolve già il brute-force.

| | LanceDB | sqlite-vec | DuckDB (vss+fts) | Chroma | Qdrant |
|---|---|---|---|---|---|
| embedded, niente server | sì | sì (è SQLite) | sì (un file) | sì | nato server (ha embedded) |
| ibrido vettori+BM25+**RRF** | **nativo** (FTS Tantivy + reranker RRF) | FTS5 + RRF **a mano** | `match_bm25()` + vss, fusione **a mano** | debole | ottimo, ma server |
| on-disk, larger-than-memory | sì (Lance, memory-mapped) | sì, full-scan | sì, colonnare | più in RAM | sì |
| build incrementale | sì (`add`) | sì | sì | sì | sì |
| porti i tuoi embedding (e5) | sì | sì | sì | sì | sì |

I server-based (Qdrant, Milvus, pgvector) sono ottimi a scala ma a 175k su un PC
sono un demone da gestire per zero beneficio: fuori per principio. Il cloud
(Pinecone, **MongoDB Atlas**) è escluso a monte: i testi uscirebbero dalla
macchina e sono © LEV. Resta una gara a tre embedded: **LanceDB, sqlite-vec,
DuckDB**.

- **sqlite-vec** — a 175k il full-scan va benissimo (di nuovo: l'ANN non serve).
  Ma fa solo la parte vettoriale: l'ibrido lo ricostruiamo con FTS5 + RRF **a
  mano**, cioè teniamo esattamente il codice (`_rrf` ecc.) che vogliamo buttare.
  Come "una sola dipendenza" è bello; come risposta al *nostro* dolore è un mezzo
  passo. Resta il **piano B** solido se LanceDB desse problemi.
- **DuckDB (vss + fts)** — l'outsider. Un file `.duckdb`, colonnare, con `fts`
  (`match_bm25()`) e `vss` (HNSW, ma si può fare similarità brute-force senza
  indice, perfetto per noi). Vantaggio strategico: lo stadio a valle è **analisi
  & dashboard**, cioè SQL analitico — il mestiere di DuckDB. Store e analisi
  vivrebbero nello stesso motore. Ma l'ibrido resta una **fusione a mano** (niente
  reranker RRF nativo) e la persistenza dell'indice vss è ancora sperimentale:
  mantiene parte del plumbing che vogliamo togliere. **Da ri-valutare** allo
  stadio analisi, non oggi.
- **Chroma** — il più semplice da avviare, ma debole sull'ibrido (che è il cuore
  del disegno) e con corruzione note dell'HNSW su uscita brusca del processo.

## Decisione: **LanceDB**

Fa nativamente ciò che abbiamo scritto a mano — vettori + BM25 + **RRF** in un
solo store — e risolve la persistenza/memoria del build: on-disk memory-mapped,
`add()` incrementale e persistente per costruzione. **Non** accelera l'embedding
(per quello servirebbe GPU o un modello più piccolo), ma toglie tutta la
fragilità di salvataggio/ripresa. È l'unico embedded che cancella il plumbing
*invece* di tenercelo (sqlite-vec) o rifarlo a mano (DuckDB).

Due scelte esplicite che tengono la migrazione davvero KISS:

1. **Niente indice ANN — ricerca flat.** Con 175k vettori si usa il brute-force
   (recall esatto, zero tuning). L'IVF-PQ di LanceDB serve da milioni di vettori
   in su: per noi sarebbe solo complessità e recall approssimato in cambio di
   niente. L'indice approssimato si valuterà *se e quando* la ricerca diventerà
   lenta davvero (vedi [`piano-di-sviluppo.md`](piano-di-sviluppo.md), Stadio 6).
2. **Tokenizer FTS multilingue, scelto a occhi aperti.** Il corpus è
   it/en/fr/es/de/pt/pl e l'FTS (Tantivy) configura *un* tokenizer/stemmer per
   colonna. Oggi `_tok` (`[a-zàèéìòóù]+`) è già italo-centrico e imperfetto sulle
   altre lingue, quindi non è una regressione: si parte da un tokenizer base
   **senza stemming aggressivo**, per non favorire una lingua, e si valuta sui
   dati. Non è un dettaglio da nascondere in un default.

## Cosa è cambiato in `vdb.py` (migrazione: FATTA)

Resta KISS: è cambiato solo il "dietro" di `Indice`.
- **Eliminati**: la cache embedding (`models/emb_*.pkl`), `bm25.pkl`/`vettori.npy`,
  la classe `Indice` e `_rrf`/`per_vettore`/`per_keyword`. ~Metà del plumbing.
  (Resta `meta.prepared.jsonl` come checkpoint dei chunk preparati per riprendere
  il build — leggero, può sparire in seguito.)
- **`costruisci`**: `table.add(batch)` a blocchi → persistente e ripartibile *per
  costruzione* (riprende dal numero di righe già in tabella).
- **`cerca`**: `table.search(query_type="hybrid")` con `RRFReranker` + filtri sui
  metadati, **senza indice ANN** (ricerca flat). Più il comando `primo-saluto`.
- **Invariati**: `documenti`, `_segmenta`, `pezzi`, `_lingua`, `Embedder`
  (l'embedding e5 lo calcoliamo comunque noi).

> Stato: indice LanceDB popolato sull'intero corpus (~175k chunk), ricerca ibrida
> nativa, ANN spento di proposito (a 175k non serve). La dipendenza `rank-bm25` è
> uscita, è entrata `lancedb`.

## Fonti

- [LanceDB — repo](https://github.com/lancedb/lancedb) ·
  [hybrid search](https://docs.lancedb.com/search/hybrid-search)
- [Confronto vector DB 2026 (4xxi)](https://4xxi.com/articles/vector-database-comparison/) ·
  [Best vector DBs 2026 (Encore)](https://encore.dev/articles/best-vector-databases) ·
  [Embedded vector DB 2026 (shaharia.com)](https://shaharia.com/blog/choosing-embeddable-vector-database-go-application/)
- [Hybrid BM25+vettori con sqlite-vec/FTS5+RRF](https://github.com/liamca/sqlite-hybrid-search)
- [DuckDB VSS](https://duckdb.org/docs/current/core_extensions/vss) ·
  [DuckDB text analytics (fts + vss + sentence-transformers)](https://duckdb.org/2025/06/13/text-analytics)