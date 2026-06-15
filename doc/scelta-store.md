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

## Valutazione (vector DB, giugno 2026)

Criteri nostri: **locale/embedded** (niente server), **ricerca ibrida**
vettori + BM25 + RRF, scala ~175k e in crescita, build **incrementale**, **poca
RAM**, dati **in locale** (copyright).

| | LanceDB | sqlite-vec | Chroma | Qdrant |
|---|---|---|---|---|
| embedded, niente server | sì | sì | sì | nato server (ha embedded) |
| ibrido vettori+BM25+RRF | **nativo** (FTS Tantivy + reranker RRF) | sì, FTS5+RRF a mano | debole | ottimo, ma server |
| on-disk, larger-than-memory | sì (Lance, memory-mapped) | full table scan | più in RAM | sì |
| build incrementale | sì (`add`) | sì | sì | sì |
| porti i tuoi embedding (e5) | sì | sì | sì | sì |

## Decisione: **LanceDB**

Fa nativamente ciò che abbiamo scritto a mano — vettori + BM25 + **RRF** in un
solo store — e risolve la persistenza/memoria del build: on-disk memory-mapped,
`add()` incrementale e persistente per costruzione. **Non** accelera l'embedding
(per quello servirebbe GPU o un modello più piccolo), ma toglie tutta la
fragilità di salvataggio/ripresa.

Alternative: *sqlite-vec* è il piano B minimalista (a 175k il full-scan va bene);
*Chroma* è il più semplice ma debole sull'ibrido (che è il cuore del disegno);
*Qdrant* è ottimo ma è infrastruttura da server. *MongoDB Atlas* è cloud → i
testi uscirebbero (copyright), escluso.

## Cosa cambia in `vdb.py` (migrazione, prossimo passo)

Resta KISS: cambia solo il "dietro" di `Indice`.
- **Si elimina**: la cache embedding (`models/emb_*.pkl`), il checkpoint
  `meta.prepared.jsonl`, `bm25.pkl`/`vettori.npy`, e `_rrf`/`per_vettore`/
  `per_keyword`. È ~metà del plumbing che ci ha fatto soffrire.
- **`costruisci`**: invece di accumulare in RAM, `table.add(batch)` documento per
  documento → persistente e ripartibile *per costruzione*.
- **`cerca`**: `table.search(...).query_type("hybrid")` con RRF + filtri sui
  metadati.
- **Restano invariati**: `documenti`, `_segmenta`, `pezzi`, `_lingua`, `Embedder`
  (l'embedding e5 lo calcoliamo comunque noi).

> In una riga: i puntelli attuali (cache, prepared, checkpoint) sono **interim**,
> nati per finire il build su CPU. La direzione è sostituirli con LanceDB, che li
> rende inutili.

## Fonti

- [LanceDB — repo](https://github.com/lancedb/lancedb) ·
  [hybrid search](https://docs.lancedb.com/search/hybrid-search)
- [Confronto vector DB 2026 (4xxi)](https://4xxi.com/articles/vector-database-comparison/) ·
  [Best vector DBs 2026 (Encore)](https://encore.dev/articles/best-vector-databases)
- [Hybrid BM25+vettori con sqlite-vec/FTS5+RRF](https://github.com/liamca/sqlite-hybrid-search)
