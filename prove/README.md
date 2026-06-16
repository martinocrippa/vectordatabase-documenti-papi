# prove/ — esperimenti

Esperimenti **usa-e-impara**: codice veloce il cui scopo è rispondere a una
domanda *prima* di costruire la cosa vera. Restano qui come memoria del **perché**
delle scelte di disegno; non sono codice di produzione (quello è `../vdb.py`).

## Gli script

| File | Domanda | Risposta / lezione |
|---|---|---|
| `ambiente_semantico.py` | gli embedding battono le parole chiave nel *marcare* i documenti di un tema? | La **classificazione per soglia non funziona** (similarità non calibrate). Il guadagno non è sul recall per-documento, ma su precisione + retrieval di passaggi. |
| `cerca_passaggi.py` | la ricerca **ibrida** (vettori+BM25+RRF) trova i passaggi giusti? | Sì: il solo BM25 perde "custodire il creato", i vettori lo trovano, l'ibrido fa emergere il consenso. → confluito in `vdb.py`. |
| `esperimento_tema.py` | su un **tema** dato, confronto BM25 vs vettori vs ibrido + footprint per Papa | Parametrico: `python prove/esperimento_tema.py <tema>` (es. `calcio`, `ambiente`). Generalizza i vecchi `esperimento_sport.py`/`esperimento_calcio.py`. |

> I precursori specifici `esperimento_sport.py` / `esperimento_calcio.py` sono
> stati **sostituiti** da `esperimento_tema.py` (stesso schema, parametrico), per
> tenere `prove/` minimale.

## Il metodo (esperimento per tema)

Per avere materiale vero **senza indicizzare tutto il corpus** (sulla CPU
l'embedding è il collo di bottiglia):

1. **Raccolta mirata.** Si scorre *tutto* il corpus e si prendono i documenti che
   nominano il tema (per parola chiave) + un **campione di sfondo** (documenti
   qualsiasi, per contrasto). Così la ricerca deve davvero distinguere il tema dal
   resto.
2. **Indice in memoria.** Chunking strutturale (via `vdb._segmenta`: tolte
   intestazioni e saluti) → embedding e5 + BM25.
3. **Confronto dei tre metodi** sulla query del tema: BM25, vettori, ibrido (RRF),
   con i risultati taggati `[TEMA]`/`[sfondo]`.
4. **Sovrapposizione**: quanti documenti trovano *entrambi* i metodi, quanti
   *solo* uno → misura la complementarità (e quindi il senso dell'ibrido).
5. **Footprint per Papa**: quanti documenti del tema finiscono nei top risultati,
   per pontificato → risponde a *"ne parla solo un Papa?"*.

I saluti tradotti e l'intestazione sono esclusi (come in `vdb.py`), i risultati
sono deduplicati per documento.

## Risultati

### Calcio / football

- **Footprint** (top-10 ibridi, unione query): **Giovanni Paolo II 16 · Francesco 11**
  (Benedetto/Leone marginali). Non è tema di nessuno in esclusiva: GP2 riceveva
  molte squadre (Lazio, Bologna, Inter…), Francesco parla di Mondiali e Serie A.
- **Footprint a parola chiave** (`calcio|football|soccer`, tutto il corpus):
  Francesco 91 · Giovanni Paolo II 67 · Benedetto XVI 22 · Leone XIV 2.
- **Confronto metodi** (top-15, documenti distinti): BM25 e vettori si
  sovrappongono solo su **4** documenti; **11 solo-vettori** e **11 solo-BM25**
  (quasi tutti calcio). → I due metodi sono **complementari**: l'ibrido copre
  ciò che uno solo perderebbe. Esempio: i vettori trovano la **FIFA**
  ("Fédération") e l'**AS Roma Calcio** per significato; BM25 prende gli **arbitri
  dei Mondiali** per parola esatta.

### Ambiente / cura del creato

Domanda di partenza: *"è solo Francesco a parlarne (e un po' comunista)?"*

**Footprint a parola chiave** (termini ambientali specifici, tutto il corpus),
in % sui documenti di ciascun Papa:

| Papa | doc ambientali | % sul suo corpus |
|---|--:|--:|
| Francesco | 1.278 | 21,1% |
| Leone XIV | 134 | 21,0% |
| Giovanni Paolo II | 2.814 | 18,3% |
| Benedetto XVI | 494 | 16,5% |

→ **Tema continuo, non esclusiva di Francesco**: tutti tra il 16% e il 21%; in
assoluto il più "ambientale" è Giovanni Paolo II (2.814 doc). Quello che è
*davvero* di Francesco è la **frase** "casa comune" (430 doc, contro 91 di GP2,
28 di Leone, 18 di Benedetto): il **tema** è di tutti, il **linguaggio** della
*Laudato si'* è la sua firma.

**Confronto regex vs embedding** (sull'indice, ~20.400 doc, tema ambiente, stesso
volume di positivi per i due metodi):

| metodo | GP2 | BXVI | FRA | LEO |
|---|--:|--:|--:|--:|
| regex (parole) | 18% | 16% | 19% | 14% |
| semantico (embedding) | 18% | 14% | 20% | 14% |

→ Il **quadro per-Papa è quasi identico**: la conclusione di continuità **regge
anche a embedding**. Ma a livello di *singolo documento* i due metodi si
sovrappongono solo per ~37%: il semantico trova testi senza la parola (es. un
messaggio interreligioso), il regex prende falsi positivi (es. "ambiente" come
contesto). **È la prova del perché serve l'ibrido**: stesso aggregato, documenti
diversi.

> La domanda completa "Francesco è comunista / senza continuità?" (tutti i temi,
> %+lift su tutto il corpus) è raccontata in
> [`sintesi-per-un-amico.md`](sintesi-per-un-amico.md): continuità schiacciante,
> accenti diversi, comunismo no.

## Lezione architetturale (dagli esperimenti)

- La **ricerca** è già istantanea (carica indice + coseno = millisecondi): il
  collo di bottiglia è l'**embedding** sulla CPU, che **nessuno store accelera**.
- La **persistenza** invece sì: la cache a mano (`models/emb_*.pkl`) + i
  checkpoint hanno permesso di finire il build su CPU a rilanci, ma alla scala
  dell'intero corpus (~175k chunk, cache ~450 MB ricaricata ogni volta) è
  **fragile**.
- **Fatto: migrato a LanceDB** (store embedded on-disk, ibrido vettori+full-text+
  RRF nativo, `add()` incrementale) → rimossi cache embedding, `vettori.npy`,
  `bm25.pkl`, `_rrf`/`per_*` (~metà del plumbing). Perché e confronto in
  [`../doc/scelta-store.md`](../doc/scelta-store.md).
