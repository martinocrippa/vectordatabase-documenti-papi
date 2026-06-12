# Mappa delle tipologie e strategia di chunking

Domanda di partenza: ogni **tipologia** di documento (che è poi un *evento* —
un Angelus, un'udienza, un'omelia…) ha una **struttura comune**? E conviene
mescolare il chunking a finestra mobile (overlap) con un chunking che segue la
**struttura già presente** nel documento?

**Risposta breve: sì e sì.** Le tipologie hanno scheletri riconoscibili, e c'è
un secondo asse altrettanto forte — il **Papa**. Sfruttare questa struttura per
tagliare i chunk migliora la ricerca. Sotto: la mappa ricavata dai dati reali e
una proposta concreta.

> Cifre ricavate dai ~25.000 file in `data/` (metadati e marcatori strutturali,
> non i testi). Fotografia al 2026-06-11.

## Lo scheletro comune a (quasi) tutti i documenti

Ogni file markdown, a prescindere dalla tipologia, ha la stessa ossatura:

```
[frontmatter YAML]            papa, tipologia, data, titolo, url, parole
# <titolo>
- **Papa:** … - **Tipologia:** … - **Data:** … - **Fonte:** …   (100% dei file)
---
<BLOCCO INTESTAZIONE IN MAIUSCOLO>     nome del Papa · tipo evento · luogo+data · (tema)
<saluto>                               "Cari fratelli e sorelle, …" (dove previsto)
<corpo>                                il contenuto vero
<coda>                                 saluti multilingua / "Dopo l'Angelus" / note
```

Il **blocco intestazione in maiuscolo** e il blocco bullet sono **metadati
travestiti da testo**: meglio tenerli fuori dai chunk di contenuto (molti sono
già nel frontmatter). La **coda multilingua** è spesso rumore per la ricerca in
italiano (vedi sotto).

## La mappa per tipologia (dati reali)

| Tipologia | N. doc | Parole (mediana) | Paragrafi | Numerati | Note `[n]` | Coda multiling. | `* * *` | "Dopo l'Angelus" |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| angelus | 2.750 | 773 | 18 | 42% | 1% | 33% | 5% | 47% |
| audiences | 2.105 | 2.140 | 41 | 58% | 4% | **92%** | 67% | — |
| homilies | 3.311 | 1.436 | 22 | 66% | 7% | 2% | 1% | — |
| speeches | 12.677 | 926 | 14 | 45% | 2% | 2% | 1% | — |
| messages | 1.747 | 885 | 15 | 33% | 7% | 1% | 0% | — |
| letters | 1.056 | 720 | 14 | 25% | 3% | 0% | 0% | — |
| cotidie | 789 | 981 | 16 | 0% | 0% | 0% | 0% | — |
| travels | 630 | **126** | 13 | 32% | 3% | 0% | 0% | — |

Letture:
- **`speeches`** è la categoria dominante (12.677 doc), corta e per lo più
  piatta: è qui che il chunking generico va bene per la maggioranza.
- **`audiences`** è il caso più strutturato: lunga (2.140 parole, 41 paragrafi),
  92% con coda multilingua, 67% con separatori `* * *`. Catechesi + appendice di
  saluti.
- **`angelus`** ha quasi sempre due parti: la riflessione e, nel 47% dei casi,
  la sezione **"Dopo l'Angelus"** (saluti, appelli, lingue diverse).
- **`cotidie`** sono sintesi giornalistiche (da *L'Osservatore Romano*): narrative
  piatte, mai numerate — un solo blocco coerente.
- **`travels`**: mediana **126 parole** → molti sono stub/pagine-indice, non
  discorsi veri. Candidati a essere filtrati o trattati a parte (vedi anomalie).

## Il secondo asse: il Papa

I paragrafi **numerati** ("1." "2." …) sono il marcatore strutturale più forte,
ma dipendono soprattutto dal pontefice, non solo dalla tipologia:

| | Giovanni Paolo II | Benedetto XVI | Francesco | Leone XIV |
|---|--:|--:|--:|--:|
| audiences numerate | **97%** | 11% | 6% | 0% |
| homilies numerate | **94%** | 0% | 6% | 0% |
| speeches numerate | **65%** | 5% | 4% | 1% |
| messages numerate | **68%** | 20% | 9% | 3% |

**Giovanni Paolo II numera quasi tutto; gli altri quasi niente.** Per i suoi
documenti la numerazione dà sezioni semantiche pronte all'uso; per gli altri
bisogna ricadere sui paragrafi. Conclusione: il chunking non può essere
*solo* type-aware, deve **adattarsi alla struttura che trova** nel singolo file.

## Gli elementi strutturali (e cosa farne)

| Elemento | Come riconoscerlo | Cosa farne |
|---|---|---|
| Blocco intestazione maiuscolo | righe dopo il `---`, prima del saluto | non embeddare come contenuto; estrarre eventuali campi (luogo, occasione) |
| Saluto | "Cari fratelli e sorelle…", "Caro…" | confine d'inizio del corpo |
| Sezione numerata | `^\s*\d{1,3}\.\s` (≥3 volte) | **unità di chunk naturale** (una per numero) |
| Separatore | `* * *` | confine duro tra blocchi |
| Coda multilingua | "Dopo l'Angelus", "Saluto", "Je salue", "I greet"… | sezione a parte, bassa priorità (vedi sotto) |
| Note a piè | `[1]`, `[2]` | staccare dal corpo, conservare come metadato |

### La coda multilingua è rumore per la ricerca in italiano

Nelle udienze (92%) e in molti Angelus la parte finale ripete lo stesso saluto
in 5-6 lingue. Per una ricerca *semantica in italiano* è ridondante e sporca i
risultati. Meglio **isolarla** in chunk dedicati con un'etichetta (`sezione:
saluti`) e, di default, **escluderla** dalla ricerca tematica (riattivabile su
richiesta).

## Proposta: chunking ibrido (struttura prima, overlap dentro)

Tre passi, esattamente come immaginavi — la struttura data dal documento più la
finestra mobile:

1. **Taglia sui confini strutturali (hard).** Prima si divide il documento nelle
   sue sezioni naturali, nell'ordine di priorità:
   `intestazione` → `corpo` → `coda multilingua`; e dentro il corpo, sulle
   **sezioni numerate** se ci sono, altrimenti sui **separatori** `* * *`,
   altrimenti sui **paragrafi**.
2. **Impacchetta fino alla taglia target (~200–400 parole).** Si uniscono unità
   piccole consecutive fino alla dimensione voluta, **senza mai attraversare un
   confine duro** (una sezione numerata non si fonde con un'altra; il corpo non
   si fonde con la coda).
3. **Overlap solo dentro la stessa sezione.** Quando una sezione è più lunga
   della taglia target e va spezzata, si aggiunge una piccola sovrapposizione
   (1–2 frasi) tra i pezzi *interni* a quella sezione. **Niente overlap che
   scavalca un confine duro.**

Così l'overlap preserva il contesto dove serve (dentro un ragionamento lungo) ma
non mescola cose che non vanno mescolate (la riflessione dell'Angelus con i
saluti finali, o due catechesi numerate diverse).

### Ricetta per tipologia

| Tipologia | Confini primari | Note |
|---|---|---|
| audiences | sezioni numerate → paragrafi; **stacca la coda multilingua** | la più strutturata; coda quasi sempre presente |
| angelus | **corpo vs "Dopo l'Angelus"**; poi paragrafi | due parti nette nel ~metà dei casi |
| homilies | sezioni numerate (JPII) → paragrafi | corpo unico, liturgico |
| speeches | sezioni numerate (JPII) → paragrafi | volume enorme: il caso "base" |
| messages | sezioni numerate → paragrafi; stacca le note `[n]` | spesso formali/numerati (JPII) |
| letters | paragrafi | corte, narrative |
| cotidie | paragrafi (un solo blocco) | sintesi giornalistiche, mai numerate |
| travels | paragrafi; **filtra gli stub** (<~80 parole) | molti non sono discorsi veri |

### Metadati per chunk

Ogni chunk porta con sé i campi del documento più l'etichetta strutturale, utile
per filtrare in ricerca e per i confronti:

```
papa, tipologia, data, titolo, url        (dal documento)
sezione   = corpo | saluti | numerata     (tipo di sezione)
n_sezione = 1, 2, …                        (numero, se sezione numerata)
i_chunk   = posizione del chunk nella sezione
```

## Anomalie / cose da decidere

- **`travels` corti** (mediana 126 parole): probabilmente stub/indici. Da
  filtrare sotto una soglia o trattare separatamente.
- **Coda multilingua**: default escluderla dalla ricerca tematica italiana.
- **Numerazione assente nei Papi recenti**: il chunking deve degradare con
  grazia ai paragrafi quando non trova numeri.

## Da qui

Questa mappa alimenta lo **Stadio 1** del piano (la primitiva `pezzi`): invece di
un chunking unico, `pezzi(documento)` riconosce le sezioni e applica la ricetta
della sua tipologia. Resta KISS: poche regole, una funzione, nessuna dipendenza
nuova — solo qualche regex sui marcatori qui sopra.

> **Stato (misura interim).** `vdb.py` non fa ancora il chunking strutturale:
> spezza a finestra fissa. Il rumore multilingue è però già attenuato per altra
> via — ogni chunk ha la **lingua** e i saluti tradotti (in lingua diversa dal
> documento) sono **esclusi di default** dalla ricerca. Restano da togliere, col
> chunking strutturale, l'**intestazione** in testa e i saluti **italiani**
> ("Dopo l'Angelus") che la lingua da sola non distingue dal corpo.
