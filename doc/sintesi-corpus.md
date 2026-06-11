# Sintesi del corpus

Questo documento descrive **che cosa contiene** il corpus dei documenti dei
Papi su cui lavora il vector database, in forma puramente **quantitativa e
strutturale**.

> **Nota sul copyright.** Qui sotto ci sono solo *metadati e conteggi* (numero
> di documenti, totali di parole, periodi, dimensioni): cifre ricavate dai file,
> **non i testi**. I testi sono di © Libreria Editrice Vaticana / Dicastero per
> la Comunicazione e **non vengono riprodotti né in questo documento né nel
> repository**. Il corpus si rigenera in locale dalla fonte ufficiale con
> [`ingestion-documenti-papi`](https://github.com/martinocrippa/ingestion-documenti-papi);
> l'`url` di ogni documento originale resta nel frontmatter dei file. Le cifre
> qui riportate sono una fotografia al **2026-06-11** e variano leggermente a
> ogni nuova esecuzione dell'ingestion.

## In una riga

**~25.000 documenti**, **~30 milioni di parole**, **quattro pontefici**, dal
**1978 a oggi**. È materiale più che sufficiente per la ricerca semantica e i
confronti tra pontificati che il progetto si propone.

## Documenti e parole per Papa

| Papa | Documenti | Parole | Media parole/doc | Periodo |
|---|--:|--:|--:|---|
| Giovanni Paolo II | 15.368 | 18.641.168 | 1.212 | 1978–2005 |
| Francesco | 6.057 | 6.977.390 | 1.151 | 2013–2025 |
| Benedetto XVI | 3.001 | 3.890.739 | 1.296 | 2005–2013 |
| Leone XIV | 639 | 658.989 | 1.031 | 2025–2026 |
| **Totale** | **25.066** | **30.168.286** | **1.204** | **1978–2026** |

Il peso di Giovanni Paolo II (oltre il 60% dei documenti) riflette un
pontificato lungo quasi 27 anni; Leone XIV è agli inizi. Questo squilibrio è un
fatto da tenere presente nelle analisi: i confronti tra Papi vanno **normalizzati**
(per documento o per periodo), non fatti sui totali grezzi.

## Documenti per tipologia

| Tipologia | Giovanni Paolo II | Francesco | Benedetto XVI | Leone XIV |
|---|--:|--:|--:|--:|
| speeches (discorsi) | 8.543 | 2.394 | 1.418 | 322 |
| homilies (omelie) | 2.293 | 580 | 352 | 86 |
| angelus | 1.541 | 689 | 459 | 61 |
| audiences (udienze) | 1.177 | 526 | 347 | 55 |
| messages (messaggi) | 671 | 802 | 183 | 91 |
| letters (lettere) | 706 | 194 | 140 | 16 |
| travels (viaggi) | 437 | 83 | 102 | 8 |
| cotidie (omelie S. Marta) | — | 789 | — | — |

Le tipologie ricalcano le sezioni del sito vaticano. `cotidie` (le omelie
mattutine a Santa Marta) esiste solo per Francesco. I discorsi (`speeches`) sono
ovunque la categoria più numerosa.

## Formato di ogni documento

Un file markdown per documento, con frontmatter YAML (i campi su cui si appoggia
l'indicizzazione) e il corpo del testo:

```
data/<papa>/<tipologia>/<nomefile>.md
```

Campi del frontmatter: `papa`, `tipologia`, `data`, `titolo`, `url`, `parole`.
La struttura esatta è mostrata in [`../esempio/`](../esempio/) con un documento
finto (nessun testo reale).

## Che cosa significa per l'indicizzazione

Alcune conseguenze pratiche di queste cifre, utili per il disegno
(vedi [`architettura.md`](architettura.md)):

- **Volume gestibile in locale.** 30M di parole, spezzate in pezzi da qualche
  centinaio di parole, danno nell'ordine delle **centinaia di migliaia di
  chunk**. A 384 dimensioni in `float32` è una matrice di pochi GB: sta in RAM,
  la ricerca brute-force con NumPy è sufficiente. Non serve un database
  vettoriale esterno per partire.
- **Testi quasi tutti in italiano** → serve un modello di embedding
  multilingue/italiano.
- **Lunghezze omogenee** (media ~1.200 parole) → una strategia di chunking
  semplice e uniforme funziona bene.
- **Metadati ricchi e affidabili** (`papa`, `tipologia`, `data`) → i confronti
  tra Papi e nel tempo si fanno **filtrando per metadato**, senza arricchimenti
  complessi nella prima versione.
