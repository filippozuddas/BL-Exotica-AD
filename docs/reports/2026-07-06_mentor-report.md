# BL-Exotica-AD — Report per i mentor
**Data:** 2026-07-06
**A cura di:** Filippo Zuddas
**Ambito:** scelte architetturali, implementazione e risultati dell'ultimo periodo (fine giugno – inizio luglio 2026), focalizzato su `0000.fil` (narrowband drifting).

---

## 1. Sintesi per chi ha poco tempo

Dopo aver verificato che **cinque famiglie di scorer non supervisionati** basate su ricostruzione o densità nello spazio latente (AE, MemAE, ViT-MAE/MAE, Mahalanobis su embedding, GMM) **falliscono tutte** nel separare segnali iniettati da RFI reale a energia comparabile, abbiamo costruito e validato un sesto approccio — **UDMA** (Unsupervised Distillation and Memory-enhanced Autoencoder, da Qi et al. 2024) — che **supera tutte le bar di accettazione pre-registrate** ed è ora lo **scorer di produzione** della pipeline.

UDMA sostituisce lo scoring a pixel (dove il segnale si annega nella media su ~98k pixel) con uno scoring nello **spazio delle feature di un encoder ViT-MAE congelato**, su una griglia di sole 384 posizioni: due studenti (un autoencoder e un autoencoder con memoria) imparano a predire le feature del teacher solo sul rumore/RFI normale; sull'anomalia falliscono in modo diverso l'uno dall'altro, e questo disaccordo è il segnale di anomalia.

**Risultato chiave:** AUC a energia abbinata 0.88–0.90 (contro 0.77–0.79 del probe precedente), con TPR@10%FP del 100% a SNR≥12 e, soprattutto, **rilevazione non nulla anche a SNR 5–7** — il punto operativo a basso SNR che tutti i tentativi precedenti non riuscivano a raggiungere.

---

## 2. Contesto: perché non i metodi "standard"

La specifica di progetto chiede di **estendere** l'approccio di Ma et al. (ContrastiveVAE + Random Forest, supervisionato su etichette ON/OFF) verso una ricerca **non supervisionata a più ampio spettro morfologico**. Questo vincolo — niente classificatore, niente etichette, niente obiettivo ON/OFF nel training — è stato mantenuto per tutta l'esplorazione ed è il motivo per cui alcune strade "facili" (es. un classificatore supervisionato sulle iniezioni, o uno stadio di vetting non-ML) sono state scartate anche quando avrebbero funzionato empiricamente.

### 2.1 Cosa abbiamo provato e perché ha fallito

| Approccio | Meccanismo | Esito | Diagnosi |
|---|---|---|---|
| **AE standard** (ricostruzione MSE) | encoder-decoder, errore di ricostruzione come score | R² AUC 0.42–0.50 (sotto il caso, sotto il livello triviale) | L'AE **copia** anche la riga iniettata: non c'è residuo da misurare. Più dati (200k→1M) peggiora il risultato, perché il modello impara a ridisegnare meglio anche le righe RFI reali. |
| **MemAE** (memoria di prototipi normali) | decoder vincolato a "prototipi" appresi solo sul normale | R² AUC 0.648 | Meglio dell'AE (il gap è **architetturale**, verificato), ma non supera in modo netto le statistiche banali (energia/picco) — vince ma di poco. |
| **ViT-MAE / MAE** (masking 75%) | ricostruzione da patch mascherate | R² AUC 0.61, collassa a "detector di energia" | Il masking rende il problema "predici rumore da rumore", non risolvibile: il modello converge alla media. |
| **Mahalanobis / GMM su embedding** | distanza statistica nello spazio latente | AUC ~0.5 (caso) su segnale vs RFI reale | L'embedding **contiene** morfologia (AUC supervisionato 0.746–0.845), ma il segnale iniettato è troppo vicino, in densità, al rumore normale: un one-class fit-on-quiet non riesce a isolarlo senza etichette. |
| **Occupancy scorer** (basato su pattern ON/OFF di cadenza) | vetting sull'occupazione spettrale della cadenza | Bar passate, ma il ramo vincente non era encoder-based | **Ritirato per principio** (vincolo architetturale: la pipeline deve restare interamente basata su encoder), non per prestazioni. |

**Conclusione condivisa dopo questi cinque fallimenti:** il problema non è la rappresentazione (la morfologia esiste nell'encoder, dimostrato più volte con AUC supervisionato 0.75–0.85), ma il **meccanismo di scoring**: ricostruzione pixel-space misura *prevedibilità*, non *anomalia* — un segnale liscio e prevedibile (una riga che deriva in frequenza) può addirittura **abbassare** l'errore medio di ricostruzione sotto il livello del solo rumore.

### 2.2 Il probe che ha aperto la strada: disaccordo AE↔MemAE

Prima di costruire l'intera architettura UDMA, abbiamo testato un'idea più semplice e a **costo zero di training**: usare la differenza pixel-per-pixel tra le ricostruzioni di un AE e di un MemAE già allenati, `‖AE(x) − MemAE(x)‖²`, come score.

- **Risultato (validato 2026-07-05):** AUC a energia abbinata 0.770–0.788 su tre seed, sia su train che su validation (mai vista) — **il primo scorer non supervisionato a battere le statistiche banali**.
- **Limite:** al di sotto di SNR 10, il TPR@10%FP scende a 0% — la coda dei punteggi RFI più difficili schiaccia la soglia operativa.
- Questo risultato ha validato il **meccanismo** (disaccordo studente-studente rileva ciò che la densità non vede) e ha motivato la costruzione dell'architettura UDMA completa, dove il disaccordo si misura non sui pixel ma sulle **feature del teacher**.

---

## 3. Metodologia di valutazione: i test che usiamo per ogni modello

Ogni modello (AE, MemAE, ViT-MAE, il probe di disaccordo, UDMA) è stato passato attraverso **la stessa batteria di test**, cosa che rende i numeri della tabella in §2.1 e delle bar in §5 direttamente confrontabili tra loro. La batteria vive principalmente in due script (`scripts/debug/encode_separation_test.py`, `scripts/debug/eti_vs_rfi_separation_test.py`) e in uno script di iniezione su dati reali (`scripts/inject_recover.py`). Di seguito il "perché" di ciascun test, non solo il "cosa": la maggior parte è nata per chiudere una falla trovata nel test precedente.

### 3.1 Il problema di fondo che tutti i test devono aggirare: il confondimento con l'energia

Un segnale iniettato più forte (SNR alto) ha quasi sempre **più energia totale** di un frammento di rumore puro. Qualunque scorer — anche uno stupido come "somma dei quadrati dei pixel" — separa perfettamente segnale da rumore semplicemente misurando l'energia, **senza aver imparato nulla sulla forma del segnale**. All'inizio del progetto un AUC alto veniva letto come "il modello riconosce la morfologia"; si è poi scoperto (giugno 2026) che in diversi casi era solo questo effetto banale. Da qui in poi ogni test riporta **sempre insieme** tre numeri, non uno solo:

1. **AUC del modello** — quanto separa bene segnale iniettato da RFI reale.
2. **AUC di un controllo a sola energia** (`energy_only`, es. somma dei quadrati del frame) — quanto separerebbe un detector che guarda solo quanta potenza c'è, senza capire la forma.
3. **AUC di statistiche banali di forma** (`trivial-stats`: picco massimo, curtosi, frazione di canali "caldi") — quanto separerebbe un detector semplice che guarda la forma grezza ma non ha un encoder.

Un modello "vince" solo se batte **entrambi** i controlli, non solo il caso (AUC 0.5).

### 3.2 Test operativo non abbinato (R1)

**Cosa fa:** inietta un segnale sintetico a SNR crescente (3, 5, 7, 10, 12, 15, 20, 30, 50...) in rumore reale, poi misura l'AUC di separazione (iniettato vs RFI reale) **ad ogni livello di SNR separatamente**, affiancando la colonna `AUC(energy)` per lo stesso confronto.

**Perché serve:** è il test più diretto e intuitivo ("a che SNR il modello inizia a funzionare?"), ma da solo è ingannevole: se `AUC(modello) ≈ AUC(energy)` a ogni SNR (Δ→0), il modello si sta comportando come un detector di energia puro e mascherato — è il caso diagnosticato per ViT-MAE (Δ≈0 ad alto SNR: "detector di energia", non di morfologia).

**Esempio di lettura:** AE standard risultava **sotto** il controllo di energia ad ogni SNR (Δ da −0.086 a −0.011) — peggio che guardare solo la potenza, segno che il modello introduce rumore proprio, non morfologia.

### 3.3 Test a energia abbinata — "matched-energy caliper" (R2, il test decisivo)

**Cosa fa:** per ogni frame RFI reale nel campione, cerca il frame "segnale iniettato" con **energia totale più vicina possibile** (entro una tolleranza, il "caliper", tipicamente 0.02–0.1) e forma una coppia. Scarta i frame che non trovano un partner abbastanza vicino. Sulle coppie rimaste (energia forzatamente equivalente tra le due classi) misura l'AUC di separazione.

**Perché serve:** è il modo per **eliminare per costruzione** il confondimento di energia descritto in §3.1, invece di limitarsi a misurarlo come in R1. Se un modello separa ancora bene le due classi *quando l'energia è identica*, la differenza residua **deve** venire dalla forma (morfologia), non dalla potenza. Questo è il test che ha bocciato Mahalanobis/GMM (AUC ~0.5 anche qui) e promosso UDMA (AUC 0.88–0.90).

**Nota metodologica importante (lezione imparata a nostre spese):** la dimensione del campione conta. Un risultato di AUC=0.927 ottenuto a fine giugno con ~75 coppie per classe **non si è riprodotto** con lo stesso seed e proveniva da una partizione casuale fortunata. Da allora ogni R2 riportato usa **almeno ~450–500 coppie per classe** (regola empirica adottata dopo l'incidente) e viene ripetuto su **più seed indipendenti** prima di essere considerato affidabile.

### 3.4 TPR @ soglia fissa di falsi positivi (test operativo di soglia)

**Cosa fa:** fissa una soglia di falsi positivi accettabile (tipicamente 10% — cioè: al massimo 1 RFI reale su 10 sopra soglia viene scambiato per candidato), calibra la soglia sul punteggio del modello sul mix di RFI reale, poi misura quanti segnali iniettati a un dato SNR superano quella soglia (**TPR**, true positive rate).

**Perché serve:** l'AUC è una media su tutte le possibili soglie e può nascondere un fallimento nel punto operativo realistico che useremmo davvero in una ricerca (bassa % di falsi positivi tollerata). È il test che ha rivelato il "buco a basso SNR": il probe di disaccordo pixel-space aveva AUC dignitoso (0.77) ma **TPR@10%FP = 0% a SNR ≤7** — la coda dei punteggi più alti di RFI difficile monopolizzava la soglia. Questo singolo numero (non l'AUC) è stato il motivo per costruire UDMA.

### 3.5 Test di induttività (train vs validation)

**Cosa fa:** ripete esattamente il test R2 su un secondo insieme di cadenze **mai viste durante il training** (validation split, diviso per cadenza intera per evitare fughe di informazione) e confronta l'AUC con quella ottenuta su train.

**Perché serve:** un modello potrebbe "memorizzare" i pattern di RFI specifici delle osservazioni di training invece di imparare una regola generale. Se l'AUC su validation crolla rispetto a train, il risultato non generalizza. Per UDMA lo scarto è Δ = 0.000–0.024 (soglia accettata: ±0.03) — il segnale è genuinamente appreso, non memorizzato.

### 3.6 Test di idoneità del teacher (gate pre-training, solo per architetture teacher-student come UDMA)

Prima di allenare un'intera architettura nuova, tre controlli via via più severi verificano che il teacher scelto sia **adatto al meccanismo**, senza bisogno di training (minuti su GPU):

- **G1 — non collasso:** le feature del teacher non devono essere degeneri/costanti su tutti gli input (altrimenti qualunque studente le predice banalmente, disaccordo nullo per costruzione).
- **G2 — responsività:** iniettando una riga sintetica, i token del teacher **devono muoversi** in modo misurabile nella posizione attraversata dalla riga rispetto a dove non passa (stesso sfondo, confronto appaiato). Se il teacher non "vede" l'anomalia, nessuno studente potrà mai discordare su di essa.
- **G3 — anteprima del meccanismo:** uno studente lineare "usa e getta" (ridge regression, nessun training vero) prova a predire le feature del teacher; se anche questo predittore banale ci riesce troppo bene sull'anomalia, il gap studente-teacher non sarà informativo per gli studenti veri (più potenti).

**Perché serve:** costruire l'intera architettura UDMA (giorni di lavoro + training) su un teacher inadatto sarebbe stato scoperto solo a valle, a costo pieno. Questi gate lo escludono in anticipo. Risultati applicati al nostro teacher (ViT-MAE, layer 3) in §4.2.

### 3.7 Test di occupanza ON/OFF (caratterizzazione, non una bar pass/fail)

**Cosa fa:** costruisce due classi con la **stessa identica morfologia e sfondo**, differendo solo nel pattern di cadenza — un segnale presente in tutte le osservazioni ("persistente") contro uno presente solo nei blocchi ON (pattern tipico di un segnale extraterrestre reale in una cadenza ON-OFF-ON-OFF-ON-OFF) — e misura se il modello li distingue.

**Perché serve:** verifica *cosa* uno scorer sta effettivamente misurando. Un AUC vicino a 0.5 qui non è un fallimento — anzi, è atteso e voluto per UDMA: significa che lo scorer giudica la morfologia del singolo frame indipendentemente dal pattern di cadenza, e che la discriminazione ON/OFF (basata su etichette implicite di cadenza) resta correttamente delegata a uno stadio successivo dedicato, non mescolata nello scorer principale (vincolo di progetto: nessun obiettivo ON/OFF nel training).

### 3.8 Mappe di anomalia qualitative (ispezione visiva)

**Cosa fa:** per singoli esempi rappresentativi (rumore puro, RFI reale, segnale iniettato) genera la mappa spaziale del punteggio di anomalia e la confronta visivamente con la posizione vera del segnale/RFI nel frame.

**Perché serve:** l'AUC dice "separa" ma non "dove" e "perché". Le mappe hanno permesso di verificare a occhio due cose che i numeri da soli non potevano garantire: (a) che il punteggio più alto cada davvero sulla cella spazio-temporale dove si trova il segnale (localizzazione, non un artefatto globale); (b) che il rischio pre-registrato di "diffusione" dell'attenzione del transformer (che avrebbe reso le mappe inutili anche con AUC alto) non si sia materializzato.

### 3.9 Test di iniezione-recupero su cadenze reali (Fase 2 della pipeline)

**Cosa fa:** a differenza di tutti i test precedenti (che lavorano su un campione cache in memoria), questo gira lo scorer finale sulle vere cadenze `.h5` su disco, iniettando segnali sintetici a SNR crescente e misurando il tasso di rilevazione con una soglia statistica (mediana + kσ del rumore locale), non a percentuale di falsi positivi fissa.

**Perché serve:** è il test più vicino alle condizioni operative reali della ricerca finale (Fase 3) — stesso formato dati, stessa pipeline di preprocessing end-to-end — e funge da riscontro indipendente rispetto alla cache di valutazione: se i due protocolli (cache vs disco) raccontano storie diverse, è un segnale di allarme. Per UDMA raccontano la stessa storia (curva liscia e monotona in SNR), che rafforza la fiducia nel risultato.

### 3.10 Riepilogo dei test

| Test | Cosa misura | Perché è stato introdotto |
|---|---|---|
| R1 (operativo non abbinato) | AUC per livello di SNR, con controllo di energia affiancato | Primo test intuitivo; rivela quando un modello è "solo" un detector di energia |
| R2 (matched-energy caliper) | AUC a energia forzatamente identica tra le classi | Elimina il confondimento di energia alla radice — il test decisivo per ogni bar |
| TPR@X%FP | Tasso di rilevazione a una soglia operativa realistica | L'AUC media nasconde i fallimenti nel punto operativo che useremmo davvero |
| Induttività (train/val) | Tenuta del risultato su cadenze mai viste | Esclude memorizzazione invece di apprendimento generale |
| Gate teacher (G1–G3) | Idoneità di un encoder-teacher prima di costruirci sopra un'architettura | Evita di scoprire "il teacher non va bene" solo dopo giorni di training |
| Occupanza ON/OFF | Se lo score dipende dal pattern di cadenza oltre che dalla morfologia | Verifica che lo scorer non mescoli morfologia e occupazione, per costruzione |
| Mappe qualitative | Localizzazione spaziale del punteggio | L'AUC non dice "dove"; verifica il rischio di diffusione/smearing |
| Iniezione-recupero su disco | Rilevazione end-to-end su dati reali, non su cache | Riscontro indipendente nelle condizioni operative reali |

---

## 4. UDMA — architettura e scelte progettuali

Riferimento: Qi et al. 2024, *"Unsupervised Spectrum Anomaly Detection With Distillation and Memory Enhanced Autoencoders"* (IEEE IoT Journal). Spec completa pre-registrata: `docs/design/udma-spec.md`.

### 4.1 Schema

```
Input (96, 1024) — cadenza narrowband, canale singolo
        │
        ▼
Teacher: encoder ViT-MAE CONGELATO (self-supervised, nostro)
  → token del blocco 3 del transformer → griglia (128, 6, 64)
        │
        ├──► Studente A (AE):    encoder conv + head di proiezione → (128, 6, 64)
        └──► Studente B (MemAE): encoder conv + memoria di prototipi + head → (128, 6, 64)

Score = disaccordo teacher↔studente-A + teacher↔studente-B + studente-A↔studente-B
        aggregato su griglia 6×64 (mean / top-k / max)
```

Decisioni principali e razionale:

- **Teacher = ViT-MAE congelato, nessun training aggiuntivo.** È lo stesso encoder self-supervised già validato (checkpoint `20260624_084754_057f87c`), quindi zero costo e piena coerenza con la storia unsupervised del progetto. Il layer scelto (blocco 3, non l'ultimo) è risultato il migliore su tutti i gate di idoneità testati (vedi §3.2).
- **Niente decoder pixel.** Gli studenti riproducono le feature del teacher su una griglia di **384 posizioni** (6×64), non i 98.304 pixel dell'input. Questo elimina strutturalmente il problema di diluizione che affliggeva ogni scorer a pixel: un'anomalia locale non si perde più in una media su decine di migliaia di pixel di rumore.
- **Due studenti, non uno.** Un autoencoder semplice e uno con memoria di prototipi (già validata nel progetto MemAE) vengono allenati insieme a predire le feature del teacher **solo sul normale**. Sull'anomalia, il vincolo strutturale della memoria fa sì che i due studenti sbaglino in modo diverso — quel disaccordo (non presente nel training) è il segnale.
- **Input invariato:** stessa cadenza (96,1024) usata in tutti i test precedenti — un solo cambiamento alla volta rispetto alla baseline.
- **Bar di accettazione fissate PRIMA di allenare** (disciplina adottata dopo un episodio di risultato non riproducibile in giugno, un AUC di 0.927 rivelatosi un artefatto di partizione casuale): 5 bar (B1–B5) su AUC, TPR a soglie di SNR fisse, induttività train/val, e sanità del test.

### 4.2 Verifica di idoneità del teacher (gate pre-training)

Prima di scrivere una riga di codice per l'architettura, abbiamo verificato che il teacher scelto fosse effettivamente adatto al meccanismo (test a costo quasi nullo, senza training):

- **Nessun collasso**: le feature del teacher non sono degeneri.
- **Responsività**: il teacher "vede" una riga iniettata attraverso i token attraversati (AUC di spostamento ≥0.99 al layer scelto).
- **Anteprima del meccanismo** con uno studente lineare "usa e getta": conferma che un predittore semplice non riesce a prevedere banalmente la risposta del teacher sull'anomalia (precondizione per cui il gap studente-teacher sia informativo).

Tutti i controlli sono passati nettamente; il layer 3 del transformer è risultato il migliore su ogni metrica, incluso il rapporto tra residuo su RFI reale e residuo su rumore puro (2.73× contro 3.86× dell'ultimo layer — un punto di partenza migliore per contenere i falsi positivi su RFI nota).

---

## 5. Risultati di validazione

Allenamento completo (30 epoche, dataset da ~560k cadenze reali, split 56/9/28), poi valutazione su 3 seed indipendenti, sia su train che su validation mai vista.

| Bar | Criterio pre-registrato | Risultato | Esito |
|---|---|---|---|
| **B1** — separazione a energia abbinata | AUC ≥ 0.80 su tutti i seed **e** superiore al probe precedente (0.770–0.788) | **0.884 / 0.887 / 0.901** | ✅ superato di +0.10–0.12 |
| **B2** — rilevazione a SNR medio-alto | TPR@10%FP ≥ 70% a SNR 12, ≥ 40% a SNR 10 | **100% / 100%** (contro 35–60% e 4–15% del probe) | ✅ |
| **B3** — rilevazione a basso SNR | TPR@10%FP > 0% a SNR 5 e 7 (il probe era fermo a 0%) | **68–92% a SNR 5, 97–100% a SNR 7** | ✅ — è esattamente il punto operativo che UDMA doveva sbloccare |
| **B4** — generalizzazione (non memorizzazione) | AUC su validation entro ±0.03 da train | Δ = 0.000–0.024 | ✅ |
| **B5** — sanità del test | il solo controllo di energia non deve già "vincere" da solo | margine ≥ +0.30 sopra il livello banale | ✅ |

**Tutte e cinque le bar sono state superate, su tutti e tre i seed, su entrambi gli split.** UDMA sostituisce quindi sia i vecchi scorer a ricostruzione (AE/MemAE/ViT-MAE) sia il probe di disaccordo a pixel come scorer di produzione.

### 5.1 Caratterizzazione qualitativa

- **Mappe di anomalia:** per esempi di rumore puro, RFI reale e segnale iniettato, le mappe (a griglia 6×64) localizzano nettamente la posizione vera dell'anomalia — la riga RFI sul suo intervallo temporale reale, il segnale iniettato sulla cella di frequenza/tempo corretta — senza la diffusione ("smearing") che era stata messa in conto come rischio accettato dell'attenzione globale del transformer. Su una RFI a banda larga che deriva in modo curvo nel tempo, il picco della mappa **segue la curva**, a conferma che il meccanismo cattura morfologia non banale, non solo un punto caldo statico.
- **Comportamento su RFI nota persistente:** una riga narrowband presente in modo continuo nella cadenza viene quasi completamente soppressa dagli studenti (che l'hanno vista spesso in training come "normale"), mentre resta un residuo solo dove la riga si discosta dal pattern appreso — esattamente il comportamento "sopprimi il noto, segnala l'inatteso" che era l'obiettivo del disegno.
- **Cecità all'occupazione ON/OFF:** UDMA, come atteso e per costruzione (nessun obiettivo di cadenza nel training), non distingue segnali ON-only da segnali persistenti (AUC ~0.49, livello del caso). Questo è corretto: la discriminazione ON/OFF resta compito di uno stadio successivo dedicato, non dello scorer di morfologia per singolo frame.

### 5.2 Test di iniezione-recupero (Fase 2, dati reali)

Su 10 cadenze reali (1440 iniezioni), lo score top-k mostra una curva di rilevazione liscia e monotona in funzione dell'SNR: dal 2.8% a SNR 3 fino al 100% a SNR 50, passando per 64% a SNR 15 e 89% a SNR 20. Lo scoring "a media" (senza top-k) resta strutturalmente più debole a ogni SNR — confermando che la diluizione su griglia (anche piccola) è un effetto reale, non un artefatto di campione.

---

## 6. Stato attuale e prossimi passi

- **UDMA è ora lo scorer di produzione validato** per `0000.fil`. La checklist di implementazione (gate → costruzione → smoke test → training → valutazione) è completa.
- **Aperto, non bloccante:** decidere come UDMA si inserisce operativamente nella pipeline di ricerca (`src/search/scorer.py`) — se sostituisce integralmente i vecchi scorer o se questi restano disponibili come confronto.
- **Fuori scope in questa versione (v2, esplicitamente rimandato):** un teacher a "campo recettivo piccolo" (CNN distillata dal ViT) nel caso in cui in futuro emerga smearing su dati più complessi; un input a 6 canali che sfrutti l'allineamento naturale tra righe della griglia del teacher e osservazioni della cadenza, per portare la sensibilità ON/OFF dentro lo scorer stesso.
- **`0001.fil` (transienti a banda larga):** non ancora affrontato — resta la prossima estensione naturale della "ricerca a spettro più ampio" richiesta dalla specifica, una volta consolidato l'uso di UDMA su `0000.fil`.

---

## 7. Riferimenti

- Specifica completa e pre-registrazione delle bar: `docs/design/udma-spec.md`
- Implementazione: `src/models/udma.py`, `configs/model/udma.yaml`, `configs/training/udma_gbt_fine.yaml`
- Script di valutazione: `scripts/debug/encode_separation_test.py`, `scripts/debug/eti_vs_rfi_separation_test.py`, `scripts/debug/udma_anomaly_maps.py`
- Paper di riferimento: Qi et al. 2024, IEEE IoT Journal 11(24):39361; Gong et al. 2019 (ICCV, MemAE)
