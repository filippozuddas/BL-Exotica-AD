# Piano di allineamento al paper UDMA — benchmark, memoria, teacher distillato, soglia

**Data**: 2026-07-14, aggiornato 2026-07-15 · **Stato**: approvato (decisioni D1–D11 sessione
di grilling 14/07; D12–D14 ricalibrazione post-Fase-0 15/07, vedi §1.1)
**Prerequisito di lettura**: `docs/design/udma-spec.md` (spec UDMA v1, Q1–Q10) e il paper
Qi et al. 2024, IEEE IoT Journal 11(24):39361 (`~/internship_BL/papers/Unsupervised_Spectrum_Anomaly_Detection_With_Distillation_and_Memory_Enhanced_Autoencoders.pdf`).

---

## 1. Contesto: esito del check implementazione-vs-paper (2026-07-14)

Confronto sistematico eq. 2–27 / Tabelle I-II-IV-VIII del paper contro `src/models/udma.py`,
`src/models/memory.py`, `configs/model/udma.yaml`.

**Fedele**: schema S-T (1 teacher + 2 studenti AE/MemAE, training solo su dati normali),
normalizzazione del target per-canale a livello dataset (eq. 3–5 ≡ `fit_udma_teacher_norm.py`),
struttura della loss (eq. 6–9, SS inclusa nel training), mappe e fusione (eq. 23–26,
`(1/d)·diff²`, pesi 0.5/0.5/0.5 esatti), hard shrinkage + rinorm L1 (eq. 13), AdamW.

**Tre deviazioni strutturali — tutte nella direzione che produce il collasso del disagreement**
(meno limitazione di capacità degli studenti, teacher più domain-matched):

- **(A) Teacher**. Paper: CNN a 3 layer (RF 7×7) *distillata da una rete generica pre-addestrata
  congelata P* (eq. 2, à la Bergmann "Uninformed Students") — lo spazio di feature del teacher è
  ancorato **fuori dal dominio per costruzione**; i dati di spettro entrano solo come input della
  distillazione, mai come target di apprendimento. Noi: ViT-MAE addestrato self-supervised sul
  dominio; il teacher nuovo (`4b5660c` ep094) è addestrato sulla *stessa cache* degli studenti —
  l'opposto esatto del paper. Il finding empirico "old teacher (SRT) ~2× meglio a ogni SNR"
  (vedi `docs/decisions/teacher-localization.md`) è il paper ri-derivato empiricamente: il
  vecchio teacher era *più fedele* al design originale.
- **(B) Memoria**. Paper: variante Park 2020 — cosine similarity, update mechanism (eq. 14–17),
  compactness+separateness (eq. 19–20, λc=λs=0.1), concat(z, z̃) verso il decoder, **M=10–50
  item con 1 query per campione** (best: 30 sintetico, 10 dati reali). Noi: variante Gong 2019 —
  dot-product, entropia, sostituzione (no concat), **N=500 slot con 384 query per-posizione**.
  Il paper è esplicito (§VI): la memoria esiste per *"limit the learning capacity of the student
  network, thereby increasing the difference between the outputs of the student network and the
  teacher network on anomalous samples"*. La nostra MemAE-studente è ordini di grandezza più
  espressiva → può tracciare il teacher anche sulle anomalie → `ss` collassa.
- **(C) Studenti**. Paper: encoder-decoder con bottleneck stretto (1×8×32), limitazione di
  capacità dichiarata. Noi: trunk+head senza bottleneck-e-riespansione (deviazione deliberata e
  validata — elimina la diluizione pixel-space — ma è il terzo limitatore di capacità rimosso).

Deviazioni minori giustificate: aggregazione topk (il footprint delle nostre anomalie è 2–3/384
patch vs ≥1 colonna intera della loro mappa full-res → la media semplice di eq. 27 da noi
diluisce); soglia (il paper calibra a **1% FAR su campioni anomaly-free** — avalla il fix
OFF-noise-core della Fase 3, contro l'attuale median+3·MAD su pool RFI-inclusive).

## 1.1 Ricalibrazione post-Fase-0 (2026-07-15) — il gap oldT/newT si è ridimensionato

L'esecuzione della Fase 0 (§3) ha rivelato che i numeri decisivi del 14/07 citati sopra
("old teacher ~2× meglio a ogni SNR", 48.9/68.0/79.3 vs 22.4/35.8/62.2) **non sono
riproducibili** — vedi memoria `benchmark_reproducibility_gap` per l'analisi completa
(checkpoint/config/codice verificati identici; unica variabile non verificabile è l'identità
esatta delle 15 cadenze del run originale, il cui log è andato perso).

Il control re-run sul Tier A congelato (`data/raw/benchmark_tierA_cadences.txt`, 2026-07-15)
dà, per **topk det@3σ_cad**:

| SNR | oldT (`133823`) | newT-bs256 (`083205`) | gap |
|---|---|---|---|
| 15 | 80.9% | 75.1% | 5.8 pt |
| 20 | 94.7% | 90.7% | 4.0 pt |
| 30 | 99.8% | 98.4% | 1.4 pt |

Il gap reale (4–6 punti) è un ordine di grandezza più piccolo di quello storico (~20–30
punti). Questo **indebolisce la narrativa causale della sezione 1** ("teacher domain-matched
collassa il disagreement") — su un confronto controllato, old-T e new-T sono quasi appaiati,
non 2× distanti. Conseguenza diretta: i bar D5/D9 tarati sul gap storico non hanno più senso
alla scala reale (uno slack assoluto di 5 punti, ad es., coprirebbe l'intero gap). Bar
corretti: D12 (memoria) e D13 (teacher 1c) sostituiscono D5/D9; vedi §2. Nuovo step di
verifica del rumore: D14/§3bis.

Questo non invalida la direzione delle Fasi 1–2 (la memoria e il teacher distillato restano
ipotesi valide, coerenti col paper), ma ne ridimensiona l'entità dell'effetto atteso — un
"PASS" ora rappresenta un guadagno reale ma piccolo, non il recupero di un gap enorme.

## 2. Log delle decisioni (grilling 2026-07-14)

| # | Decisione |
|---|---|
| D1 | Sequenza: Fase 0 (hardening benchmark) → Fase 1 (memoria) → Fase 2 (teacher 1c, parte dopo i risultati di Fase 1) → Fase 3 (soglia, codice in parallelo, validazione sul vincitore) |
| D2 | Benchmark a due tier: A = screening (15 cadenze fisse, SNR 15/20/30, n=30, seed fisso); B = conferma (50 cadenze, SNR 10–50), solo per l'arm vincente |
| D3 | Esperimento memoria sul teacher **nuovo** `4b5660c` ep094, batch 256 / lr 1e-3 (= run di controllo `20260713_083205`) |
| D4 | Un solo arm: `mem_slots=30`, `shrink_threshold=1/30` |
| D5 | ~~Bar memoria: PASS = ≥metà del gap verso oldT a SNR 15 e 20 (≥35% e ≥52% det@3σ_cad topk) senza regressione a SNR30; KILL = entro ~5 punti dalla baseline newT ovunque~~ **SUPERSEDUTO 2026-07-15, vedi D12** |
| D6 | P = **ResNet-18 ImageNet, layer3** (256 ch, stride /16 → griglia (6,64) nativa su (96,1024)) |
| D7 | Doppio gate: `teacher_sensitivity_test.py` prima su P (kill economico), poi distilla T, poi gate su T |
| D8 | T = riuso `build_encoder` (stesso trunk degli studenti) + head 1×1 → **128 canali** su (6,64); D = proiezione 1×1 128→256 usata solo in distillazione, poi scartata |
| D9 | ~~Bar teacher 1c: PASS = entro −5 punti da oldT (48.9/68.0/79.3 @ SNR 15/20/30) a ogni SNR → teacher di produzione; WIN = ≥ oldT; KILL = sotto il miglior newT (incl. eventuale fix memoria)~~ **SUPERSEDUTO 2026-07-15, vedi D13** |
| D10 | Fase 3 nel piano: ceiling OFF-noise-core per-cadenza + operating point FP-fisso/top-N; definition-of-done end-to-end |
| D11 | L'esperimento "teacher GBT degradato" (next step registrato in memoria il 14/07) NON si esegue: **la Fase 2 È quel test** — se P (mai addestrata sul dominio) passa il gate e il teacher distillato detecta bene, la domanda domain-mismatch-vs-learnability è risposta per costruzione |
| D12 | **(2026-07-15)** Bar memoria corretto — vedi §1.1 per la ricalibrazione. PASS = l'arm mem30 raggiunge/supera **oldT intero** (≥80.9% SNR15 E ≥94.7% SNR20 det@3σ_cad topk) senza regressione a SNR30 (≥98.4%); KILL = miglioramento <2 punti su newT-bs256 sia a SNR15 sia a SNR20; zona grigia → `val_ss` + mappe prima di decidere. Il vecchio bar (D5, "metà gap") è stato tarato su un gap storico (~20-30 punti) rivelatosi non riproducibile (vedi memoria `benchmark_reproducibility_gap`); il gap reale (4-6 punti) rende "metà gap" ≈2-3 punti, indistinguibile dal rumore run-to-run |
| D13 | **(2026-07-15)** Bar teacher 1c corretto — WIN = ≥ oldT (80.9/94.7/99.8) a ogni SNR (invariato); PASS = entro **−2 punti** da oldT a ogni SNR (stretto da −5 a −2: con lo slack vecchio, newT-bs256 stesso — 75.1/90.7/98.4 — passerebbe quasi automaticamente a SNR20/30, rendendo il bar un timbro senza significato); KILL = ≤ newT-bs256 (o il vincitore di Fase 1 se PASS) a qualunque SNR |
| D14 | **(2026-07-15)** Prima di avviare la Fase 1: un **run di replica** (stesso checkpoint, seed di iniezione diverso) su Tier A per stimare il rumore run-to-run di det@3σ_cad — finché non è misurato, D12/D13 sono tarati su n=1 campione per checkpoint e i margini (2 punti) potrebbero essere sotto il rumore reale. Vedi §3bis |

Assunzioni registrate: arm memoria single-variable (solo `mem_slots`/`shrink_threshold` cambiano);
distillazione = cache train completa, 1–2 epoche, AdamW, L2 su griglia, input a P = snippet
standardizzati replicati ×3 senza rinormalizzazione ImageNet (valida il gate); studenti di Fase 2
ereditano il `mem_slots` vincente di Fase 1; upgrade Park-v2 (concat/comp-sep/update/cosine) fuori
scope, follow-up solo se Fase 1 passa; produzione nel frattempo = ricetta old-teacher
(`20260713_133823_7779a9c`); tutte le run sul server; pesi torchvision da procurare sul server.

## 3. Fase 0 — Hardening del benchmark (bloccante, solo lavoro locale + sync)

Motivazione: ≥6 incidenti registrati con la stessa forma ("numeri decisivi da eval non
riproducibili"); i det@3σ_cad decisivi del 14/07 esistevano solo su stdout e sono andati persi.

- **0.1** `scripts/inject_recover.py`: la tabella per-cadenza (righe ~383–406: `det@3s_pool`,
  `det@3s_cad`, `det@5s_cad`, `eta2`) va **salvata su CSV** accanto alle righe pooled già scritte
  in `inject_recovery_results.csv` (riga ~425) — stesso file, colonne nuove, o file gemello
  `inject_recovery_per_cadence.csv`.
- **0.2** `scripts/inject_recover.py`: **auto-salvataggio dello stdout** in
  `<out_dir>/run.log` (tee interno via `logging`/duplicazione di stream) — niente più dipendenza
  dal `| tee` manuale.
- **0.3** Diagnosi e fix del blocco **"RFI-INCLUSIVE BACKGROUND" stale** (riga ~286): nel
  confronto del 14/07 era byte-identico tra due run con background diversi (topk 0.2340/0.6556/517
  in entrambe vs 0.3206/2609 nella sezione per-run). Trovare la causa (probabile stampa da stato
  cache/variabile riusata) e correggerla.
- **0.4** **Congelare le liste cadenze**: oggi Tier A = "prime 15 righe di
  `heldout_cadences.txt`" (implicito via `--max_cadences 15`, righe 172–173). Scrivere le liste
  esplicite `data/raw/benchmark_tierA_cadences.txt` (le stesse 15 della run decisiva del 14/07) e
  `benchmark_tierB_cadences.txt` (le 50 della run n=50), tracciate in git, e usarle via
  `--cadence_list`. Seed di iniezione fisso e documentato nel file stesso (header commento).
- **0.5** **Commit** dei config finora untracked: `configs/model/udma_old_teacher.yaml`,
  `configs/training/udma_gbt_fine_control_bs256.yaml`,
  `configs/training/udma_gbt_fine_old_teacher_fixed_preproc.yaml`, più questo documento.
- **0.6** ✅ **FATTO 2026-07-15.** Sync al server e re-run di controllo del Tier A sulla ricetta
  oldT (`133823`): i numeri **non hanno riprodotto** 48.9/68.0/79.3 (ottenuto 80.9/94.7/99.8 —
  vedi §1.1 e memoria `benchmark_reproducibility_gap`). Anche newT-bs256 (`083205`) ri-eseguito
  sullo stesso Tier A: 75.1/90.7/98.4 (non 22.4/35.8/62.2). Entrambi i numeri corretti sono ora
  la baseline canonica; i vecchi sono ritirati come non riproducibili. Il valore di questo step
  è esattamente quello previsto: ha scoperto che checkpoint/config/codice identici non bastano
  senza lista cadenze congelata + log salvato.

**Comando benchmark Tier A (canonico, da qui in poi)**:
```
python scripts/inject_recover.py \
  --checkpoint <ckpt> --model_config <model.yaml> \
  --cadence_list data/raw/benchmark_tierA_cadences.txt \
  --snr_list 15 20 30 --n_injections 30 --methods recon topk
```
Metrica decisiva: **topk det@3σ_cad** (+ rank% e η² come lenti secondarie).

## 3bis. Run di replica per stima del rumore (D14) — ✅ FATTO 2026-07-15

Motivazione: D12/D13 erano tarati su **n=1 misurazione per checkpoint** (un solo run oldT, un
solo run newT-bs256). Prima di gatekeeping su margini di 2 punti, serviva sapere se 2 punti
fosse dentro o fuori il rumore run-to-run dello stesso identico checkpoint.

- **3bis.1** ✅ Ri-eseguito il comando Tier A canonico su **newT-bs256** (`083205`) con
  `--seed 43` (vs il canonico `--seed 42`) — stessa lista cadenze, stesso checkpoint, stesso
  config, cambia solo il seed di iniezione.
- **3bis.2** ✅ **Risultato**: topk det@3σ_cad seed43 = 75.1/90.4/98.0% vs seed42 =
  75.1/90.7/98.4% — spread di **0.0/0.3/0.4 punti**. Rumore run-to-run ben sotto 1 punto →
  i margini di 2 punti in D12/D13 sono ampiamente al di sopra del rumore, **nessun
  allargamento necessario**. Fase 1 può procedere su D12 con fiducia che il bar non sia
  accidentalmente dentro il rumore di misura.
- **3bis.3** ✅ Memoria aggiornata (`benchmark_reproducibility_gap`).

## 4. Fase 1 — Esperimento memoria (`mem_slots` 500→30)

Domanda: *la restrizione di capacità dello studente MemAE riapre il disagreement collassato dal
teacher domain-matched?* Meccanismo del paper (§VI) applicato al nostro caso.

- **1.1** Nuovo config `configs/model/udma_mem30.yaml`: copia di `udma.yaml` (teacher nuovo
  ep094 + norm_stats ep094 invariati) con `student.mem_slots: 30` e
  `student.shrink_threshold: 0.0333` (=1/30, default paper/Gong; l'attuale 0.002 era 1/500).
  Nient'altro cambia (λ=1/1/1, entropy_weight 2e-4, head, filters).
- **1.2** Nuovo config `configs/training/udma_gbt_fine_mem30.yaml`: copia di
  `udma_gbt_fine_control_bs256.yaml` (batch 256, lr 1e-3, AdamW, 60 epoche, monitor `val_st_sum`)
  puntato al model config 1.1.
- **1.3** ✅ Training sul server (2×4090), `outputs/20260715_102544_36e4358`. `val_ss` è
  rimasto stabilmente 2.2-2.8× più alto della run newT-bs256 (mem_slots=500) per tutte le 41
  epoche — segnale meccanicistico confermato e robusto, il meccanismo scatta come da attese.
- **1.4** ✅ Eval: benchmark Tier A provvisorio su due checkpoint indipendenti (epoch 22 ed
  epoch 31, 9 epoche di distanza) invece che aspettare `last.ckpt`/early-stop naturale — il
  training era su un plateau di loss senza early-stop imminente (patience counter mai oltre
  1-2/8). Baseline di confronto corrette (vedi memoria `benchmark_reproducibility_gap`,
  supersedono i vecchi 22.4/35.8/62.2 e 48.9/68.0/79.3): newT-bs256 = 75.1/90.7/98.4,
  oldT = 80.9/94.7/99.8. Risultato mem30: epoch22 = 78.7/89.6/97.6, epoch31 = 77.3/89.6/97.3
  — **stesso quadro piatto/negativo a 9 epoche di distanza**, nessun guadagno reale nonostante
  `val_ss` costantemente elevato.
- **1.5** ✅ Verdetto secondo D12: **KILL**. A SNR20 regressione netta (-1.1pt su newT-bs256);
  a SNR15 il nominale +2.2pt è dentro il rumore di misura quantificato in
  `benchmark_reproducibility_gap` (spread 0.0-0.4pt tra run identiche a seed diverso) — non
  distinguibile dal rumore a quella scala. PASS (raggiungere oldT) mai vicino su nessuno SNR.
  Training fermato dall'assistente a epoch 41 (autorizzazione preventiva dell'utente), GPU
  liberate. Arm chiuso, tutto il peso passa a Fase 2 — `mem_slots` resta **500** per gli
  studenti di Fase 2 (5.3), non 30.
- **1.6** ✅ Memoria aggiornata: `udma_mem30_fase1_result`.

Costo: 1 retrain studenti + 2 run Tier A (provvisorie, su checkpoint non a convergenza) +
1 run di controllo per il rumore. Nessuna modifica a `src/`. Sweep {10, 50, 100} e Park-v2
(menzionati come follow-up se PASS) non eseguiti — arm KILLED, non pursued.

## 5. Fase 2 — Teacher distillato da rete generica (opzione 1c, paper-faithful)

Domanda pratica: *un teacher ancorato a feature out-of-domain per costruzione raggiunge la ricetta
old-teacher, rendendo la proprietà-che-fa-funzionare un invariante di progetto?*
Domanda scientifica (D11): risponde anche a domain-mismatch-vs-learnability senza run dedicate.

**Decisione 2026-07-15 (post-KILL Fase 1)**: lo sweep di follow-up `{10, 50, 100}` menzionato
in 1.5 per il caso PASS è **deprioritizzato, non eseguito**. Motivazione: il disaccordo
prodotto da N=30 non era anomaly-specific (rumore diffuso anche su `val_st1`/dati normali,
non concentrato sulle iniezioni) — una restrizione più blanda (50/100) è una versione più
debole della stessa leva già rivelatasi inefficace, non ci si aspetta un esito diverso. Il
gate su P (5.1, PASS con margini larghi) punta più probabilmente alla vera causa (teacher
domain-matched troppo prevedibile, non capacità degli studenti) — tutto il peso va lì. Se
anche la Fase 2 non chiude il gap, tornare sulla memoria richiederà un design diverso (Park
completo: concat/compactness-separateness/update, non solo N più piccolo), non un semplice
sweep numerico.

### 5.1 Gate su P (kill-check economico, prima di scrivere la distillazione) — ✅ FATTO 2026-07-15

- Wrapper minimale `ResNetTeacher` (`scripts/debug/resnet_teacher.py`): torchvision
  `resnet18(weights=IMAGENET1K_V1)` congelata, forward = replica 1→3 canali, estrazione feature a
  fine `layer3` → (B, 256, 6, 64). Nessuna rinormalizzazione ImageNet (input = snippet già
  median/MAD-standardizzati; il gate ha validato la scelta).
- Primo run di `scripts/debug/teacher_sensitivity_test.py --architecture resnet18` (branch
  additivo, path ViT invariato): tutti e 4 i gate PASS con margini ampi — ma usava l'iniettore
  custom (gaussiano, non setigen) allora presente in `injection_vs_rfi_test.py`. Standardizzato
  tutto il codebase su setigen nella stessa sessione (`injection_vs_rfi_test.py`,
  `scripts/slides/recon_grid.py`), poi ri-eseguito il gate con l'iniettore corretto.
- **Diagnosi del mismatch** (`scripts/debug/snr_convention_check.py`, CPU-only): l'iniettore
  custom iniettava **16.34× più energia fisica** del setigen a parità di SNR nominale (fattore
  costante su tutto lo sweep, non rumore) — "SNR=20" del vecchio iniettore corrispondeva a
  ~SNR=80 reale. Il risultato setigen è quindi quello corretto da fidarsi (è la stessa
  convenzione usata in tutta la pipeline, training incluso).
- **Sweep SNR allargato** (`--snr_list 10 20 30 40 60 80 100 150`) per il metodo di
  energy-matching di G3b (che richiede sovrapposizione tra energia iniettata ed energia RFI
  reale — a SNR realistici l'iniezione è tipicamente meno energetica dell'RFI reale, coerente
  con `docs/decisions/scoring-history.md` §1): **tutti e 4 i gate PASS** — G1=0.640, G2=0.985,
  G3a=0.811 (soglia 0.70, riferimento fisso SNR=20, invariato dall'allargamento), G3b=0.949
  (soglia 0.60). Nessun fallback WRN-50-2/opzione 1b necessario.
- Nota infra: pesi torchvision scaricati automaticamente al primo uso (il server ha accesso a
  internet; il test iniziale con `urllib` nudo aveva dato un falso 403 per assenza di header).
- Dettagli completi: `docs/decisions/teacher-localization.md`.

### 5.2 Distillazione P→T — ✅ FATTO 2026-07-15

- **Modello**: `TeacherCNN` in `src/models/udma.py`: trunk = `build_encoder` con la stessa
  parametrizzazione degli studenti (filters [32,64,128,256], convs_per_block 2, latent_dim
  128). Espone la stessa interfaccia di `TeacherViT` (`forward` normato, `grid_size`,
  `channels`, buffer `mu`/`sigma`, `fit_normalization`); `build_udma` ha un nuovo ramo
  `teacher.type: cnn_distilled` (default `vit_mae` invariato, `_load_teacher_cnn`).
- **Script** `scripts/distill_teacher.py`: P congelata, T + testa di allineamento D (conv 1×1
  128→256, scartata a fine training) addestrate con `L = ‖D(T(x)) − P(x)‖²`; cache train
  completa (994k snippet), 2 epoche, AdamW lr 1e-3. Loss 0.247→~0.005, convergenza pulita.
  Checkpoint: `outputs/udma_teacher_distill/cnn_distilled_resnet18.pt`.
- **Gate su T**: ✅ ri-run di `teacher_sensitivity_test.py --architecture cnn_distilled`
  (nuovo branch, `_TeacherCNNGateAdapter`) — **tutti e 4 i gate PASS**: G1=1.367 (più forte di
  P), G2=0.975 (quasi invariato da P's 0.985), G3a=0.748 (soglia 0.70, un po' di sensibilità
  persa vs P's 0.811 ma solido), G3b=0.966 (più forte di P). T mantiene la proprietà chiave di
  P (ancorato out-of-domain, sensibile alle iniezioni) dopo la distillazione. Dettagli:
  memoria `udma_resnet_teacher_gate`.
- **Norm stats**: da fare — `fit_udma_teacher_norm.py` su T → `outputs/udma_teacher_norm/cnn_distilled_*.pt`.

### 5.3 Studenti + eval

- Config `configs/model/udma_cnn_teacher.yaml`: teacher = T distillato; `mem_slots` = **500**
  (Fase 1 KILLED 2026-07-15, vedi memoria `udma_mem30_fase1_result` — non 30); resto identico.
  Training config gemello a batch 256 / lr 1e-3.
- Retrain studenti → benchmark Tier A → verdetto secondo D13 (PASS/WIN/KILL vs oldT
  80.9/94.7/99.8, non i vecchi 48.9/68.0/79.3 — vedi `benchmark_reproducibility_gap`). Se
  PASS/WIN → Tier B di conferma → T diventa il teacher di produzione; registrare in memoria
  anche la risposta alla domanda scientifica D11.
  **VERDETTO (2026-07-16): WIN netto.** Checkpoint `outputs/20260715_204845_6c46c35/checkpoints/
  epoch=057-val_loss=0.0438.ckpt` (epoch 57 = val_st_sum-best), Tier A completa (15 cadenze,
  n_injections=30). topk det@3σ_cad = **93.33/98.44/100.0%** @ SNR 15/20/30, sopra oldT
  (80.9/94.7/99.8) a ogni SNR, margine più ampio a SNR15 (+12.4 punti). D11 risposta: il teacher
  distillato da un backbone generico (paper-fedele) NON solo pareggia ma supera oldT sulla
  detection — ma vedi `udma_teacher_rf_leakage_refuted`/`udma_voyager_shortlist_off_leak_concern`
  per l'asse ortogonale (localizzazione ON/OFF): T non localizza (contrast 2.26 vs oldT 32-35),
  quindi NON idoneo come teacher di produzione per la pipeline di ricerca nonostante il WIN in
  detection — i due assi richiedono teacher diversi, vedi memoria per la narrativa completa.

## 6. Fase 3 — Soglia e criterio end-to-end

Codice model-independent: si può scrivere durante le run GPU di Fase 1/2. Validazione: solo
sulla ricetta vincente.

- **3.1** `src/search/candidates.py`: nuova funzione di ceiling OFF-noise-core per-cadenza —
  pool dei valori di cella delle righe OFF (1,3,5) attraverso gli snippet della cadenza, strip
  della coda RFI (break robusto), quantile alto (~0.999) del noise-core residuo. (Progettazione
  già validata a mano su cad02: ceiling ≈0.9 contro thresh_3=0.16 sepolto nel rumore; il
  quantile grezzo senza strip è WRONG — prende l'RFI a ~31 e uccide tutto.)
- **3.2** `scripts/inference.py`: il row-hit test di `full_row_hits`/`on_off_contrast`
  (righe ~438–447) usa il ceiling 3.1 al posto di `thresh_3`; la selezione candidati passa a un
  operating point dichiarato (FP-fisso stile 1%-FAR del paper, o top-N per cadenza) invece del
  3σ gaussiano su 131k snippet (che produce centinaia di crossing su rumore puro per costruzione
  — meccanismo HIP114176 già confermato).
- **3.3** **Definition-of-done end-to-end** (chiude il piano): su una cadenza held-out con
  iniezione a SNR noto (es. 20), il segnale iniettato sopravvive alla short-list di
  `inference.py`; su una cadenza pure-noise (es. cad02/MESSIER42) la short-list scende a ~0
  candidati con decadimento senza plateau. Nessuna ricetta ha ancora passato questo test.

## 7. Rischi e fallback

| Rischio | Mitigazione/fallback |
|---|---|
| Feature ImageNet insensibili alla struttura radio fine | Gate su P prima di ogni investimento (5.1); fallback WRN-50-2, poi 1b |
| La distillazione perde la sensibilità di P | Doppio gate (P e T) separa i due effetti; leva = epoche/head di D |
| `mem_slots=30` sotto-capacità anche per i dati normali (st2 non converge) | Sintomo visibile in `val_st2` durante il training; leva = 50/100 nello sweep follow-up |
| Benchmark Tier A troppo rumoroso per la bar ±5 punti | 0.6 (riproduzione oldT) quantifica il rumore del metro prima di usarlo per decidere |
| Fase 1 e 2 entrambe KILL | Produzione resta oldT `133823` (validata); si riapre il "teacher degradato" come diagnostico (D11 decade) |

**Ordine di esecuzione riassunto**: 0.1–0.6 → 1.1–1.6 → (5.1 può partire appena il server è
libero, è solo un gate) → 5.2–5.3 → 3.3. Fase 3 codice (3.1–3.2) in qualunque momento.
