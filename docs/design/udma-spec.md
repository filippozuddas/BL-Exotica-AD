# UDMA-GBT — Design Spec (Q1–Q10)

**Data:** 2026-07-05
**Riferimento:** Qi et al. 2024, *"Unsupervised Spectrum Anomaly Detection With Distillation and Memory Enhanced Autoencoders"*, IEEE IoT Journal 11(24):39361 (`~/internship_BL/papers/Unsupervised_Spectrum_Anomaly_Detection_...pdf`)
**Stato:** spec pre-implementazione, bar di accettazione pre-registrate (§Q9) PRIMA di qualunque run.

---

## Contesto e motivazione

Il probe λ₃ (disaccordo pixel-space `‖AE(x)−MemAE(x)‖²`, zero training, `DisagreementPair` in
`scripts/debug/encode_separation_test.py`) è stato validato il 2026-07-05:

| Test | Risultato |
|---|---|
| Matched-energy AUC vs RFI reale (topk, n≈475/classe) | **0.770 / 0.776 / 0.788** (seed 42/7/123, train) |
| Matched-energy AUC su **val** (osservazioni mai viste) | **0.781** — induttivo, non memorizzazione |
| TPR@10%FP (mix RFI di train) | ~100% @ SNR≥20, 88–97% @ 15, 35–60% @ 12, **0% @ ≤7** |
| Occupanza (ON-only vs persistente, stessa morfologia) | 0.423 (mean) / 0.451 (topk) — **cieco all'ON/OFF**, traccia l'estensione della riga |

È il **primo scorer non supervisionato che batte le statistiche triviali a energia matched**
(cinque famiglie precedenti a chance: recon-MSE/topk, latent-density, GMM, dist384, mlp-probe).
Interpretazione: `‖AE−MemAE‖²` è un **novelty detector morfologico** — sopprime la RFI in-distribution
(gli studenti concordano) e segnala morfologia mai vista (l'AE la copia, il MemAE la ridisegna dai
prototipi normali).

**Perché costruire l'UDMA completo:** il buco residuo è il punto operativo a basso SNR
(TPR@10%FP = 0% sotto SNR 10 sul mix RFI difficile: la coda alta dei punteggi topk della RFI schiaccia
la soglia, il vecchio failure mode "max-error RFI" sopravvive nel punto operativo anche se non più nel
ranking). Nel paper, λ₃ (`map_ss`) è il termine **minore**; l'effetto principale sono i due termini
teacher-student (`map_st1`, `map_st2`): studenti che regrediscono le **feature del teacher**, non i
pixel. Spostare lo score dallo spazio pixel (dove vive la coda RFI) allo spazio feature è
l'attacco diretto al meccanismo diagnosticato ("recon misura la predicibilità, non l'anomalia").

Vincoli rispettati: pipeline interamente encoder-based (nessuno stadio non-ML), nessuna etichetta,
nessun classificatore, nessun obiettivo ON/OFF nel training (v1).

---

## Q1 — Sorgente del teacher

**Decisione: teacher = encoder ViT-MAE congelato (nostro, self-supervised su GBT), feature a livello
di token. NESSUNO stadio di distillazione P→T in v1.**

- `ViTMAE.encode_tokens(x)` esiste già: `(B, 384, 128)` → reshape/transpose a **(B, 128, 6, 64)**
  (griglia patch 16×16 su input (96,1024): 6 righe temporali × 64 colonne di frequenza).
  Post-gate (v. Esito sotto): il teacher legge i token dal **blocco 3** del transformer
  (`teacher_layer: 3` in config), non dal layer finale.
- Checkpoint pinnato: **`outputs/training/20260624_084754_057f87c/checkpoints/epoch=006-val_loss=2.1715.ckpt`**
  (canonico, `loss_mode: generative`, matcha `configs/model/vit_mae.yaml`; NON la variante denoising
  `20260626_122208_2f09a7c`). Stesso checkpoint di B3/diagnostiche → confrontabilità diretta.

**Razionale.** (a) Domain-matched e label-free: preserva la storia unsupervised/encoder-based. La
distillazione P→T del paper serve a comprimere una rete generica, allineare le dimensioni e imporre un
receptive field piccolo — i primi due non si applicano (feature già compatte e di dominio); il terzo
serve alla *localizzazione*, che non è il nostro readout primario (score per frame). (b) B3 = 0.845:
la morfologia vive dimostrabilmente in questo encoder. (c) Costo zero: nessun training del teacher.

**Alternative respinte.**
- *RST AST (progetto rst)*: supervisionato su etichette di iniezione ETI → lega la ricerca alle classi
  note, contro l'obiettivo "broader range" e la storia unsupervised. Escluso per principio, non per
  performance.
- *CNN ImageNet-pretrained (approccio del paper)*: domain-foreign su spettrogrammi standardizzati non
  limitati; dipendenza esterna. Riserva di v2 se il teacher ViT delude.

**Obiezione pre-empita:** "dist384/Mahalanobis su questi stessi token ha già fallito." Vero, ma quello
era scoring di *densità* (questo token è lontano dalla nuvola normale?); il meccanismo S-T è
*prediction-gap* (uno studente addestrato solo sul normale sa predire la risposta del teacher qui?).
Il probe λ₃ ha già dimostrato che i gap di predizione/accordo funzionano dove la densità no.

**Rischio accettato (v1):** l'attenzione globale del ViT può "spalmare" la risposta all'anomalia su
più token. Accettabile per score frame-level; se le mappe risultano diffuse → v2: teacher CNN a
receptive field piccolo distillato dal ViT (lo stadio P→T del paper, reintrodotto).

**Gate pre-flight (OBBLIGATORIO, prima di implementare):** `scripts/debug/teacher_sensitivity_test.py`
verifica direttamente l'idoneità del teacher, zero training, ~minuti su GPU:
- **T1/G1 — no collapse**: rel-std ≥ 0.05 e participation-ratio rank ≥ 16/128 sui token quiet.
- **T2/G2 — responsività** (appaiata, token-level): `‖T(x+riga) − T(x)‖` sui token attraversati
  dalla riga vs non attraversati, stesso background → **AUC ≥ 0.80 a SNR 20** (dove la recon già
  rileva, quindi i token DEVONO portare la riga). Se fallisce, gli studenti predicono il teacher
  banalmente anche sulle anomalie → UDMA morto con questo teacher.
- **T3/G3 — preview del meccanismo** (studente lineare ridge, closed-form): residuo di predizione
  patch-pixel→token fittato sul normale; **G3a** AUC token (riga vs normale held-out) ≥ 0.70 a
  SNR 20; **G3b** preview frame-level a energia matched vs RFI reale ≥ 0.60 (lower bound: gli
  studenti conv sono più forti di una ridge). Bonus readout: rapporto residuo RFI/quiet held-out =
  preview del rischio FP su RFI conosciuta.
- Rami di fallback stampati dallo script: G2 fail → `--layer 3/4` (feature intermedie) o altro ckpt,
  poi v2; G2 pass + G3 fail → teacher responsivo ma troppo predicibile → layer intermedio, poi v2.

### Esito del gate (2026-07-05) — TEACHER IDONEO, con un emendamento documentato

Run su ckpt canonico, layer final / block3 / block4 (`outputs/sweeps/teacher_sensitivity/`):

| Layer | G2 disp-AUC @20 (min su SNR 5–40) | G3a tokAUC @20 | G3b preview | residuo RFI/quiet |
|---|---|---|---|---|
| final | 0.977 (0.970) | 0.858 | 0.821 | 3.86× |
| **block3** | **0.999 (0.998)** | 0.854 | **0.840** | **2.73×** |
| block4 | 0.996 (0.996) | 0.852 | 0.832 | 2.84× |

- **G2/G3a/G3b: PASS su tutti i layer**, con margini larghi — top-8 hit 100% e Cohen's d 6–10 a
  *tutti* gli SNR inclusi 5–7: il teacher risponde alle righe, localizza, e lo studente lineare non
  le sa predire. Il meccanismo S-T è vivo.
- **EMENDAMENTO G1**: il sub-gate "PR-rank pooled ≥ 16" è risultato **mal specificato**, non
  indicativo: la covarianza dei token pooled su tutte le posizioni è dominata dalla struttura dei
  positional embedding, che è *costante per posizione* — uno studente la predice gratis, quindi non
  può contare contro il teacher. Inoltre il contenuto dei frame quiet (rumore) è genuinamente
  low-dim. G1 è ridefinita come **solo collapse-check** (rel-std ≥ 0.05, dead dims ≤ 10% — passata
  con ampio margine: rel-std 0.50–2.40, 0 dead dims); il rango per-position-centered è riportato
  come metrica informativa (warning solo se < 4). I due test *diretti* del meccanismo (G2, G3) che
  il rango avrebbe dovuto approssimare sono passati in modo netto — il proxy era sbagliato, non il
  teacher. Script aggiornato di conseguenza (stesso giorno, prima di qualunque training UDMA).
- **Scelta layer: `teacher_layer: 3` (config, default)** — domina o pareggia su ogni metrica:
  G2 0.999 vs 0.977 (final), G3b 0.840 vs 0.821, e soprattutto il **residuo RFI/quiet più basso
  (2.73× vs 3.86×)** = miglior punto di partenza per il pavimento di FP su RFI conosciuta.
- **Metrica monitorata per gli studenti**: il rapporto residuo-su-RFI / residuo-su-quiet (per lo
  studente lineare: 2.73× a block3). Gli studenti conv devono portarlo verso ~1×: è la condizione
  perché il punto operativo a basso SNR si apra (nel preview lineare, frameAUC < 0.5 sotto SNR 10
  proprio perché la RFI reale resta impredicibile per una ridge).
- Caveat onesto su G3b: lo studente lineare NON batte le trivial stats (0.840 vs 0.821, n=32/classe)
  — accettabile per un lower bound lineare; per l'UDMA vero resta vincolante la bar B5
  (margine ≥ +0.15 su trivial).
- Nota di condizionamento del target (lever per le 2 iterazioni di tuning, Q4): con contenuto
  low-rank, la Norm per-canale amplifica canali quasi-rumore → in caso di mappe ST rumorose,
  proiettare il target sui top-k componenti PCA (o pesare i canali per varianza) invece della
  standardizzazione per-canale pura.

**Conferma finale (rilancio con lo script corretto, stesso giorno):** `--layer 3` con la nuova
definizione di G1 dà verdetto ufficiale dello script stesso — `G1 collapse: PASS`,
`G2 responsiveness: PASS (0.999)`, `G3a: PASS (0.854)`, `G3b: PASS (0.840)` →
**"teacher FIT: proceed with the UDMA build (spec Q1 confirmed)."** Gate Q1 chiuso, nessun
emendamento manuale necessario oltre al fix dello script. `teacher_layer: 3` è la decisione finale.

## Q2 — Normalizzazione dell'output del teacher

Per-canale (128 dim) su tutto il training set: `Norm(T(x)) = (T(x) − μ) / (σ + 1e−6)` con μ, σ ∈ R¹²⁸
calcolati offline una volta (paper Eq. 3–5), salvati nel checkpoint UDMA (buffer del modulo).
Il target di regressione degli studenti è `Norm(T(x))`.

## Q3 — Studenti: architettura

**Decisione: due studenti con lo stesso trunk conv del ConvAE validato + projection head — NIENTE
decoder pixel.**

Coincidenza architetturale chiave: l'encoder CNN (4 blocchi, riduzione 16×) produce
`(B, 64, 6, 64)` — **stessa griglia spaziale (6,64) dei token del teacher**. Quindi:

- **Studente A (AE):** `Encoder` esistente (filters [32,64,128,256], latent_dim 64) → head di
  proiezione `Conv2d(64→128, 1×1)` preceduta da 1–2 conv 3×3 (64→128→128) per contesto locale.
  Output `(B, 128, 6, 64)`.
- **Studente B (MemAE):** identico + `MemoryUnit` (500 slot, shrink 0.002, addressing per posizione
  spaziale — il modulo già validato) tra encoder e head.

**Niente upsampling/transposed-conv:** il "decoder" è la head sulla stessa griglia. Più leggero, più
veloce, e la risoluzione della mappa (384 posizioni vs 98k pixel) **elimina strutturalmente il
problema di diluizione** che ha afflitto tutti gli scorer pixel-space.

**Init:** from scratch (default). Il task è regressione di feature, non ricostruzione pixel: il
transfer dai checkpoint AE/MemAE esistenti è di valore incerto e sporca l'interpretabilità. Flag di
config `init_from:` opzionale per l'ablation.

## Q4 — Loss di training

Training congiunto dei due studenti (teacher frozen, `torch.no_grad()` + `.eval()`):

```
L = λ1·‖Norm(T(x)) − S_AE(x)‖² + λ2·‖Norm(T(x)) − S_Mem(x)‖² + λ3·‖S_AE(x) − S_Mem(x)‖²
    + entropy_weight · H(addressing)          (MemoryUnit esistente, 2e-4)
```

- **λ3 in training MINIMIZZA il disaccordo sul normale** (paper Eq. 8–9): è ciò che affila il
  disaccordo sulle anomalie. Default λ1=λ2=λ3=1.0, esposti in config.
- Le loss compactness/separateness del paper (Park-style, Eq. 19–22): **DIFFERITE a v2** — la nostra
  MemoryUnit (hard shrinkage + entropia) è già validata; aggiungere solo se l'addressing degenera.
- `compute_loss` ritorna `(total, {"st1":…, "st2":…, "ss":…, "entropy":…})` — stesso protocollo di
  VAE/MemAE, il trainer Lightning li logga senza modifiche.

## Q5 — Scoring

Tre mappe sulla griglia (6,64), media sui 128 canali:

```
map_st1 = mean_c (Norm(T(x)) − S_AE(x))²
map_st2 = mean_c (Norm(T(x)) − S_Mem(x))²
map_ss  = mean_c (S_AE(x) − S_Mem(x))²
map_cob = w1·map_st1 + w2·map_st2 + w3·map_ss     (default 0.5/0.5/0.5, config)
```

Aggregazione frame-level: `anomaly_score(x, method='recon'|'topk'|'max')` = mean / top-k (default
0.02 → ~8 posizioni su 384) / max di `map_cob`. **Duck-type-compatibile con gli harness esistenti**
(`recon_score` chiama `anomaly_score(x, method=…)`) → `encode_separation_test.py --scoring recon`
funziona senza modifiche. Primaria attesa: topk; decisione empirica sui tre.

Il λ₃ pixel-space validato (vecchi checkpoint AE/MemAE) resta disponibile come quarto candidato di
fusione a valle — costo zero, valutato solo in fase di analisi, non parte del modello.

## Q6 — Geometria dell'input

**v1 = (96,1024) single-channel** (cadenza impilata sull'asse tempo) — identico a ogni baseline e
diagnostica esistente. *Un cambiamento alla volta:* v1 introduce solo il meccanismo feature-space S-T.

**v2 (esplicitamente fuori scope qui):** input a 6 canali (6,16,1024). Nota per il futuro: le righe
della griglia token (6,64) sono già allineate 1:1 con le osservazioni (patch alte 16 px = 1 oss.),
quindi un'estensione cadence-aware può operare sulla griglia del teacher senza cambiare il teacher.
In v1 nessuno scoring cadence-aware: la discriminazione ON/OFF resta allo stadio a valle.

## Q7 — Dati e training config

- Dataset: stesso mmap 1M+ (`cache_gbt_fine`), split 56/9/28 per cadenza, preprocessing online
  invariato.
- Optimizer: AdamW + cosine annealing (paper), lr 1e-3 (il loro 5e-3 è per 9k campioni; noi ~560k),
  batch 256 (target (128,6,64) ≈ 49k float — molto più leggero della recon pixel), bf16-mixed.
- Epoche: max 30, early stopping su `val_st1+val_st2` (il MemAE pixel convergeva a ep. ~20).
- Teacher in `eval()` congelato dentro lo step (nessun BN nel ViT, LayerNorm ok in eval).
- Costo stimato: dominato dal forward del teacher ≈ un'inferenza ViT-MAE per batch + due studenti
  leggeri → ore, non giorni, su 1–2× RTX 4090.
- Seed/reproducibilità: protocollo esistente (`seed_everything`, run_id timestamp+git-hash).

## Q8 — Layout del codice

- `src/models/udma.py`:
  - `TeacherViT` — wrapper frozen: carica il ckpt ViT-MAE, `encode_tokens` → (B,128,6,64), applica
    Norm (buffer μ/σ); metodo `fit_normalization(loader)` per il calcolo offline.
  - `FeatureStudent` — trunk `build_encoder(...)` + head; flag `memory: bool` che inserisce
    `MemoryUnit` (riuso di `src/models/memory.py`).
  - `UDMA(nn.Module)` — teacher + 2 studenti; `compute_loss` (Q4), `anomaly_map`, `anomaly_score`
    (Q5); attributo `learning_rate` come gli altri backbone.
- `configs/model/udma.yaml` — `architecture: udma`; sezione teacher (ckpt path + vit config path);
  sezione student (riusa lo schema encoder di convae.yaml); λ, pesi di scoring, topk_frac.
- `build_autoencoder()` in `src/models/autoencoder.py`: route `architecture: udma` → `build_udma()`.
  Il trainer Lightning esistente lo avvolge senza modifiche.
- Eval: nessun nuovo harness — `encode_separation_test.py --scoring recon --model_config
  configs/model/udma.yaml --checkpoint <udma.ckpt>` (R1+R2), `eti_vs_rfi` per la caratterizzazione
  di occupanza.

## Q9 — Bar di accettazione PRE-REGISTRATE

Tutte su `--n_samples 2000`, `--recon_method topk` salvo indicato; confronto contro il probe λ₃
sugli **stessi** seed/split. Disciplina post-ritrattazione-0.927: le bar si fissano ORA.

| # | Bar | Soglia |
|---|---|---|
| B1 | Matched-energy AUC (R2, train, seed 42/7/123) | **≥ 0.80 su tutti i seed** e **> λ₃ probe** (0.770/0.776/0.788) sugli stessi seed |
| B2 | TPR@10%FP, mix RFI train | **≥ 70% @ SNR 12** (λ₃: 35–60%) e **≥ 40% @ SNR 10** (λ₃: 4–15%) |
| B3 | Basso SNR | TPR@10%FP **> 0% a SNR 5 e 7** (λ₃: 0%) |
| B4 | Induttività | AUC matched-energy su val entro **±0.03** dal train |
| B5 | Sanità harness | residuo energy-only ≤ 0.58 sulle coppie matched; margine su trivial ≥ +0.15 |

- **B1 fallita ⇒ l'UDMA non paga la complessità: si tiene il probe λ₃ come scorer di produzione**,
  risultato negativo documentato. B2/B3 fallite con B1 passata ⇒ vittoria parziale: si adotta UDMA
  per il ranking ma il gap operativo a basso SNR resta aperto (documentare, poi v2).
- Caratterizzazione senza bar: test di occupanza (atteso ~cieco, come λ₃); mappe di anomalia
  qualitative su 3 esempi (noise/RFI/ETI) per il rischio smearing (Q1).
- Budget di tuning: **≤ 2 iterazioni** (λ di training, capacità studenti, topk_frac) prima del
  verdetto. Niente fishing.

## Q10 — Rischi e criteri di kill

| Rischio | Sintomo | Mitigazione |
|---|---|---|
| Feature del teacher troppo facili da predire (lisce/low-rank) → gap nullo ovunque | `val_st*` → ~0, mappe piatte anche su ETI iniettata | ↓ capacità studenti (filters/2); ↑ shrink memoria; v2: teacher CNN small-RF distillato |
| λ3 di training fa collassare gli studenti l'uno sull'altro (accordo anche sulle anomalie → `map_ss` morta) | AUC del solo `map_ss` ≪ probe pixel-λ₃ | λ3_train ∈ {1, 0.1, 0} (una delle 2 iterazioni di tuning); nel paper il vincolo strutturale del MemAE-studente basta a preservare il disaccordo |
| Smearing da attenzione globale (Q1) | mappe diffuse, localizzazione persa, topk≈mean | accettato per score frame-level; altrimenti v2 teacher small-RF |
| Scelta del teacher ckpt non ottimale | — | pinnato al canonico 057f87c per confrontabilità; ablation su ckpt più recenti solo dopo il verdetto v1 |

**Kill criterion:** B1 non soddisfatta dopo le 2 iterazioni di tuning ⇒ stop UDMA, il probe λ₃
(già validato) resta lo scorer; si documenta e si passa allo stadio di vetting di cadenza della
pipeline.

---

## Checklist di implementazione

0. **Gate pre-flight del teacher** (Q1): `teacher_sensitivity_test.py` sul ckpt canonico — G1–G3b
   devono passare PRIMA di scrivere `udma.py`. In caso di fail: `--layer 3/4`, altro ckpt, poi v2.
1. `src/models/udma.py` (TeacherViT, FeatureStudent, UDMA, build_udma) + route in
   `build_autoencoder`.
2. `configs/model/udma.yaml` + config di training (`configs/training/udma.yaml`, batch 256, lr 1e-3,
   max 30 epoche, early stopping su val_st).
3. Script/step per `fit_normalization` (μ/σ del teacher sul train set) — one-shot, salvato nel ckpt.
4. Smoke test CPU forward/loss su input random (dev machine, senza dati).
5. Training sul server (comando documentato nel config).
6. Eval: R1+R2 (3 seed, train + val) + occupanza + 3 mappe qualitative → verdetto contro Q9.
