# UDMA — architecture and design decisions

The production anomaly scorer. Adapts Qi et al. 2024 (*Unsupervised Spectrum
Anomaly Detection With Distillation and Memory-Enhanced Autoencoders*, IEEE IoT
Journal 11(24):39361) to GBT `0000.h5` cadences.

Read [`01_scoring-history.md`](01_scoring-history.md) first — it explains why a
feature-space disagreement scorer was necessary at all.

**Implementation:** `src/models/udma.py`, `configs/model/udma.yaml`
**Results against the bars defined here:** [`05_results.md`](05_results.md)

---

## 1. The idea in one paragraph

A frozen teacher network maps a spectrogram to a compact feature grid. Two
lightweight CNN **students** are trained — on normal noise/RFI only — to
regress that grid. On in-distribution data the students learn to reproduce the
teacher's response and to agree with each other; on morphology never seen in
training they fail, and *they fail differently*. That prediction gap is the
anomaly signal. There is **no pixel decoder anywhere**.

This is the feature-space generalization of the validated pixel-space probe
`‖AE(x) − MemAE(x)‖²`. Moving the score off the pixel manifold is the direct
attack on the diagnosed failure mechanism: reconstruction error measures
predictability, not anomalousness, and the RFI tail lives in pixel space.

Constraints honoured: fully encoder-based, no labels, no classifier, no ON/OFF
objective in training.

## 2. Teacher

**Decision: the teacher is our own frozen ViT-MAE encoder, self-supervised on
GBT data, read at transformer block 3. No P→T distillation stage.**

`ViTMAE.encode_tokens(x)` gives `(B, 384, 128)`, reshaped to `(B, 128, 6, 64)` —
a patch grid of 6 time rows × 64 frequency columns over a `(96, 1024)` input.
The checkpoint is pinned for comparability across every diagnostic.

**Rationale.** Domain-matched and label-free, preserving the unsupervised
history. The paper's P→T distillation exists to compress a generic network,
align dimensions, and impose a small receptive field; the first two do not apply
(our features are already compact and in-domain), and the third serves
*localization*, which is not the primary frame-level readout.

**Rejected alternatives.** A supervised AST teacher (ties the search to known
signal classes — excluded on principle, not performance) and an
ImageNet-pretrained CNN (domain-foreign, external dependency).

**Pre-empted objection.** "Mahalanobis on these same tokens already failed."
True, but that was *density* scoring (is this token far from the normal cloud?).
The student–teacher mechanism is a *prediction gap* (can a student trained only
on normal data predict the teacher's response here?). The pixel-space probe
already showed prediction gaps work where density does not.

### 2.1 Mandatory pre-flight gates

`scripts/debug/teacher_sensitivity_test.py` — zero training, minutes on GPU.
**Run this before adopting any new teacher checkpoint or read-layer.**

- **G1 — no collapse:** rel-std ≥ 0.05, dead dims ≤ 10%.
- **G2 — responsiveness:** paired token-level `‖T(x+line) − T(x)‖` on tokens the
  line crosses vs. tokens it does not, same background. AUC ≥ 0.80 at SNR 20.
  If this fails, students predict the teacher trivially even on anomalies and
  UDMA is dead with that teacher.
- **G3a/G3b — mechanism preview:** a closed-form ridge student fitted on normal
  data; token AUC ≥ 0.70 at SNR 20, and a frame-level matched-energy preview
  ≥ 0.60 against real RFI (a lower bound — conv students are stronger).

**Gate outcome (2026-07-05): teacher fit, block 3 selected.**

| Layer | G2 disp-AUC @20 | G3a token AUC | G3b preview | RFI/quiet residual |
|---|---|---|---|---|
| final | 0.977 | 0.858 | 0.821 | 3.86× |
| **block3** | **0.999** | 0.854 | **0.840** | **2.73×** |
| block4 | 0.996 | 0.852 | 0.832 | 2.84× |

Block 3 dominates or ties on every metric, and crucially has the lowest
RFI/quiet residual ratio — the best starting point for the false-positive floor
on known RFI. The ratio students must drive toward ~1× is the condition for the
low-SNR operating point to open up.

**One gate was amended, honestly.** The original G1 sub-gate "pooled
participation-ratio rank ≥ 16" was **mis-specified**: the covariance of tokens
pooled over all positions is dominated by positional-embedding structure, which
is constant per position — a student predicts it for free, so it cannot count
against the teacher. G1 was redefined as a collapse check only, and the rank is
reported as informative. The two *direct* tests the rank was meant to
approximate (G2, G3) passed decisively: the proxy was wrong, not the teacher.

## 3. Teacher output normalization

Per-channel over the whole training set,
`Norm(T(x)) = (T(x) − μ) / (σ + 1e−6)` with μ, σ ∈ ℝ¹²⁸ computed offline once
(paper eq. 3–5) and stored as module buffers. This is the students' regression
target. Computed by `scripts/fit_udma_teacher_norm.py`.

## 4. Students

**Two students sharing the validated ConvAE trunk, plus a projection head — no
pixel decoder.**

The key architectural coincidence: the CNN encoder (4 blocks, 16× reduction)
produces `(B, 64, 6, 64)` — **the same spatial grid as the teacher's tokens**,
because a `(16,16)` patch on a `(96,1024)` input equals the encoder's 16×
downsampling.

- **Student A (AE):** existing encoder → projection head `Conv2d(64→128, 1×1)`
  preceded by 1–2 3×3 convs for local context. Output `(B, 128, 6, 64)`.
- **Student B (MemAE):** identical, plus a `MemoryUnit` (500 slots, shrink
  0.002, per-spatial-position addressing) between trunk and head.

No upsampling or transposed convolution: the "decoder" is the head on the same
grid. The map resolution (384 positions vs. ~98k pixels) **structurally
eliminates the dilution** that defeated every pixel-space scorer.

Students are initialized from scratch — the task is feature regression, not
pixel reconstruction, so transfer from AE/MemAE checkpoints is of uncertain
value and would muddy interpretability.

## 5. Training loss

Joint training of both students, teacher frozen (`no_grad()` + `.eval()`):

```
L = λ1·‖Norm(T(x)) − S_AE(x)‖²
  + λ2·‖Norm(T(x)) − S_Mem(x)‖²
  + λ3·‖S_AE(x) − S_Mem(x)‖²
  + entropy_weight · H(addressing)
```

λ3 **minimizes disagreement on normal data** (paper eq. 8–9) — that is what
sharpens disagreement on anomalies. Defaults λ1=λ2=λ3=1.0, all exposed in
config. `compute_loss` returns `(total, {st1, st2, ss, entropy})`, so the
Lightning trainer logs them unmodified.

The paper's Park-style compactness/separateness losses are deferred: our
MemoryUnit (hard shrinkage + entropy) is already validated.

## 6. Scoring

Three maps on the `(6, 64)` grid, averaged over 128 channels:

```
map_st1 = mean_c (Norm(T(x)) − S_AE(x))²      teacher vs AE student
map_st2 = mean_c (Norm(T(x)) − S_Mem(x))²     teacher vs MemAE student
map_ss  = mean_c (S_AE(x)   − S_Mem(x))²      student-student
map_cob = w1·map_st1 + w2·map_st2 + w3·map_ss  (default 0.5/0.5/0.5)
```

Frame-level aggregation: `anomaly_score(x, method='recon'|'topk'|'max')`.
**Production uses `topk` with `topk_frac = 0.01`** (~4 of 384 positions), swept
empirically. Top-k beats mean at every SNR — grid dilution is a real effect even
at 384 positions.

Note that equal fixed weights dilute `map_ss`, which is the strongest single
term at SNR 10. See [`05_results.md`](05_results.md) §4.

## 7. Input geometry

`(96, 1024)` single-channel — the cadence stacked on the time axis, identical to
every existing baseline and diagnostic. One change at a time: the only new
mechanism is feature-space student–teacher scoring.

Because the token grid's 6 rows already align 1:1 with the cadence's 6
observations (patch height 16 px = one observation), a future cadence-aware
extension could operate on the teacher's grid without changing the teacher. In
the current version there is no cadence-aware scoring: ON/OFF discrimination
lives in the downstream stage
([`04_candidate-filtering.md`](04_candidate-filtering.md)).

## 8. Training configuration

- Same 1M+ memmap dataset, 56/9/28 per-cadence split, online preprocessing.
- AdamW + cosine annealing, lr 1e-3, batch 256, bf16-mixed.
- Early stopping on `val_st1 + val_st2`.
- Teacher frozen in `eval()` inside the step.
- Cost dominated by the teacher forward — hours, not days, on 1–2× RTX 4090.

## 9. Pre-registered acceptance bars

Fixed **before any run**, as standing discipline after the retracted 0.927
result (see [`01_scoring-history.md`](01_scoring-history.md) §2.1). Compared
against the pixel-space probe on the same seeds and splits.

| # | Bar | Threshold |
|---|---|---|
| B1 | Matched-energy AUC, 3 seeds | ≥ 0.80 on all seeds **and** > probe (0.770/0.776/0.788) |
| B2 | TPR@10%FP, train RFI mix | ≥ 70% @ SNR 12, ≥ 40% @ SNR 10 |
| B3 | Low SNR | TPR@10%FP > 0% at SNR 5 and 7 |
| B4 | Inductiveness | val AUC within ±0.03 of train |
| B5 | Harness sanity | energy-only residual ≤ 0.58; margin over trivial ≥ +0.15 |

**Kill criterion:** B1 unmet after at most 2 tuning iterations ⇒ stop UDMA and
keep the pixel-space probe as the scorer. Tuning budget was capped at 2
iterations deliberately — no fishing.

All five bars passed; see [`05_results.md`](05_results.md).

## 10. Risks accepted at design time

| Risk | Symptom | Mitigation |
|---|---|---|
| Teacher features too easy to predict → gap vanishes | `val_st*` → 0, flat maps even on injected signal | reduce student capacity; increase memory shrink; small-RF distilled teacher |
| Training λ3 collapses students onto each other → `map_ss` dead | `map_ss` AUC ≪ pixel probe | λ3 ∈ {1, 0.1, 0} |
| Global-attention smearing → maps diffuse, localization lost | topk ≈ mean | accepted for frame-level scoring |

The smearing risk did **not** materialize with this teacher — but it is exactly
what a distilled out-of-domain teacher later produced. That investigation is
[`03_teacher-localization.md`](03_teacher-localization.md).

---

## 11. Alignment with Qi et al. 2024

An equation-by-equation audit (eq. 2–27, Tables I/II/IV/VIII) against
`src/models/udma.py`, `src/models/memory.py`, `configs/model/udma.yaml`.

**Faithful:** the S–T scheme (1 teacher + 2 students, trained on normal data
only), dataset-level per-channel target normalization (eq. 3–5), loss structure
(eq. 6–9, including the SS term), maps and fusion (eq. 23–26, `(1/d)·diff²`,
weights 0.5/0.5/0.5 exactly), hard shrinkage with L1 renormalization (eq. 13),
AdamW.

**Three structural deviations — all pushing the same way**, toward less student
capacity limitation and a more domain-matched teacher, i.e. toward disagreement
collapse:

- **(A) Teacher.** The paper uses a 3-layer CNN (RF 7×7) *distilled from a
  frozen generic pretrained network*, anchoring the feature space out of domain
  **by construction**; spectrum data enters only as distillation input, never as
  a learning target. We use a domain-trained ViT-MAE — the exact opposite.
- **(B) Memory.** The paper uses the Park 2020 variant: cosine similarity, an
  update mechanism, compactness+separateness losses, `concat(z, z̃)`, and
  **M = 10–50 items with 1 query per sample**. We use Gong 2019: dot-product,
  entropy, substitution, **500 slots with 384 per-position queries**. The paper
  is explicit that memory exists to *"limit the learning capacity of the student
  network"* — ours is orders of magnitude more expressive, so it can track the
  teacher even on anomalies.
- **(C) Students.** The paper uses an encoder–decoder with a narrow bottleneck
  (1×8×32) as a declared capacity limiter. We use trunk+head with no
  bottleneck — a deliberate, validated deviation that removes pixel-space
  dilution, but it is the third capacity limiter removed.

**Justified minor deviations.** Top-k aggregation (our anomaly footprint is 2–3
of 384 patches versus at least a full column of their full-resolution map, so
eq. 27's plain mean dilutes) and threshold calibration — the paper calibrates at
1% FAR on anomaly-free samples, which endorses the OFF-noise-core ceiling
described in [`04_candidate-filtering.md`](04_candidate-filtering.md).

**How the deviations were tested.** (A) and (B) were both investigated
empirically rather than assumed. Restoring the paper's memory size (`mem_slots`
500 → 30) produced a strong mechanistic signal that never converted into a
detection gain, and was dropped. Restoring the paper's out-of-domain distilled
teacher produced the best detector built so far but destroyed localization. Both
outcomes are in [`03_teacher-localization.md`](03_teacher-localization.md).
