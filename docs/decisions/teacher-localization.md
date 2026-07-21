# Teacher choice and the localization problem

Why UDMA uses a **domain-matched ViT-MAE teacher** instead of the paper's
out-of-domain distilled CNN, and the record of the experiments that settled it.

Rationale behind `scripts/debug/teacher_row_leakage_test.py`,
`scripts/debug/teacher_sensitivity_test.py`, and the teacher block of
`configs/model/udma.yaml`.

**Related:** [`../design/udma-paper-alignment.md`](../design/udma-paper-alignment.md) ·
[`candidate-filtering.md`](candidate-filtering.md)

---

## 1. The deviation from the paper

Qi et al. 2024 use a 3-layer CNN teacher (RF 7×7) *distilled from a frozen,
generic pretrained network P*, à la Bergmann's "Uninformed Students". The
teacher's feature space is anchored **out of domain by construction**: spectrum
data enters only as distillation input, never as a learning target.

This project instead uses a ViT-MAE trained self-supervised **on the domain** —
the exact opposite. The deviation was deliberate but needed justifying, because
all three structural deviations from the paper (teacher, memory variant,
student bottleneck) push in the same direction: toward collapse of the
student-disagreement signal.

## 2. The symptom: diffuse anomaly maps

The distilled CNN teacher (read at ResNet-18 `layer3`) produced **diffuse** UDMA
anomaly maps. An ON-only signal (Voyager-1) bled into the OFF rows, collapsing
`on_off_contrast` from ~32 (ViT-MAE teacher) to ~2.3 and breaking every ON/OFF
short-list filter downstream.

The mechanism is specifically **cross-observation leakage**. The `(96, 1024)`
input stacks 6 observations of 16 time bins each. For a teacher feature at an
OFF row to move when signal is injected only into ON observations, the
teacher's *vertical* receptive field must cross the 16-bin observation
boundary:

| Stage | Grid | Vertical RF | Spans (obs of 16 px) |
|---|---|---|---|
| stem (conv1+maxpool) | (24, 256) | ~11 px | < 1 → no cross-obs leak *in theory* |
| layer1 | (24, 256) | ~43 px | ~2.7 → partial |
| layer2 | (12, 128) | ~99 px | ~6 → whole cadence |
| layer3 (deployed) | (6, 64) | ~211 px | all → maximal |

## 3. The probe

`teacher_row_leakage_test.py` measures leakage **directly on the teacher's
features** — no students, no distillation, no training. Inject an ON-only line
into a quiet frame, take the feature displacement `‖P(x+s) − P(x)‖` per
(row, col) cell, and compare the response in ON feature-rows vs OFF
feature-rows. A localizing teacher keeps displacement in ON rows (high ON/OFF
ratio); a leaky one spreads it (ratio → 1).

The test is clean because:

1. `inject_narrowband_on_only` renders the signal **only** into obs 0/2/4 —
   OFF frames are byte-identical.
2. `preprocess_raw` normalizes **per observation** (the `gbt_fine` normalization
   fix), so an ON injection cannot shift OFF pixels via renormalization.

Any OFF-row feature motion is therefore pure receptive-field leakage. Feature
rows map to observations by the stage's vertical downsampling: obs `i` occupies
rows `[i*(nh/6) : (i+1)*(nh/6))`, ON = obs {0, 2, 4}. No column window or
morphology assumption is used (row max over the full frequency axis).

Cost: `n_frames × len(snr_list) × len(stages)` frozen forwards — minutes on
GPU, **no training**.

## 4. Three hypotheses, all refuted (2026-07-16)

The probe supports several `--architecture` modes, each isolating one variable.
All three ran as zero-training controls:

| Hypothesis | Control | Result |
|---|---|---|
| **Receptive field** is the lever | ResNet-18 RF sweep across stem/layer1/layer2/layer3 | **Refuted** — ratio stayed ~2–3 at *every* depth, including the small-RF stem |
| **Architecture** (CNN vs Transformer) is the lever | `--architecture udma_student` on two trained students sharing one trunk but different teacher targets | **Refuted** — the old-target student localizes, the new-target one does not; same architecture both times |
| **Pretraining objective** (classification vs reconstruction) is the lever | `--architecture convnextv2_fcmae` — same ImageNet domain, same CNN family, but masked-patch reconstruction pretraining | **Refuted** — ConvNeXtV2-FCMAE ≈ ResNet, stayed diffuse |

The `convnextv2_fcmae` control matters most: the resnet18-vs-vit_mae comparison
confounds *domain* (ImageNet vs GBT) with *objective* (classification vs
reconstruction), changing both at once. FCMAE holds domain wrong while fixing
the objective. It stayed diffuse. (It was chosen over a ViT-MAE-on-ImageNet
alternative because it is fully convolutional, so the non-224×224 input needs
no positional-embedding interpolation.)

**Conclusion: only domain-matched GBT training ever localizes.** Domain
matching is necessary regardless of architecture, receptive field, or
pretraining objective. Stop hunting backbones.

## 5. Detection and localization are orthogonal

A separate result complicates the picture. The distilled CNN teacher, despite
failing to localize, is the **best detector yet built**:

| Teacher | det@3σ_cad, SNR 15 / 20 / 30 |
|---|---|
| CNN (distilled, `d13`) | **93.33 / 98.44 / 100.0** |
| ViT-MAE (old, domain-matched) | 80.9 / 94.7 / 99.8 |

The same checkpoint that wins detection cannot localize. Different teachers win
each axis, so the open design question is **detection-tuning on ViT-MAE vs. a
two-model pipeline**, not "which teacher is better".

## 6. What the teacher actually buys

Component attribution (2026-07-20, `udma_component_attribution.py`) at matched
energy against **real RFI**: UDMA beats trivial statistics by +0.07–0.09 AUC.
The memory unit earns its place.

Two caveats worth keeping visible:

- Clean negatives give the **opposite** answer. Only matched-energy-vs-real-RFI
  is a meaningful benchmark here.
- Equal-weight fusion of the three maps dilutes `ss`, which is by far the best
  single term at SNR 10.

## 7. Related closed threads

- **`mem_slots` = 30 (paper-faithful Park variant).** Killed 2026-07-15. The
  mechanistic signal was strong throughout (`val_ss` 2.2–2.8×) but never
  converted to a Tier-A detection gain — flat or negative at two checkpoints
  9 epochs apart. `mem_slots` stays at 500.
- **`topk_frac`.** Swept over both students; the optimum is 0.01, not the 0.02
  that was in the production config. No model switch — just the config bump.
- **Benchmark reproducibility.** A Fase-0 control re-run reproduced
  80.9/94.7/99.8% from an identical checkpoint/config/code where the historical
  record said 48.9/68.0/79.3%. The old cadence list is unrecoverable, so the
  new number is canonical.
