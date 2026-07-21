# Anomaly-scoring diagnosis: reconstruction error fails, embedding-distance works

**Date:** 2026-06-25
**Author:** Filippo Zuddas
**Audience:** Maura Pilia, Vishal Gajjar
**Scope:** ViT-MAE backbone, GBT `0000.fil` fine product, real SRT background (val split). Synthetic narrowband injections (ON-only, drift 0.3 Hz/s).

---

## TL;DR

1. **Reconstruction error — our primary anomaly score — does not work.** On noise/RFI-dominated data it degenerates into a plain *energy detector*: it flags anything brighter than the noise floor (including strong RFI) and is blind below SNR≈20. Two backbones fail it in opposite ways (the plain AE copies its input → 0 % detection; the masked ViT-MAE collapses to the noise mean → energy threshold).
2. **The model itself is fine.** At *matched energy*, the encoder's learned features still separate an injected narrowband from RFI (AUC 0.75), where hand-crafted shape statistics cannot (AUC 0.53). The failure was the *scoring rule*, not the representation.
3. **A fully-unsupervised fix works.** A Mahalanobis one-class score on the frozen encoder embedding, fit on the **full background** (RFI included, no labels), separates injected narrowband from real RFI with **AUC 0.93 at SNR 10** (82 % detection at a 10 % RFI false-positive rate), rising to ≥0.96 / 100 % by SNR 15. Crucially this holds even though energy works *against* the signal in this test.

**Recommendation:** drop reconstruction-error scoring; adopt Mahalanobis-on-embedding (full-background fit) as the anomaly score. Then validate on the morphologies that justify the autoencoder over turboSETI (broadband / wideband / transient).

---

## 1. The problem: reconstruction error is an energy detector

The pipeline scored anomalies by reconstruction MSE (`(x − x̂)²`). A reconstruction diagnostic on the 1 M-snippet ViT-MAE showed the model outputs ≈0 regardless of input (reconstruction energy < 1 % of input energy). The training loss had converged to ≈ the energy of the input itself — i.e. the model contributes nothing; the "error" just measures how much energy is in the frame.

A side-by-side SNR sweep (n = 500, identical harness, ON-only narrowband injection) confirms both backbones fail, oppositely:

| Backbone | Behaviour | det@3σ vs SNR |
|---|---|---|
| Plain AE (`convae`) | Copies the input (full visibility) → reconstructs the signal too | **0 % at every SNR (3–50)** |
| ViT-MAE (75 % mask) | Collapses to the noise mean → residual = energy | 0 % until SNR 15, 92 % at SNR 20 |

Mechanism: masked patches of near-white noise cannot be *predicted* from visible noise (no correlation), so the MSE-optimal output is the mean (≈0 after log1p + median/MAD). The plain AE avoids the collapse only by *copying*, which then also reconstructs anomalies. Either way, reconstruction MSE ≈ input energy → no morphology sensitivity, and the SNR≈20 "floor" is just an energy threshold that strong RFI also trips.

## 2. Is the representation any good? (matched-energy test)

The key question: does the encoder's internal representation carry *morphology*, or only energy? Naively comparing signal-vs-quiet is circular — a signal *is* extra energy. We controlled for it: pair each RFI snippet to an injected snippet of nearly identical total energy (1:1 nearest-neighbour, caliper 0.02, n = 108/class), then ask a linear readout to separate them.

| Feature set | AUC (injected vs RFI, matched energy) |
|---|---|
| Total energy (sanity — should be chance) | 0.56 |
| Hand-crafted shape stats (peakiness, kurtosis, top-pixel fraction) | **0.53** (chance) |
| Encoder embedding | **0.75** |

The embedding separates narrowband from RFI at matched energy, and does so *beyond* what trivial shape statistics capture. The encoder learned genuine structure (drift-line geometry, ON-only cadence pattern). **The representation is sound; the scoring rule was the problem.**

(Note: this 0.75 is a *supervised* upper bound. It says the information is present, not that an unsupervised scorer can reach it — see §3.)

## 3. A fully-unsupervised scorer that works

Per project principle, the search must stay unsupervised (no ETI/RFI labels). We test one-class detectors on the frozen embedding.

A first attempt fitting on *quiet only* (Mahalanobis, full covariance) detected from SNR≈7 — but this is misleading: it measures "distance from quiet", which flags RFI too (RFI is far from quiet by construction). It cannot win the false-positive battle.

The operational test fits the detector on the **full background** (noise + RFI together, as in a real search where you cannot pre-label), then asks the question that matters: *does an injected signal stand out from real RFI?* Held-out split (70/30); negatives = held-out RFI (75); positives = injection into held-out quiet (79). Energy here works **against** the signal (injected is low-energy, RFI is high-energy), so any separation is morphology, not energy.

| SNR | AUC (signal vs RFI) | Detection @ 10 % RFI false-positive |
|---|---|---|
| 7 | 0.80 | 35 % |
| **10** | **0.93** | **82 %** |
| 12 | 0.95 | 95 % |
| 15 | 0.96 | 100 % |
| 20+ | ≥0.96 | 100 % |

The unsupervised scorer keeps the signal while rejecting RFI, from SNR≈10. Reconstruction error, by contrast, ranked RFI *above* signals.

## 4. Conclusion & recommendation

- **Drop reconstruction-error scoring** as the primary anomaly score.
- **Adopt Mahalanobis-on-embedding, fit on the full background** (Ledoit-Wolf shrinkage covariance, no labels, no retraining of the existing checkpoint).
- This is a methodology fix, not a model change: the same trained ViT-MAE encoder is reused.

## 5. Next steps (validation before production)

1. **Other morphologies.** Narrowband (tested here) is turboSETI's domain. The autoencoder's value-add is broadband / wideband / transient signals — repeat the operational test (§3) with the broadband generator and on products `0001`/`0002`. This is the decisive test of the "broader range of signals" goal.
2. **Operating point.** 10 % RFI false-positive is illustrative; tune against the Phase-3 post-processing (spatial dedup, RFI excision) to reach a visually-validatable candidate count.
3. **Robustness.** Sweep drift rate; confirm across checkpoints.
4. **Productionise** the full-background fit + Mahalanobis scorer in `src/search/scorer.py`.

## 6. Training-objective improvement directions (second-order)

The fix in §4 is a *scoring* change and needs no retraining. A separate, optional lever is the **training objective** that shapes the encoder embedding. This matters because the current representation was learned only as a *by-product* of masked reconstruction — which is why the matched-energy separation is moderate (AUC 0.75, not 0.95) and the operational floor is SNR≈10 rather than lower.

Principle: **the latent space is shaped by what the pretext task forces the encoder to attend to.** Masked autoencoding (MAE) asks "predict the hidden patches"; on noise-dominated data the hidden patches are mostly unpredictable, so the learning signal is sparse and the encoder learns structure only weakly. Other objectives force a denser, more purposeful representation. The **scorer (Mahalanobis-on-embedding) stays fixed**; only the teacher changes:

```
[teacher: MAE → denoising]  →  better encoder  →  [scorer: Mahalanobis] (unchanged)
```

| Objective | Idea | Assessment for this data |
|---|---|---|
| **Lower mask ratio** (e.g. 0.5) | Same MAE, less mean-collapse, more visible context | Cheapest tweak; try first |
| **Denoising** (corrupt real frames → recover) | Dense gradient on every pixel; forces a full model of "normal"; naturally erases unfamiliar/faint structure → anomalies land off-manifold | ⭐ Most promising general improvement; fully unsupervised (no clean target — add noise to real frames) |
| **Predictive ON→OFF** (Zhang et al. 2019) | Predict ON observations from OFF; a sky-localised signal is unpredictable from OFF | ⚠️ Physically ideal for SETI, **but our data violates it**: RFI is *intermittent*, so OFF does not reliably predict ON — this is exactly why cadence-aware *scoring* was demoted (4.4× baseline variance). Empirical caution before investing |
| **Contrastive** (DINO/SimCLR-style) | Embeddings invariant to augmentations | Can yield highly discriminative embeddings; more complex to implement |

These are **not blockers**: a working unsupervised pipeline already exists (§3–4). They are optimisations to push the detection floor lower and improve robustness on harder (non-narrowband) morphologies, best weighed against the §5 validation results before investing.

## Appendix — reproducibility

- Diagnostic script: `scripts/debug/encode_separation_test.py` (block 1 encoder-collapse check / block 2 naive distance / block 3 matched-energy morphology AUC / block 4 unsupervised OCC fit-on-quiet / block 5 operational signal-vs-RFI).
- SNR sweep: `scripts/debug/cadence_snr_sweep.py` (recon, AE vs ViT-MAE).
- ViT-MAE checkpoint: `outputs/training/20260624_084754_057f87c/checkpoints/epoch=006-val_loss=2.1715.ckpt`.
- Plain AE checkpoint: `outputs/training/20260618_094052_e880462/checkpoints/epoch=019-val_loss=1.4737.ckpt`.
- Cache: `data/processed/cache_gbt_fine/` (val split); injection: ON-only narrowband, drift 0.3 Hz/s, `inject_narrowband_on_only` in `scripts/debug/injection_vs_rfi_test.py`.
- All numbers above: n_samples = 1000 (block 5), 500 (blocks 1–4), seed 42.
