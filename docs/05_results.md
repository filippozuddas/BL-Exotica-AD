# Results and current limitations

Validation results for the production UDMA scorer, and an honest list of what is
known to be wrong or unverified.

**Last updated:** 2026-07-21
**Model:** [`02_udma-architecture.md`](02_udma-architecture.md) ·
**Bars:** §9 of that document, pre-registered before any run

---

## 1. Acceptance bars — all passed

Full training run, ~560k real cadence snippets, 56/9/28 split. Evaluated on 3
independent seeds, on both train and never-seen validation.

| Bar | Pre-registered criterion | Result | |
|---|---|---|---|
| **B1** matched-energy separation | AUC ≥ 0.80 on all seeds and above the prior probe (0.770–0.788) | **0.884 / 0.887 / 0.901** | ✅ +0.10–0.12 |
| **B2** mid-SNR detection | TPR@10%FP ≥ 70% @ SNR 12, ≥ 40% @ SNR 10 | **100% / 100%** (probe: 35–60%, 4–15%) | ✅ |
| **B3** low-SNR detection | TPR@10%FP > 0% at SNR 5 and 7 (probe was 0%) | **68–92% @ SNR 5, 97–100% @ SNR 7** | ✅ |
| **B4** generalization | val AUC within ±0.03 of train | Δ = 0.000–0.024 | ✅ |
| **B5** harness sanity | energy alone must not already win | margin ≥ +0.30 over trivial | ✅ |

All five bars passed, on all three seeds, on both splits. B3 is the one that
mattered most: the low-SNR operating point is precisely what every prior scorer
failed to reach, and what UDMA was built to unlock.

UDMA therefore replaces both the reconstruction scorers (AE / MemAE / ViT-MAE)
and the pixel-space disagreement probe as the production scorer.

## 2. Qualitative behaviour

**Anomaly maps localize.** On pure noise, real RFI, and injected signal, the
`(6, 64)` maps put the peak at the true position — the RFI line on its actual
time interval, the injected signal on the correct time/frequency cell. The
global-attention smearing accepted as a design risk did **not** materialize. On
a broadband RFI feature that drifts along a curve, the map peak *follows the
curve*, confirming the mechanism captures non-trivial morphology rather than a
static hot pixel.

**Known RFI is suppressed.** A narrowband line present continuously through a
cadence is almost entirely suppressed by the students, which saw it often in
training as normal; residual survives only where the line departs from the
learned pattern. This is the intended "suppress the known, flag the unexpected"
behaviour.

**Blind to ON/OFF occupancy, by construction.** UDMA does not distinguish
ON-only from persistent signals (AUC ~0.49, chance). This is correct: there is
no cadence objective in training, and ON/OFF discrimination is the downstream
stage's job ([`04_candidate-filtering.md`](04_candidate-filtering.md)).

## 3. Injection-recovery sensitivity

10 real cadences, 1440 injections. The top-k score gives a smooth, monotonic
detection curve:

| SNR | 3 | 15 | 20 | 50 |
|---|---|---|---|---|
| Detection | 2.8% | 64% | 89% | 100% |

Mean-based scoring is structurally weaker at every SNR, confirming that grid
dilution is a real effect even on a 384-position grid — this is why `topk` is
the production aggregation.

Two measurement caveats worth knowing, both discovered later:

- Early weak injection-recovery numbers were a **pooled-threshold artifact**, not
  a model problem. Between-cadence variance dominates (η² 0.94 for topk), so a
  threshold pooled across cadences is invalid. With a per-cadence threshold the
  detection curve recovers to 21/64/82/94/98% at SNR 10/15/20/30/50.
- Detection rates measured against a quiet-only baseline mislead badly at small
  n: one arm read 100% at n=50 and **0% at n=500**. Sample ≥500 per stratum.

## 4. Does the architecture earn its complexity?

Component attribution at matched energy against **real RFI**: UDMA beats trivial
statistics by **+0.07–0.09 AUC**. The memory unit earns its place.

Two caveats that matter for interpretation:

- Against **clean** negatives the comparison gives the opposite answer. Only
  matched-energy-versus-real-RFI is a meaningful benchmark here — against empty
  sky, any brightness detector looks good.
- Equal-weight fusion **dilutes `map_ss`**, which is by far the strongest single
  term at SNR 10. Weight tuning is an open, unexploited lever.

## 5. Search on real data

A full inference pass over the Exotica `0000.h5` held-out set (364 cadences) has
been run. Short-list behaviour on 280k held-out candidates:

- **73% are rejected at the OFF-noise ceiling before the ON/OFF logic engages.**
  The ceiling, not the ON/OFF rules, is the binding constraint on the false-alarm
  rate.
- The accepted short list has a median ON/OFF contrast of only **1.21** — weak.

A real-signal control is available: on a Voyager-1 cadence the pipeline ranks
the true carrier first, with an ON/OFF contrast of ~32 and a clean 3/3 short
list with no noise entries.

## 6. Known limitations

Listed because they are real, not because they are resolved.

**Short-list threshold does not scale.** The `thresh_5` floor was validated at
Voyager scale (~2046 snippets per cadence, ~15% band coverage). At full-band
Exotica scale (~1000 clusters per cadence) the multiple-comparisons burden is far
higher and false positives reappear. Fixable post-hoc from the saved CSVs
without re-running the scan. See
[`04_candidate-filtering.md`](04_candidate-filtering.md) §3.1.

**`off_ceiling_probe` default is mistuned for production.** The default of 300
was set on the narrowband Voyager file; on full-band Exotica cadences it covers
only ~0.2% of the band. Production runs should use ~2000–3000.

**Detection and localization are won by different models.** The distilled CNN
teacher detects better (93.3/98.4/100.0 at SNR 15/20/30 vs 80.9/94.7/99.8) but
does not localize; the domain-matched teacher localizes but detects worse. There
is currently no single checkpoint that wins both. See
[`03_teacher-localization.md`](03_teacher-localization.md) §5.

**One historical benchmark is unreproducible.** A control re-run with verified
identical checkpoint, config, and code produced 80.9/94.7/99.8% where the
recorded figures were 48.9/68.0/79.3%. The only unverifiable variable is the
exact cadence list of the original run, whose log was lost. The new numbers are
canonical; the old ones should not be cited.

**Patch-geometry refinement is deferred**, documented rather than fixed.

## 7. Not yet addressed

- **`0001.h5` (broadband transients)** — the natural next extension of the
  "broader morphology" goal in the project specification, once `0000.h5` is
  consolidated. Generators and DM-sweep sizing exist; no model has been trained.
- **`0002.h5` (wideband / modulated)** — frame geometry still to be determined.
- **turboSETI cross-comparison** — the optional final comparison against
  traditional narrowband SETI, and goal #5 of the project specification.
