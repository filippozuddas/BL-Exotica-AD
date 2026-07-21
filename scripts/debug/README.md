# scripts/debug

Diagnostic and one-off exploration scripts. **Not part of the pipeline** —
`scripts/` proper holds thin CLI entry points; this directory is investigative.
Scripts here are written to answer one question and are kept because the answer
matters, not because they are maintained interfaces.

Most require a preprocessed cache and are meant to run on the data host. Usage
examples in each script's docstring use `/path/to/...` placeholders.

The findings these scripts produced are written up in
[`../../docs/`](../../docs/) — read those first; the scripts
are the evidence, the decision records are the conclusions.

## Teacher / student distillation (UDMA)

Findings: [`docs/03_teacher-localization.md`](../../docs/03_teacher-localization.md)

- `teacher_sensitivity_test.py` — teacher fitness gates (G1–G3b). **Run this before
  adopting any new teacher checkpoint or read-layer.**
- `teacher_row_leakage_test.py` — training-free ON→OFF row-leakage probe; measures
  whether a teacher localizes an ON-only signal or bleeds it into OFF rows.
  Supports four `--architecture` targets, each isolating one variable.
- `resnet_teacher.py` — out-of-domain ResNet-18 teacher wrapper (paper-faithful route)
- `udma_component_attribution.py` — is UDMA earning its complexity? Per-term budget,
  student redundancy, and AUC of each component against trivial baselines
- `udma_anomaly_maps.py`, `smoke_test_udma.py` — map inspection and a fast wiring check

## Scorer separation and baselines

Findings: [`docs/01_scoring-history.md`](../../docs/01_scoring-history.md)

- `encode_separation_test.py` — the main separation harness; includes `DisagreementPair`,
  the zero-training `‖AE(x) − MemAE(x)‖²` probe that motivated UDMA
- `eti_vs_rfi_separation_test.py` — controlled ETI-vs-RFI test holding background,
  morphology, and SNR fixed so cadence occupancy is the only varying axis
- `injection_vs_rfi_test.py` — injection-recovery vs RFI separation
- `statistical_baseline.py` — trivial model-free scorers (energy, max pixel, peak SNR).
  The bar any learned scorer must clear.
- `snr_convention_check.py` — verifies the SNR convention used across injectors

## MemAE memory

- `visualize_memae_memory.py` — real-patch gallery per most/least-used slot, plus a
  usage histogram. Uses real patches rather than decoder synthesis: the Qi et al.
  Fig. 14 decode-a-broadcast-item method produces a checkerboard artifact on this
  repo's per-position memory variant (confirmed against a random-vector control).
  Not applicable to UDMA's memory-augmented student, which has no pixel decoder.
- `injection_memory_addressing.py` — tracks which memory slot an injected
  narrowband-drift signal addresses as a function of SNR. Produced the direct
  causal evidence that injected signals route to explicit narrowband/drift
  prototypes and get redrawn as normal.

## Reconstruction visualization

- `ae_recon_visual.py`, `three_way_recon_visual.py`, `reconstruction_diagnostic.py`

## Search output and data characterization

- `analyze_inference_fp.py` — false-positive analysis on an inference run
- `rfi_composition_analysis.py` — RFI characterization of the training set
- `off_row_heterogeneity_test.py` — OFF-row heterogeneity across a cadence
- `exotica_dist_analysis.py` — distributional analysis of the real `.0000.h5` batch
  (normalization sanity, RFI occupancy, band/cadence completeness, ON vs OFF).
  Runs on the data host; feeds `notebooks/05_exotica_0000_distributions.ipynb`.
- `exotica_cadence_summary.py` — header-only cadence census; runs anywhere
  `data/processed/exotica_0000_headers.csv` exists, no data host needed
- `normalization_diagnostic.py` — preprocessing sanity checks

## Candidate plotting

- `plot_single_candidate.py` — re-plot one candidate from a completed
  `scripts/inference.py` pass without re-scanning the cadence
- `plot_shortlist_candidates.py` — re-plot a whole short list

## Superseded scripts

Diagnostics tied to approaches since abandoned (ON/OFF cadence scoring, SSAST
strategies, and the failed dist384 / GMM / MLP-probe / Mahalanobis scorer
families) are **not included in this repository**. They remain in the git
history, and their conclusions are recorded in
[`docs/01_scoring-history.md`](../../docs/01_scoring-history.md).
