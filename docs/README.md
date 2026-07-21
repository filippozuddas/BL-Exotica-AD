# Documentation

Design rationale, decision records, and mentor-facing reports for BL-Exotica-AD.

Code carries API documentation only. The *why* — rejected alternatives,
hand-validated numbers, failed experiments worth not repeating — lives here.

## Start here

If you are reading this repository for the first time, read
[`decisions/scoring-history.md`](decisions/scoring-history.md). Five scorer
families failed before the current one worked, and that history explains almost
every design choice in `src/`.

## Decision records

Grouped by topic, kept current as decisions change.

| Document | Covers |
|---|---|
| [`decisions/scoring-history.md`](decisions/scoring-history.md) | Why reconstruction-error scoring fails; the five failed scorer families; the disagreement probe; how UDMA was reached; standing architectural constraints |
| [`decisions/teacher-localization.md`](decisions/teacher-localization.md) | Domain-matched vs. out-of-domain teacher; the receptive-field/architecture/objective hypotheses and their refutation; detection vs. localization trade-off |
| [`decisions/candidate-filtering.md`](decisions/candidate-filtering.md) | Detection thresholds and the OFF-noise ceiling; ON/OFF contrast and short-list rules; the rejected column-coherence gate |

## Design specifications

| Document | Covers |
|---|---|
| [`design/udma-spec.md`](design/udma-spec.md) | UDMA-GBT specification (Q1–Q10), with acceptance bars pre-registered before any run |
| [`design/udma-paper-alignment.md`](design/udma-paper-alignment.md) | Equation-by-equation audit against Qi et al. 2024; the three structural deviations and the plan that addressed them |

## Reports

Point-in-time writeups. Not maintained — read them as dated.

| Document | Date |
|---|---|
| [`reports/2026-07-06_mentor-report.md`](reports/2026-07-06_mentor-report.md) | 2026-07-06 — architecture choices, implementation, and results (Italian) |
| [`reports/2026-06-25_scoring-diagnosis.md`](reports/2026-06-25_scoring-diagnosis.md) | 2026-06-25 — reconstruction-error diagnosis (English) |

## `archive/`

Superseded session handoffs and plans for approaches since abandoned. Kept for
provenance; **conclusions here may be outdated or explicitly retracted** — the
decision records above supersede them.

## References

1. Ma et al. 2023 — *A Deep-learning Search for Technosignatures from 820 Nearby Stars* — [arXiv:2301.12670](https://arxiv.org/abs/2301.12670)
2. Lacki et al. 2020 — *One of Everything: The Breakthrough Listen Exotica Catalog* — [arXiv:2006.11304](https://arxiv.org/abs/2006.11304)
3. Gong et al. 2019 — *Memorizing Normality to Detect Anomaly (MemAE)* — [arXiv:1904.02639](https://arxiv.org/abs/1904.02639)
4. He et al. 2022 — *Masked Autoencoders Are Scalable Vision Learners* — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
5. Qi et al. 2024 — *Unsupervised Spectrum Anomaly Detection With Distillation and Memory-Enhanced Autoencoders* — IEEE IoT Journal 11(24):39361
