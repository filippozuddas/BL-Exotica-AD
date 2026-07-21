# Documentation

Design rationale and results for BL-Exotica-AD.

Code carries API documentation only. The *why* — rejected alternatives,
hand-validated numbers, and experiments worth not repeating — lives here.

Read in order. Each document answers one question and is kept current; there is
no dated correspondence and no superseded material in this directory.

| # | Document | Answers |
|---|---|---|
| 01 | [Scoring history](01_scoring-history.md) | Why does this project not just use reconstruction error? |
| 02 | [UDMA architecture](02_udma-architecture.md) | What is the production model, and how does it relate to the paper it adapts? |
| 03 | [Teacher localization](03_teacher-localization.md) | Why is the teacher domain-matched instead of paper-faithful? |
| 04 | [Candidate filtering](04_candidate-filtering.md) | How does a scored snippet become a candidate a human looks at? |
| 05 | [Results and limitations](05_results.md) | Does it work, and what is still broken? |

**If you only read one:** [`01_scoring-history.md`](01_scoring-history.md). Five
scorer families failed before the current one worked, and that history explains
almost every design choice in `src/`.

**If you want the honest status:** [`05_results.md`](05_results.md) §6, which
lists known limitations rather than resolved ones.

## Conventions

- **English throughout**, including in code comments and docstrings.
- Results are reported against **bars pre-registered before the run**. This is
  standing discipline after a headline result was retracted as a partition
  artifact (see [`01_scoring-history.md`](01_scoring-history.md) §2.1).
- Superseded documents are deleted, not archived — they remain in the git
  history. A reader should never have to work out which document is current.

## References

1. Ma et al. 2023 — *A Deep-learning Search for Technosignatures from 820 Nearby Stars* — [arXiv:2301.12670](https://arxiv.org/abs/2301.12670)
2. Lacki et al. 2020 — *One of Everything: The Breakthrough Listen Exotica Catalog* — [arXiv:2006.11304](https://arxiv.org/abs/2006.11304)
3. Gong et al. 2019 — *Memorizing Normality to Detect Anomaly (MemAE)* — [arXiv:1904.02639](https://arxiv.org/abs/1904.02639)
4. He et al. 2022 — *Masked Autoencoders Are Scalable Vision Learners* — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
5. Park et al. 2020 — *Learning Memory-guided Normality for Anomaly Detection* — [arXiv:2003.13228](https://arxiv.org/abs/2003.13228)
6. Qi et al. 2024 — *Unsupervised Spectrum Anomaly Detection With Distillation and Memory-Enhanced Autoencoders* — IEEE IoT Journal 11(24):39361
