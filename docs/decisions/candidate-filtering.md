# Candidate filtering — thresholds, ON/OFF logic, and short-list rules

Rationale behind `src/search/candidates.py`. The code there carries API
documentation only; the reasoning, the hand-validated numbers, and the
explicitly rejected alternatives live here.

**Related:** [`scoring-history.md`](scoring-history.md) ·
[`teacher-localization.md`](teacher-localization.md) ·
[`../design/udma-spec.md`](../design/udma-spec.md)

---

## 1. Geometry: why grid rows map to observations

UDMA's `(6, 64)` feature grid comes from a `(16, 16)` patch over the
`(96, 1024)` input — one grid row per 16-row observation. Grid rows therefore
correspond 1:1, in order, to the 6 observations of a standard ABACAD cadence
(ON, OFF, ON, OFF, ON, OFF). Every ON/OFF rule below is built on that mapping.

A real target-locked signal scores high on ON rows and low on OFF rows;
persistent RFI scores similarly on both.

## 2. Spatial deduplication (`cluster_candidates`)

`scripts/inference.py` scans each cadence with overlapping windows
(`stride_infer < fchans`), so one wide or strong line trips many adjacent
windows above threshold. `cluster_candidates` groups them into one candidate
per contiguous run, keeping the peak-scoring snippet — a 1-D
connected-component grouping on the frequency axis.

This is the "spatial deduplication" stage of Phase 3. It is **pure
deduplication of the model's own score, with no new discriminating logic**: it
does not decide RFI vs. technosignature, it only collapses duplicate detections
of the same event into one entry.

The pattern is ported from the sibling `rst` project
(`rst_seti.inference.engine.InferenceEngine.cluster_detections`), which used
the same sliding-window geometry (snippet width 1024 channels, stride 512) but
applied it to a binary ETI/RFI probability instead of a continuous score.

## 3. The detection threshold: why a Gaussian 3σ threshold was replaced

**Problem (2026-07-14).** The GBT short list came back as pure noise. The cause
was the threshold, not the model: a Gaussian `median + 3*MAD_sigma` threshold
computed over the pooled score distribution sat *below* the OFF-row noise
ceiling.

Hand-validated on cad02:

| Quantity | Value |
|---|---|
| Gaussian 3σ threshold (`thresh_3`) | **0.16** |
| Real OFF-row noise ceiling | **~0.9** |
| Raw 0.999 quantile of pooled OFF values | **~31** |

The Gaussian threshold is an order of magnitude below the real noise ceiling —
MAD is supposed to be robust, but once RFI dominates the pool it is pulled
down by the very contamination it is meant to resist.

**Rejected alternative: a raw high quantile of the OFF values.** That goes too
far the other way — it is dominated by the RFI tail (~31 above), so it would
discard any candidate that RFI does not outshine.

**Adopted: `off_noise_ceiling`.** OFF-target rows carry no target signal by
construction, so any high value there is noise or RFI, never a real detection.
The function iteratively 3σ-clips the pooled OFF values (median/MAD, the same
scheme as `bandpass_correct`) to isolate the **noise core**, then takes a high
quantile of the surviving core. This lands at the real ceiling (~0.9) rather
than the noise floor (0.16) or the RFI tail (31).

Sanity check on the same run: cad02's candidate count decays 332 → 4 → 1 → 0
as the filter chain tightens, which is the correct answer for Orion (no
expected signal).

### 3.1 Known scale limitation

The `thresh_5` short-list floor was validated only at Voyager scale
(~2046 snippets/cadence, ~15% band coverage). At full-band Exotica scale
(~1000 clusters/cadence) the multiple-comparisons burden is far higher and
HIP114176-style false positives reappear. This is a **threshold-scale bug, not
a model bug**, and is fixable post-hoc from the saved CSVs without re-running a
scan.

Relatedly, `off_ceiling_probe` defaults to 300, tuned on the narrowband Voyager
file. On real full-band Exotica cadences that is only ~0.2% coverage — scale it
to ~2000–3000 for production runs.

## 4. `on_off_contrast` — ranking diagnostic

Reports mean-ON / mean-OFF for one snippet. Two design points:

**Why per-row hit counts exist.** ON-mean vs OFF-mean alone cannot distinguish
"present in every ON pointing" from "one transient that happened to land inside
a single ON block". A single-scan RFI burst produces a high contrast too —
observed 2026-07-06, where the top-contrast SRT candidate was a broadband
feature confined to one ON block and absent from the other two.
`n_on_hits`/`n_off_hits` count how many *individual* rows independently clear
the per-cadence threshold, so a genuine target-locked signal must hit most ON
rows and no OFF rows.

**Why a column window.** The search runs over a small column window around the
peak column rather than the exact column, tolerating a few channels of drift
between observations taken minutes apart.

This function is a **pure diagnostic**: it computes numbers and ranks plots. It
never thresholds or discards anything — that decision stays with the human
reviewer.

## 5. `full_row_hits` — short-list volume reduction

Separate from `on_off_contrast`, which still drives plot ranking.

**Why the column restriction is dropped.** `on_off_contrast`'s `col_window` is
anchored to a single peak column shared by all 6 rows. A fast or non-linear
drifter that shifts columns block-to-block falls outside that window and is
misread as OFF-absent — documented case: a satellite-like chirp visible in all
6 blocks scored `n_off_hits=0`. `full_row_hits` takes each row's own max over
the whole frequency axis, trading that blind spot for a different one (an
unrelated OFF-row event anywhere in the snippet's span now counts as a hit).

Used **only** to decide short-list membership (shown for manual vetting vs.
kept in the full CSV), never for ranking and never as a silent discard.

**Short-list rule.** `n_on_hits_full >= 2` AND not `off_leak`, where `off_leak`
requires ≥2 OFF rows to clear `threshold` *and* reach at least `leak_frac`
(default 0.3) of the weakest ON row's peak.

**Why the magnitude gate.** Added 2026-07-16 after a real-signal check: the
genuine Voyager-1 carrier (`on_off_contrast` 32×) was binary-rejected by two
low-amplitude OFF blips averaging ~3% of the ON magnitude. The gate stops
barely-above-threshold OFF noise from killing an otherwise overwhelming
candidate. Persistent RFI and fast chirps — the cases this filter exists to
catch — still trip `off_leak`, because their OFF-row magnitude is comparable to
their own ON-row magnitude, not merely above the noise floor.

### 5.1 Rejected: column-coherence gate

Proposed 2026-07-16: require each row's hit to land near the strongest ON row's
peak column. **Rejected on the arithmetic.**

At this product's grid resolution (~45 Hz/column) and with ~600 s between ON
blocks, even the project's *median* drift rate (0.3 Hz/s) shifts the peak by
~4 columns block-to-block:

```
0.3 Hz/s × 600 s = 180 Hz ≈ 4 columns
```

A tight column tolerance would therefore silently reject genuine drifting
technosignatures — precisely this pipeline's stated differentiator over
narrowband-only turboSETI. The trade was: a handful of visually-obvious noise
blips (cheap for a human reviewer to dismiss) in exchange for **silent false
negatives on the exact signals being searched for**. Not worth it.

## 6. Where the short list actually loses candidates

Measured 2026-07-21 over 280k heldout candidates: **73% die at the OFF-noise
ceiling before the ON/OFF logic engages at all.** `off_ceiling` dominates
FAR@1%, so the ceiling — not the ON/OFF rules — is the binding constraint. The
accepted short list has a median ON/OFF contrast of only 1.21, which is why
§3.1's scale limitation matters more than any refinement to §4–§5.
