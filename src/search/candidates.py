"""
Candidate post-processing: frequency-adjacency deduplication of anomaly scores.

Ports the connected-component clustering pattern validated in the sibling
``rst`` project (``rst_seti.inference.engine.InferenceEngine.cluster_detections``,
same sliding-window geometry: snippet width 1024 channels, stride 512) to this
project's continuous anomaly score (instead of a binary ETI/RFI probability).

Pure deduplication of the model's own score — no new discriminating logic:
``scripts/inference.py`` scans each cadence with overlapping windows
(``stride_infer < fchans``), so a single wide or strong line triggers many
adjacent windows above threshold. This groups those into one candidate per
contiguous run, keeping the peak-scoring snippet. This is exactly the
"spatial deduplication" post-processing stage named in CLAUDE.md's Phase 3 —
it does not decide RFI vs. technosignature, it only collapses duplicate
detections of the same event down to one entry.
"""

import numpy as np
import pandas as pd

__all__ = ["cluster_candidates", "on_off_contrast"]


def _summarize_cluster(cluster_idx: np.ndarray, f_starts: np.ndarray,
                        scores: np.ndarray, fchans: int, df: float) -> dict:
    chunk_scores = scores[cluster_idx]
    chunk_f = f_starts[cluster_idx]
    peak_local = int(np.argmax(chunk_scores))
    f_peak = int(chunk_f[peak_local])
    return {
        "f_start_peak": f_peak,
        "freq_mhz_peak": f_peak * df / 1e6 if df else 0.0,
        "peak_score": float(chunk_scores[peak_local]),
        "mean_score": float(chunk_scores.mean()),
        "n_snippets": int(len(cluster_idx)),
        "f_start_first": int(chunk_f[0]),
        "f_start_last": int(chunk_f[-1]),
        "cluster_width_channels": int(chunk_f[-1] - chunk_f[0] + fchans),
    }


def cluster_candidates(
    f_starts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    stride: int,
    fchans: int,
    df: float = 0.0,
) -> pd.DataFrame:
    """Group adjacent above-threshold snippets into distinct candidates.

    Two snippets belong to the same cluster if their ``f_start`` differs by at
    most ``stride`` (i.e. they are adjacent/overlapping in the sliding-window
    grid); a below-threshold snippet breaks the chain — a 1D connected-component
    grouping on the frequency axis, identical in spirit to
    ``InferenceEngine.cluster_detections`` in the ``rst`` project.

    Args:
        f_starts: ``(N,)`` window start channel for every scored snippet in
            one cadence (any order — sorted internally).
        scores: ``(N,)`` anomaly score per snippet, same order as ``f_starts``.
        threshold: candidates are snippets with ``score > threshold`` (e.g.
            ``median + k * MAD_sigma`` from ``robust_stats``).
        stride: sliding-window stride (``frame.stride_infer``) — the max gap
            between consecutive ``f_start`` values to still count as the same
            event.
        fchans: window width in channels, used for ``cluster_width_channels``.
        df: Hz/channel, used to also report ``freq_mhz_peak`` (0 = omit).

    Returns:
        DataFrame, one row per cluster, sorted by ``peak_score`` descending:
        ``f_start_peak``, ``freq_mhz_peak``, ``peak_score``, ``mean_score``,
        ``n_snippets``, ``f_start_first``, ``f_start_last``,
        ``cluster_width_channels``. Empty (with these columns) if nothing
        exceeds ``threshold``.
    """
    columns = [
        "f_start_peak", "freq_mhz_peak", "peak_score", "mean_score",
        "n_snippets", "f_start_first", "f_start_last", "cluster_width_channels",
    ]
    f_starts = np.asarray(f_starts)
    scores = np.asarray(scores)
    order = np.argsort(f_starts)
    f_starts, scores = f_starts[order], scores[order]

    above = scores > threshold
    if not above.any():
        return pd.DataFrame(columns=columns)

    idx = np.flatnonzero(above)
    rows = []
    start = 0
    for i in range(1, len(idx)):
        gap = f_starts[idx[i]] - f_starts[idx[i - 1]]
        if gap > stride:
            rows.append(_summarize_cluster(idx[start:i], f_starts, scores, fchans, df))
            start = i
    rows.append(_summarize_cluster(idx[start:], f_starts, scores, fchans, df))

    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("peak_score", ascending=False).reset_index(drop=True)


def on_off_contrast(
    anomaly_map: np.ndarray,
    col_window: int = 3,
    on_rows=(0, 2, 4),
    off_rows=(1, 3, 5),
    eps: float = 1e-8,
    threshold: float = None,
) -> dict:
    """ON/OFF contrast + consistency diagnostic for one snippet's UDMA anomaly map.

    UDMA's ``(6, 64)`` feature grid comes from a ``(16, 16)`` patch on the
    ``(96, 1024)`` input, i.e. one grid row per 16-row observation — so grid
    rows correspond 1:1, in order, to the 6 observations of the cadence
    (standard ABACAD: ON, OFF, ON, OFF, ON, OFF). A real signal present only
    when the telescope points at the target scores high on ON rows and low
    on OFF rows; persistent RFI scores similarly on both. But ON-mean vs
    OFF-mean alone can't tell "present in every ON pointing" apart from "one
    transient event that happened to land inside a single ON block by
    coincidence" — a single-scan RFI burst can produce a high contrast too
    (observed 2026-07-06: the top-contrast SRT candidate turned out to be a
    broadband feature confined to one ON block, absent from the other two).
    ``n_on_hits``/``n_off_hits`` (only computed if ``threshold`` is given)
    count how many *individual* ON/OFF rows independently clear the same
    per-cadence detection threshold used for clustering — a real target-locked
    signal should hit most/all ON rows and no OFF rows, not just one.

    Searches a small column window around the snippet's peak column (rather
    than the exact column) to tolerate the signal drifting a few channels
    between observations taken minutes apart.

    Pure diagnostic: computes numbers, does not threshold or filter
    anything — the decision stays with the human reviewer.

    Args:
        anomaly_map: ``(6, 64)`` map from ``UDMA.anomaly_map`` /
            ``anomaly_map_components`` for one snippet.
        col_window: half-width, in grid columns, of the drift-tolerance
            search window around the peak column.
        on_rows: grid row indices corresponding to ON-target observations.
        off_rows: grid row indices corresponding to OFF-target observations.
        eps: floor for the OFF mean to avoid division by ~0.
        threshold: per-cadence detection threshold (e.g. ``median + 3*MAD_sigma``
            of the whole-cadence ``topk``/``recon``/``max`` score, same units
            as ``anomaly_map`` since the scalar score is a reduction over this
            same map) — if given, adds ``n_on_hits``/``n_off_hits``.

    Returns:
        dict with ``on_off_contrast`` (mean ON / mean OFF; higher is more
        target-like), ``on_mean``, ``off_mean``, and if ``threshold`` is
        given, ``n_on_hits``, ``n_off_hits`` (counts out of ``len(on_rows)``/
        ``len(off_rows)``).
    """
    nh, nw = anomaly_map.shape
    col_peak = int(np.argmax(anomaly_map.max(axis=0)))
    lo = max(0, col_peak - col_window)
    hi = min(nw, col_peak + col_window + 1)

    on_idx = [r for r in on_rows if r < nh]
    off_idx = [r for r in off_rows if r < nh]
    on_row_vals = anomaly_map[on_idx, lo:hi].max(axis=1) if on_idx else np.array([])
    off_row_vals = anomaly_map[off_idx, lo:hi].max(axis=1) if off_idx else np.array([])
    on_mean = float(on_row_vals.mean()) if len(on_row_vals) else 0.0
    off_mean = float(off_row_vals.mean()) if len(off_row_vals) else 0.0

    result = {
        "on_off_contrast": on_mean / max(off_mean, eps),
        "on_mean": on_mean,
        "off_mean": off_mean,
    }
    if threshold is not None:
        result["n_on_hits"] = int((on_row_vals > threshold).sum())
        result["n_off_hits"] = int((off_row_vals > threshold).sum())
    return result
