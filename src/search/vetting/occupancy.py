"""Occupancy scorer — Stadio B (vetting) plugin for ``0000.fil`` cadence frames.

Every frame-level scoring line on ``0000.fil`` (plain AE, MemAE, ViT-MAE) is
closed or capped (see memory ``current_training_status`` /
``eti_vs_rfi_occupancy_diagnostic``): reconstruction error is at chance on the
one axis that IS separable — cadence occupancy (ETI present in the 3 ON
observations, absent from the 3 OFF observations; RFI is either persistent
across all 6 or randomly intermittent). This module implements that occupancy
statistic directly as an inference-time scorer, no training objective involved
(the no-ON/OFF-training-objective ban in CLAUDE.md is about the model, not
inference-time vetting; see memory ``arch_constraint_no_on_off``, relaxed for
this exact use on 2026-07-01).

Design (frozen by ``docs/2026-07-02_occupancy_scorer_plan.md`` section 2):
for every candidate drifting track ``tau = (start_channel, drift_chans)``
inside a ``(96, 1024)`` cadence frame (6 observations x 16 bins), compute the
boxcar-smoothed per-observation mean along that track, then

    S_on(tau)  = min over ON  observations of the per-obs mean   (strict AND)
    S_off(tau) = max over OFF observations of the per-obs mean   (strict absence)
    C(tau)     = S_on(tau) - S_off(tau)
    score      = max over tau of C(tau)

A persistent RFI line scores ~0 (S_on ~= S_off). An intermittent line that
misses even one ON block collapses S_on (the defect of the earlier "loose"
cadence scoring this replaces). Per-channel bandpass residual cancels to
first order because ON and OFF sample the same channel via the same track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["OccupancyScorer", "TrackInfo", "VettingFilter"]


class VettingFilter(Protocol):
    """Minimal plugin protocol for Stadio B vetting filters.

    ``score_frames`` is the shared surface (0000/0002); Phase 3 additionally
    wants ``apply(candidates) -> candidates`` once a filter is wired into
    ``search/candidates.py`` — out of scope until the pass/fail decision on
    this scorer (plan section 8).
    """

    def score_frames(self, maps: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, "TrackInfo"]:
        ...


@dataclass
class TrackInfo:
    """Best track per frame, plus enough state for the top-negatives diagnostic
    (plan section 3.3): manual (a) coincident-RFI vs (b) min/max-logic-failure
    triage needs the winning ``(start_channel, drift_chans)`` and the six
    per-observation boxcar means that produced the score."""

    score: np.ndarray            # (B,)
    start_channel: np.ndarray    # (B,) int, c0 at t=0
    drift_chans: np.ndarray      # (B,) int, total channel displacement over the frame
    obs_means: np.ndarray        # (B, n_obs) boxcar mean along the winning track, per observation


class OccupancyScorer:
    """Strict ON/OFF track-joint occupancy scorer (plan section 2.1-2.2).

    Product-specific piece: the per-observation statistic along a track is
    narrowband-shaped (boxcar mean of a drifting line). The shared piece
    (min-ON / max-OFF / difference) is meant to be reused verbatim by a future
    0002.fil variant with a different per-obs statistic (plan section 2.4).

    All state that only depends on frame geometry (the drift bank, the
    validity mask) is precomputed once in ``__init__``.
    """

    def __init__(
        self,
        fchans: int = 1024,
        tchans_per_obs: int = 16,
        n_obs: int = 6,
        on_indices: Sequence[int] = (0, 2, 4),
        off_indices: Sequence[int] = (1, 3, 5),
        boxcar_width: int = 3,
        drift_step: int = 2,
        max_drift_chans: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        if set(on_indices) & set(off_indices):
            raise ValueError(f"on_indices/off_indices overlap: {on_indices} / {off_indices}")

        self.fchans = fchans
        self.tchans_per_obs = tchans_per_obs
        self.n_obs = n_obs
        self.total_tchans = tchans_per_obs * n_obs
        self.on_indices = tuple(on_indices)
        self.off_indices = tuple(off_indices)
        self.boxcar_width = boxcar_width
        self.drift_step = drift_step
        # Window-limited max: a track spanning +-fchans channels over the full
        # frame corresponds exactly to the generator's _max_drift_rate (see
        # plan section 2.2 derivation) — the same physical bound the injector
        # itself refuses to exceed.
        self.max_drift_chans = fchans if max_drift_chans is None else max_drift_chans
        self.device = torch.device(device)

        deltas = np.arange(-self.max_drift_chans, self.max_drift_chans + 1, drift_step, dtype=np.int64)
        t = np.arange(self.total_tchans, dtype=np.float64)
        # rel[d, t] = round(delta_d * t / (total_tchans - 1)) -> channel offset
        # from the t=0 start channel, along the absolute cadence timeline.
        rel = np.round(np.outer(deltas.astype(np.float64), t) / (self.total_tchans - 1)).astype(np.int64)

        self.deltas = deltas
        self._rel = torch.as_tensor(rel, device=self.device)  # (n_drift, total_tchans)

        min_rel = rel.min(axis=1)
        max_rel = rel.max(axis=1)
        lo = np.maximum(0, -min_rel)
        hi = np.minimum(fchans - 1, fchans - 1 - max_rel)
        c0_grid = np.arange(fchans, dtype=np.int64)
        # Only tracks entirely inside the window are valid hypotheses — no
        # wrap-around fallback (unit test: edge tracks must be excluded, not
        # silently rolled).
        valid_mask = (c0_grid[None, :] >= lo[:, None]) & (c0_grid[None, :] <= hi[:, None])
        self._lo = lo
        self._hi = hi
        self._c0_grid = torch.as_tensor(c0_grid, device=self.device)
        self._valid_mask = torch.as_tensor(valid_mask, device=self.device)

    def _boxcar(self, maps: torch.Tensor) -> torch.Tensor:
        """Width-``boxcar_width`` running mean along the frequency axis, edge-replicated."""
        w = self.boxcar_width
        if w <= 1:
            return maps
        pad = w // 2
        b, t, fd = maps.shape
        padded = F.pad(maps, (pad, pad), mode="replicate")
        kernel = torch.full((1, 1, w), 1.0 / w, device=maps.device, dtype=maps.dtype)
        flat = padded.reshape(b * t, 1, fd + 2 * pad)
        smoothed = F.conv1d(flat, kernel)
        return smoothed.reshape(b, t, fd)

    def score_frames(self, maps: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, TrackInfo]:
        """``(B, total_tchans, fchans)`` (or unbatched ``(total_tchans, fchans)``)
        preprocessed/residual maps -> ``(scores, TrackInfo)``, higher = more
        occupancy-consistent with an ON-only cadence signal.

        Loops over the ``n_drift`` (~1025) hypotheses in Python, fully
        vectorized over ``(batch, start_channel, time)`` inside each iteration
        — matches the plan's ~1e8 ops/frame budget (n_drift * fchans *
        total_tchans), trivial per-iteration cost on GPU.
        """
        maps_t = torch.as_tensor(np.asarray(maps), dtype=torch.float32, device=self.device)
        if maps_t.dim() == 2:
            maps_t = maps_t.unsqueeze(0)
        b, t, fd = maps_t.shape
        if t != self.total_tchans or fd != self.fchans:
            raise ValueError(
                f"Expected frames of shape ({self.total_tchans}, {self.fchans}), got ({t}, {fd})."
            )
        smoothed = self._boxcar(maps_t)  # (B, T, Fd)

        n_drift = self._rel.shape[0]
        best_score = torch.full((b,), float("-inf"), device=self.device)
        best_c0 = torch.zeros(b, dtype=torch.long, device=self.device)
        best_delta_idx = torch.zeros(b, dtype=torch.long, device=self.device)
        best_obs_means = torch.zeros(b, self.n_obs, device=self.device)

        on_idx = torch.as_tensor(self.on_indices, device=self.device)
        off_idx = torch.as_tensor(self.off_indices, device=self.device)
        batch_arange = torch.arange(b, device=self.device)

        for d in range(n_drift):
            valid_d = self._valid_mask[d]  # (Fd,)
            if not torch.any(valid_d):
                continue
            rel_d = self._rel[d]  # (T,)
            # channel[c0, t] = c0 + rel_d[t], clamped (not wrapped) for a safe
            # gather index; out-of-window (c0) hypotheses are masked below.
            channel = torch.clamp(self._c0_grid.unsqueeze(1) + rel_d.unsqueeze(0), 0, fd - 1)  # (Fd, T)
            index = channel.view(1, fd, t, 1).expand(b, fd, t, 1)
            src = smoothed.unsqueeze(1).expand(b, fd, t, fd)
            gathered = torch.gather(src, dim=3, index=index).squeeze(3)  # (B, Fd, T)

            obs_means = gathered.view(b, fd, self.n_obs, self.tchans_per_obs).mean(dim=3)  # (B, Fd, n_obs)
            s_on = obs_means[:, :, on_idx].min(dim=2).values
            s_off = obs_means[:, :, off_idx].max(dim=2).values
            c = s_on - s_off  # (B, Fd)
            c = c.masked_fill(~valid_d.unsqueeze(0), float("-inf"))

            c_best_this_d, c0_this_d = c.max(dim=1)
            improve = c_best_this_d > best_score
            if not torch.any(improve):
                continue
            best_score = torch.where(improve, c_best_this_d, best_score)
            best_c0 = torch.where(improve, c0_this_d, best_c0)
            best_delta_idx = torch.where(improve, torch.full_like(best_delta_idx, d), best_delta_idx)
            sel_obs_means = obs_means[batch_arange, c0_this_d]  # (B, n_obs)
            best_obs_means = torch.where(improve.unsqueeze(1), sel_obs_means, best_obs_means)

        deltas_t = torch.as_tensor(self.deltas, device=self.device)
        drift_chans = deltas_t[best_delta_idx]

        info = TrackInfo(
            score=best_score.cpu().numpy(),
            start_channel=best_c0.cpu().numpy().astype(np.int64),
            drift_chans=drift_chans.cpu().numpy().astype(np.int64),
            obs_means=best_obs_means.cpu().numpy(),
        )
        return info.score, info
