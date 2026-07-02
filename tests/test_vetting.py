"""Occupancy scorer contract tests (plan section 5) — CPU only, hand-built maps.

No data/training dependency: synthetic (96, 1024) maps (white noise + an
additive drifting line), constructed directly in numpy so these run locally.
"""

import numpy as np
import pytest
import torch

from src.search.vetting import OccupancyScorer

FCHANS = 1024
TCHANS_PER_OBS = 16
N_OBS = 6
TOTAL_TCHANS = TCHANS_PER_OBS * N_OBS  # 96
ON_INDICES = (0, 2, 4)
OFF_INDICES = (1, 3, 5)


def make_track_map(c0, delta, present_obs, amp=15.0, noise_std=1.0, seed=0):
    """White-noise (total_tchans, fchans) map with an additive line along the
    track (c0, delta), rendered only into observations in ``present_obs``."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, noise_std, size=(TOTAL_TCHANS, FCHANS))
    t = np.arange(TOTAL_TCHANS)
    channel = np.clip(np.round(c0 + delta * t / (TOTAL_TCHANS - 1)).astype(int), 0, FCHANS - 1)
    obs_of_t = t // TCHANS_PER_OBS
    present = np.isin(obs_of_t, list(present_obs))
    m[np.arange(TOTAL_TCHANS)[present], channel[present]] += amp
    return m, channel


@pytest.fixture
def scorer():
    return OccupancyScorer(fchans=FCHANS, tchans_per_obs=TCHANS_PER_OBS, n_obs=N_OBS,
                            on_indices=ON_INDICES, off_indices=OFF_INDICES,
                            boxcar_width=3, drift_step=2, device="cpu")


def test_on_only_line_scores_high_and_recovers_track(scorer):
    c0, delta = 400, 300
    m, _ = make_track_map(c0, delta, present_obs=ON_INDICES, seed=1)
    scores, info = scorer.score_frames(m)
    assert scores[0] > 3.0
    assert abs(int(info.start_channel[0]) - c0) <= 1
    # drift_step=2, so the recovered delta can be off by up to one step.
    assert abs(int(info.drift_chans[0]) - delta) <= scorer.drift_step


def test_persistent_line_scores_near_zero(scorer):
    c0, delta = 400, 300
    m, _ = make_track_map(c0, delta, present_obs=range(N_OBS), seed=2)
    scores, _ = scorer.score_frames(m)
    assert abs(scores[0]) < 1.0


@pytest.mark.parametrize("missing", [(0,), (2,), (4,)])
def test_intermittent_on_line_collapses_score(scorer, missing):
    # Missing even one of the 3 ON blocks drags S_on down to the noise floor
    # (min-ON), regardless of how bright the line is in the other two blocks —
    # the defect of "loose" cadence scoring that strict min/max fixes.
    c0, delta = 400, 300
    present = set(ON_INDICES) - set(missing)
    m, _ = make_track_map(c0, delta, present_obs=present, seed=3)
    scores, _ = scorer.score_frames(m)
    assert abs(scores[0]) < 1.5


def test_line_present_in_one_off_block_suppresses_score(scorer):
    # A line leaking into a single OFF block pulls S_off up to ~S_on -> C ~ 0.
    c0, delta = 400, 300
    m, _ = make_track_map(c0, delta, present_obs=set(ON_INDICES) | {1}, seed=4)
    scores, _ = scorer.score_frames(m)
    assert abs(scores[0]) < 1.5


def test_pure_noise_null_distribution_centred_near_zero(scorer):
    rng = np.random.default_rng(5)
    maps = rng.normal(0.0, 1.0, size=(8, TOTAL_TCHANS, FCHANS))
    scores, _ = scorer.score_frames(maps)
    assert abs(scores.mean()) < 1.0


def test_edge_tracks_excluded_no_wraparound(scorer):
    # A track spanning the full window (delta == max_drift_chans) cannot fit
    # any start_channel inside [0, fchans-1] -> no valid hypothesis at all.
    d_idx = int(np.where(scorer.deltas == scorer.max_drift_chans)[0][0])
    assert scorer._lo[d_idx] > scorer._hi[d_idx]
    # A near-edge but still valid track must be recovered without the scorer
    # ever preferring an off-window (wrapped) hypothesis.
    c0, delta = 5, 900
    m, channel = make_track_map(c0, delta, present_obs=ON_INDICES, seed=6)
    assert channel.max() < FCHANS  # sanity: track itself stays fully in-window
    scores, info = scorer.score_frames(m)
    assert 0 <= int(info.start_channel[0]) < FCHANS
    assert scores[0] > 3.0


def test_batched_matches_frame_by_frame_loop(scorer):
    maps = np.stack([
        make_track_map(300, 200, ON_INDICES, seed=10)[0],
        make_track_map(600, -400, ON_INDICES, seed=11)[0],
        np.random.default_rng(12).normal(0.0, 1.0, size=(TOTAL_TCHANS, FCHANS)),
    ])
    batched_scores, batched_info = scorer.score_frames(maps)
    for i in range(maps.shape[0]):
        single_scores, single_info = scorer.score_frames(maps[i])
        assert batched_scores[i] == pytest.approx(single_scores[0], abs=1e-4)
        assert batched_info.start_channel[i] == single_info.start_channel[0]
        assert batched_info.drift_chans[i] == single_info.drift_chans[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available")
def test_cpu_matches_gpu():
    cpu_scorer = OccupancyScorer(device="cpu")
    gpu_scorer = OccupancyScorer(device="cuda")
    m, _ = make_track_map(400, 300, ON_INDICES, seed=20)
    cpu_scores, cpu_info = cpu_scorer.score_frames(m)
    gpu_scores, gpu_info = gpu_scorer.score_frames(m)
    assert cpu_scores[0] == pytest.approx(gpu_scores[0], abs=1e-3)
    assert cpu_info.start_channel[0] == gpu_info.start_channel[0]
    assert cpu_info.drift_chans[0] == gpu_info.drift_chans[0]


def test_drift_sign_convention_self_consistent(scorer):
    """The scorer's own track model (channel(t) = c0 + round(delta*t/(T-1)),
    same convention used by ``make_track_map``) recovers the injected sign —
    i.e. the drift bank is internally self-consistent, both signs searched."""
    m_pos, _ = make_track_map(100, 600, ON_INDICES, seed=30)
    _, info_pos = scorer.score_frames(m_pos)
    assert info_pos.drift_chans[0] > 0

    m_neg, _ = make_track_map(900, -600, ON_INDICES, seed=31)
    _, info_neg = scorer.score_frames(m_neg)
    assert info_neg.drift_chans[0] < 0


def test_drift_sign_convention_matches_setigen():
    """End-to-end: an ETI signal actually rendered by setigen's
    NarrowbandDriftingGenerator (drift > 0 -> increasing channel index, per
    ``_sample_start_channel``) is recovered by the scorer with the same sign."""
    pytest.importorskip("setigen")
    from src.data.synthetic import NarrowbandParams, NarrowbandDriftingGenerator

    params = NarrowbandParams(fchans=FCHANS, tchans=TOTAL_TCHANS)
    gen = NarrowbandDriftingGenerator(params, seed=42)
    scorer = OccupancyScorer(fchans=FCHANS, tchans_per_obs=TCHANS_PER_OBS, n_obs=N_OBS,
                              on_indices=ON_INDICES, off_indices=OFF_INDICES,
                              boxcar_width=3, drift_step=2, device="cpu")

    for drift_rate, expect_sign in [(0.9, 1), (-0.9, -1)]:
        background = gen.synthetic_background().reshape(N_OBS, TCHANS_PER_OBS, FCHANS)
        out, info = gen.inject_signal_cadence(
            background, on_indices=ON_INDICES, snr=40.0,
            start_channel=FCHANS // 2, drift_rate=drift_rate,
        )
        frame = np.concatenate(out, axis=0)  # (total_tchans, fchans), same as preprocess_raw
        scores, track = scorer.score_frames(frame)
        assert scores[0] > 0
        assert int(np.sign(track.drift_chans[0])) == expect_sign
