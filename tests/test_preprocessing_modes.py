"""Normalization-order guard for the injection harnesses.

The load-bearing invariant is that a benchmark's background and its injections
live in the *same* normalization domain as the pipeline the benchmark claims to
characterise. Commit ``4b5660c`` (2026-07-09) switched ``scripts/inference.py``
and ``src/data/torch_dataset.py`` from concatenate-then-normalize to
normalize-per-observation-then-concatenate, but left ``scripts/inject_recover.py``
on the old order. Nothing failed: both harnesses kept running and kept producing
plausible detection rates, in two different domains, for two more weeks.

That is the failure these tests exist to make loud. They pin each mode against a
verbatim copy of the implementation it is supposed to mirror, so a future edit to
either side breaks a test instead of quietly moving a benchmark number.
"""

import numpy as np
import pytest

from scripts.inject_recover import (
    PREPROC_MODES, normalize_frames, preprocess_injected, preprocess_raw_window,
)
from src.data.preprocessing import bandpass_correct, core_transform

PREPROC = {"bandpass_method": "polynomial", "poly_degree": 3, "mad_epsilon": 1e-6}
N_OBS, TCHANS_PER_OBS, FCHANS = 6, 16, 256
TCHANS = N_OBS * TCHANS_PER_OBS


@pytest.fixture
def stepped_cadence():
    """Six observations with a real power step between them.

    A genuine GBT cadence alternates ON and OFF pointings with different system
    temperatures, so the per-observation mean level genuinely differs. That step
    is the only thing the two modes disagree about, which is why every test here
    needs it present rather than a flat background.
    """
    rng = np.random.default_rng(0)
    gains = np.array([1.0, 1.6, 1.0, 1.6, 1.0, 1.6])
    bandpass = 1.0 + 0.3 * np.linspace(-1, 1, FCHANS) ** 2
    obs = rng.normal(loc=10.0, scale=1.0, size=(N_OBS, TCHANS_PER_OBS, FCHANS))
    return (obs * gains[:, None, None] * bandpass[None, None, :]).astype(np.float32)


def _reference_per_obs(frames):
    """Verbatim from scripts/inference.py::_preprocess_at (post-4b5660c)."""
    normed = [
        core_transform(bandpass_correct(f, method="polynomial", poly_degree=3), 1e-6)
        for f in frames
    ]
    return np.concatenate(normed, axis=0)[:TCHANS, :]


def _reference_legacy(frames):
    """Verbatim from scripts/inject_recover.py before 2026-07-23."""
    stacked = np.concatenate(frames, axis=0)[:TCHANS, :]
    stacked = bandpass_correct(stacked, method="polynomial", poly_degree=3)
    return core_transform(stacked, 1e-6)


def test_per_obs_matches_production_inference(stepped_cadence):
    """'per_obs' must be bit-identical to what the production search computes.

    Exact equality, not a tolerance: the two are meant to be the same arithmetic
    in the same order. Any drift at all means a benchmark and the search it
    describes have started measuring different things.
    """
    frames = list(stepped_cadence)
    out = normalize_frames(frames, PREPROC, TCHANS, mode="per_obs")
    np.testing.assert_array_equal(out, _reference_per_obs(frames))


def test_legacy_concat_reproduces_historical_runs(stepped_cadence):
    """'legacy_concat' must still reproduce every pre-2026-07-23 benchmark."""
    frames = list(stepped_cadence)
    out = normalize_frames(frames, PREPROC, TCHANS, mode="legacy_concat")
    np.testing.assert_array_equal(out, _reference_legacy(frames))


def test_modes_disagree_on_a_stepped_cadence(stepped_cadence):
    """The two modes are genuinely different measurements, not a refactor.

    If this ever passes trivially the fixture has lost its power step and the
    rest of the file stops testing anything.
    """
    frames = list(stepped_cadence)
    per_obs = normalize_frames(frames, PREPROC, TCHANS, mode="per_obs")
    legacy = normalize_frames(frames, PREPROC, TCHANS, mode="legacy_concat")
    assert not np.allclose(per_obs, legacy, atol=0.05)


def test_per_obs_removes_the_between_observation_step(stepped_cadence):
    """Each observation is centred on its own noise; legacy leaves the step in.

    This is the mechanism behind the whole divergence: under 'legacy_concat' the
    inter-observation gain step survives into the frame and inflates the block
    MAD, so every window is divided by a number that depends on the ON/OFF
    contrast rather than on its own noise.
    """
    frames = list(stepped_cadence)
    per_obs = normalize_frames(frames, PREPROC, TCHANS, mode="per_obs")
    legacy = normalize_frames(frames, PREPROC, TCHANS, mode="legacy_concat")

    def obs_medians(frame):
        return np.array([np.median(frame[i * TCHANS_PER_OBS:(i + 1) * TCHANS_PER_OBS])
                         for i in range(N_OBS)])

    assert np.ptp(obs_medians(per_obs)) < 0.1
    assert np.ptp(obs_medians(legacy)) > np.ptp(obs_medians(per_obs))


def test_background_and_injection_paths_stay_in_the_same_domain(stepped_cadence):
    """The probe path and the injection path must never drift apart.

    ``preprocess_raw_window`` builds the background a threshold is derived from;
    ``preprocess_injected`` builds the snippets compared against it. They are
    separate entry points with separate call sites, and a detection rate computed
    across a mismatch between them is meaningless while looking entirely healthy.
    """
    obs_arrays = [stepped_cadence[i] for i in range(N_OBS)]
    for mode in PREPROC_MODES:
        via_probe = preprocess_raw_window(obs_arrays, 0, FCHANS, PREPROC,
                                          tchans=TCHANS, mode=mode)
        via_injection = preprocess_injected(stepped_cadence, PREPROC,
                                            tchans=TCHANS, mode=mode)
        # Tolerance, not equality: preprocess_injected casts to float32 for the
        # model, preprocess_raw_window does not.
        np.testing.assert_allclose(via_probe, via_injection, rtol=1e-5, atol=1e-5)


def test_unknown_mode_raises():
    """Fail loudly rather than silently falling through to one of the two."""
    with pytest.raises(ValueError, match="preproc_mode"):
        normalize_frames([np.ones((TCHANS_PER_OBS, FCHANS))], PREPROC, TCHANS,
                         mode="per_observation")
