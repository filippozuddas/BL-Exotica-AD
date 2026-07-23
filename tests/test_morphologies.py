"""Morphology injectors: ON-only semantics and in-band containment.

The containment tests are the load-bearing ones. setigen's non-linear paths take
a coefficient evaluated over the cadence's absolute timeline, so an
under-constrained parameterisation sweeps the signal straight out of the window
and every downstream sensitivity number silently becomes "no signal present"
rather than "signal not detected" — a failure that looks exactly like a real
negative result. See the module docstring of src/data/morphologies.py.
"""

import numpy as np
import pytest
import yaml

from src.data.morphologies import MORPHOLOGIES, build_morphology

ON, OFF = (0, 2, 4), (1, 3, 5)


@pytest.fixture(scope="module")
def data_cfg():
    with open("configs/data/gbt_fine.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def background():
    """Flat unit background: any excess is unambiguously injected signal."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=10.0, scale=0.1, size=(6, 16, 1024))


@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_off_observations_untouched(name, data_cfg, background):
    """OFF observations must come back bit-for-bit, as a real signal would vanish.

    Compared in float32 because the injector returns float32: against a float64
    background every element differs by ~5e-7 from the cast alone, which is not
    signal. Asserting in the returned dtype keeps this an exact-equality test
    rather than a tolerance that could hide a genuinely leaking injection.
    """
    inj = build_morphology(name, data_cfg, seed=1)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=50.0)

    expected = background.astype(np.float32)
    for i in OFF:
        np.testing.assert_array_equal(out[i], expected[i])


@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_signal_lands_in_band(name, data_cfg, background):
    """Every ON observation must actually receive power.

    Not just the cadence as a whole: a non-linear track that leaves the band
    partway through would still add energy to the first ON block while being
    absent from the last, which is the exact silent failure this guards.
    """
    inj = build_morphology(name, data_cfg, seed=1)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=50.0)

    # Peak excess, not summed excess: summing 16384 float32 cells carries ~0.02
    # of accumulation error, a noise floor comparable to a faint injection. A
    # single-pixel maximum has no such floor.
    expected = background.astype(np.float32)
    for i in ON:
        peak = float((out[i] - expected[i]).max())
        assert peak > 0, f"{name}: ON observation {i} received no power"


@pytest.mark.parametrize("name", MORPHOLOGIES)
@pytest.mark.parametrize("seed", range(48))
def test_in_band_across_seeds(name, data_cfg, background, seed):
    """Containment must hold for the whole sampled parameter range, not one draw.

    The excursion/amplitude fractions are sampled per site, so a single seed
    exercises one point in that range; the sweep will draw hundreds.
    """
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=50.0)

    expected = background.astype(np.float32)
    for i in ON:
        peak = float((out[i] - expected[i]).max())
        assert peak > 0, f"{name}: ON observation {i} empty at seed {seed}"


@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_morphology_frozen_across_snr(name, data_cfg, background):
    """Amplitude is the only thing an SNR sweep may vary.

    Re-injecting the same Site at two SNRs must move the same pixels: if the
    shape were re-sampled per SNR, the sweep would confound amplitude with
    morphology and its curve would be uninterpretable.
    """
    inj = build_morphology(name, data_cfg, seed=3)
    site = inj.sample_site(fchans=1024, total_tchans=96)

    lo, _ = inj.inject(background, site, snr=10.0)
    hi, _ = inj.inject(background, site, snr=40.0)

    expected = background.astype(np.float32)
    support_lo = (lo[list(ON)] - expected[list(ON)]) > 1e-9
    support_hi = (hi[list(ON)] - expected[list(ON)]) > 1e-9
    # The louder injection may light up marginally more of the profile tail, so
    # require containment rather than equality.
    assert support_lo.sum() > 0
    assert np.all(support_hi[support_lo])


@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_info_is_traceable(name, data_cfg, background):
    """Every row of the results CSV must identify the shape that produced it."""
    inj = build_morphology(name, data_cfg, seed=5)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    _, info = inj.inject(background, site, snr=20.0)

    assert info["morphology"] == name
    assert info["snr"] == 20.0
    assert "path" in info and "start_channel" in info


def test_unknown_morphology_names_alternatives(data_cfg):
    with pytest.raises(ValueError, match="narrowband_drift"):
        build_morphology("does_not_exist", data_cfg)
