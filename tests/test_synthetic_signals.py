"""Tests for the setigen-native signal generators (src/data/synthetic.py).

These cover the narrowband-drifting and wideband-pulsed generators: output
contract, reproducibility, SNR monotonicity, the (empirically pinned) setigen
drift-direction convention, spectral narrowness/width, and pulse periodicity.

The whole module requires setigen — hence the module-level ``importorskip``,
which skips the entire file when setigen is absent. The pure-numpy broadband
tests deliberately live in ``test_data.py`` so they keep running without setigen.
"""

import numpy as np
import pytest

pytest.importorskip("setigen")

from src.data.synthetic import (  # noqa: E402
    NarrowbandParams,
    NarrowbandDriftingGenerator,
    WidebandParams,
    WidebandPulsedGenerator,
)


def noise_background(p, seed=0):
    """chi^2-like positive background, mean ~ p.noise_mean, shape (tchans, fchans)."""
    rng = np.random.default_rng(seed)
    return rng.chisquare(df=2, size=(p.tchans, p.fchans)) * (p.noise_mean / 2.0)


def nb_params(**over):
    """Deterministic narrowband params: single fixed freq profile, no
    scintillation, fixed width offset — so power/direction comparisons aren't
    confounded by random profile draws."""
    p = NarrowbandParams(
        df=2.7939677238464355, dt=18.25361108, fch1=0.0, fchans=256, tchans=32,
        freq_profiles=("gaussian",), freq_profile_weights=(1.0,),
        time_profiles=("constant",), time_profile_weights=(1.0,),
        eti_width_offset_min=5.0, eti_width_offset_max=5.0,
    )
    for k, v in over.items():
        setattr(p, k, v)
    return p


def wb_params(**over):
    """Deterministic wideband params: single fixed (gaussian) freq profile."""
    p = WidebandParams(
        df=2861.02294921875, dt=1.0737418239999998, fch1=0.0, fchans=256, tchans=64,
        freq_profiles=("gaussian",), freq_profile_weights=(1.0,),
    )
    for k, v in over.items():
        setattr(p, k, v)
    return p


# ----------------------------- narrowband ---------------------------------- #
def test_narrowband_output_contract():
    p = nb_params()
    gen = NarrowbandDriftingGenerator(p, seed=5)
    bg = noise_background(p, seed=8)
    out, info = gen.inject_signal(bg)
    assert out.shape == bg.shape
    assert np.isfinite(out).all()
    for key in ("snr", "drift_rate", "start_channel", "width", "slope", "f_profile", "t_profile"):
        assert key in info


def test_narrowband_same_seed_same_output():
    p = nb_params()
    bg = noise_background(p, seed=11)
    a, ia = NarrowbandDriftingGenerator(p, seed=99).inject_signal(bg.copy())
    b, ib = NarrowbandDriftingGenerator(p, seed=99).inject_signal(bg.copy())
    np.testing.assert_array_equal(a, b)
    assert ia == ib


def test_narrowband_power_increases_with_snr():
    # Fix drift/start/seed; vary only SNR -> injected power must scale up.
    p = nb_params()
    bg = noise_background(p, seed=3)

    def injected_power(snr):
        gen = NarrowbandDriftingGenerator(p, seed=7)
        out, _ = gen.inject_signal(bg.copy(), snr=snr, start_channel=128, drift_rate=0.05)
        return float((out - bg).sum())

    assert injected_power(40.0) > injected_power(8.0) > 0.0


def test_narrowband_drift_direction_matches_setigen_convention():
    # Isolate the pure injected signal by subtraction; argmax tracks the tone.
    p = nb_params()
    bg = noise_background(p, seed=4)
    gen = NarrowbandDriftingGenerator(p, seed=4)

    out_pos, _ = gen.inject_signal(bg, snr=50.0, start_channel=128, drift_rate=0.3)
    am_pos = (out_pos - bg).argmax(axis=1)
    assert am_pos[-1] > am_pos[0]  # drift>0 -> peak moves to HIGHER channel index

    out_neg, _ = gen.inject_signal(bg, snr=50.0, start_channel=128, drift_rate=-0.3)
    am_neg = (out_neg - bg).argmax(axis=1)
    assert am_neg[-1] < am_neg[0]  # drift<0 -> LOWER channel index


def test_narrowband_is_spectrally_narrow():
    p = nb_params()
    bg = noise_background(p, seed=4)
    gen = NarrowbandDriftingGenerator(p, seed=4)
    out, _ = gen.inject_signal(bg, snr=50.0, start_channel=128, drift_rate=0.0)
    row = (out - bg)[0]  # first time bin
    lit = int((row > 0.1 * row.max()).sum())
    assert lit < 0.1 * p.fchans  # narrowband: occupies << fchans channels


def test_narrowband_low_snr_is_near_background():
    p = nb_params()
    bg = noise_background(p, seed=4)
    gen = NarrowbandDriftingGenerator(p, seed=4)
    out, _ = gen.inject_signal(bg, snr=0.01, start_channel=128, drift_rate=0.0)
    assert float((out - bg).sum()) < 0.05 * float(bg.sum())


# ------------------------------ wideband ----------------------------------- #
def test_wideband_output_contract():
    p = wb_params()
    gen = WidebandPulsedGenerator(p, seed=5)
    bg = noise_background(p, seed=8)
    out, info = gen.inject_signal(bg)
    assert out.shape == bg.shape
    assert np.isfinite(out).all()
    for key in ("snr", "period_bins", "pulse_width_bins", "width", "drift_rate",
                "start_channel", "f_profile"):
        assert key in info


def test_wideband_same_seed_same_output():
    p = wb_params()
    bg = noise_background(p, seed=11)
    a, ia = WidebandPulsedGenerator(p, seed=77).inject_signal(bg.copy())
    b, ib = WidebandPulsedGenerator(p, seed=77).inject_signal(bg.copy())
    np.testing.assert_array_equal(a, b)
    assert ia == ib


def test_wideband_power_increases_with_snr():
    p = wb_params()
    bg = noise_background(p, seed=3)

    def injected_power(snr):
        gen = WidebandPulsedGenerator(p, seed=7)
        out, _ = gen.inject_signal(bg.copy(), snr=snr)
        return float((out - bg).sum())

    assert injected_power(40.0) > injected_power(8.0) > 0.0


def test_wideband_is_spectrally_wide_and_periodic():
    # Force a wide extent and a short period so >=2 pulses fit the 64-bin frame.
    p = wb_params(frac_low=0.4, frac_high=0.4,
                  period_bins_min=12.0, period_bins_max_frac=0.25, duty_max=0.25)
    bg = noise_background(p, seed=4)
    gen = WidebandPulsedGenerator(p, seed=4)
    out, _ = gen.inject_signal(bg, snr=80.0)
    sig = out - bg

    # Spectrally wide: at the brightest time bin many channels are lit.
    row = sig[sig.sum(axis=1).argmax()]
    lit = int((row > 0.1 * row.max()).sum())
    assert lit > 0.25 * p.fchans

    # Periodic in time: the frequency-collapsed series has >=2 distinct pulses
    # (count contiguous runs above half-max).
    ts = sig.sum(axis=1)
    on = ts > 0.5 * ts.max()
    runs = int(np.sum(np.diff(on.astype(int)) == 1)) + (1 if on[0] else 0)
    assert runs >= 2


def test_wideband_is_wider_than_narrowband():
    # Same band; the wideband signal must light up far more channels than a
    # narrowband tone at a fixed time bin.
    nbp = nb_params(fchans=256)
    nb_bg = noise_background(nbp, seed=4)
    nb_out, _ = NarrowbandDriftingGenerator(nbp, seed=4).inject_signal(
        nb_bg, snr=50.0, start_channel=128, drift_rate=0.0)
    nb_row = (nb_out - nb_bg)[0]
    nb_lit = int((nb_row > 0.1 * nb_row.max()).sum())

    wbp = wb_params(fchans=256, frac_low=0.4, frac_high=0.4)
    wb_bg = noise_background(wbp, seed=4)
    wb_out, _ = WidebandPulsedGenerator(wbp, seed=4).inject_signal(wb_bg, snr=80.0)
    wb_sig = wb_out - wb_bg
    wb_row = wb_sig[wb_sig.sum(axis=1).argmax()]
    wb_lit = int((wb_row > 0.1 * wb_row.max()).sum())

    assert wb_lit > 10 * nb_lit
