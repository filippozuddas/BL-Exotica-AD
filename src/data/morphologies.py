"""Signal morphologies behind one interface, for the sensitivity sweeps.

Every injection benchmark in this project has so far injected exactly one thing:
a linearly-drifting narrowband carrier (``NarrowbandDriftingGenerator``). That
makes the central thesis — that an unsupervised autoencoder is sensitive to
*arbitrary* signal morphologies, unlike narrowband-only turboSETI — untested by
construction: the search has only ever been shown signals of the one class the
traditional pipeline already finds.

This module exposes a family of morphologies through a single interface so that
``scripts/inject_recover.py`` (scorer-level) and ``scripts/pipeline_sensitivity.py``
(end-to-end through the ON/OFF rule stage) can sweep all of them by name,
without either script knowing how any one of them is rendered.

    injector = build_morphology("narrowband_accel", data_cfg, seed=123)
    site = injector.sample_site(fchans, total_tchans)   # morphology, frozen
    for snr in snr_list:                                # amplitude only
        windows, info = injector.inject(obs_windows, site, snr)

The ``sample_site`` / ``inject`` split is load-bearing and matches the existing
convention in ``NarrowbandDriftingGenerator.sample_cadence_signal_params``: the
morphology is drawn ONCE per injection site and reused across the whole SNR
sweep, so SNR is the sweep's single independent variable. Re-sampling per SNR
would confound amplitude with shape.

Two families implement the interface:

* :class:`SetigenMorphology` — signals expressible as setigen's
  ``(path, t_profile, f_profile)`` decomposition, rendered ON-only through
  ``_SetigenInjector.inject_on_only_cadence``. Covers the drifting, accelerating,
  sinusoidal and pulsed classes.
* :class:`DirectArrayMorphology` — signals that decomposition cannot express,
  because their frequency extent varies with time (a dispersed sweep is not a
  ``path`` times an ``f_profile``). Renders a 2D template directly onto each ON
  observation, following ``BroadbandTransientGenerator``'s approach.

**SNR is not commensurable across morphologies.** Each family normalises
amplitude by its own convention (setigen's frame-integrated
``Frame.get_intensity`` for the first, per-transient energy for the second), and
a pulsed signal's "snr" is a per-pulse peak rather than an integrated level.
Compare the *shape* of survival curves across morphologies — where they fall
off, which pipeline stage kills them — never absolute SNR between two of them.

**Band-excursion discipline.** setigen's non-linear paths take a coefficient,
not an excursion, and are evaluated on the cadence's ABSOLUTE timeline (~1728 s
for the 0000 product), not one observation. Feeding them a narrowband drift rate
(~0.3 Hz/s) would sweep ~448 kHz across a ~2.9 kHz window: the signal leaves the
frame within a few bins and the sweep silently measures "no signal present"
rather than "signal not detected". So the non-linear morphologies here sample a
**total excursion as a fraction of the band** and solve for the coefficient,
and place ``start_channel`` so the whole track stays in-band.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import setigen as stg
    from astropy import units as u
except ImportError:  # pragma: no cover - mirrors synthetic.py's optional import
    stg = None
    u = None

from src.data.synthetic import (
    NarrowbandParams,
    NarrowbandDriftingGenerator,
    WidebandParams,
    WidebandPulsedGenerator,
)

ON_INDICES = (0, 2, 4)


@dataclass
class Site:
    """One frozen injection site: the morphology, minus its amplitude.

    ``payload`` carries whatever the owning morphology needs to render (setigen
    profiles and a path builder, or a 2D template factory). ``meta`` is the
    human/CSV-facing description and is written verbatim into the results table,
    so every row can be traced back to the shape that produced it.
    """

    payload: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class SetigenMorphology:
    """Morphologies expressible as setigen ``(path, t_profile, f_profile)``.

    Delegates the ON-only cadence assembly — absolute-timeline ``ts`` shifting,
    one constant physical amplitude across ON frames, byte-identical OFF frames —
    to ``_SetigenInjector.inject_on_only_cadence``, so all morphologies in this
    family share exactly the same injection semantics as the validated
    narrowband path and differ only in the three profile functions.
    """

    def __init__(self, name: str, generator, sampler: Callable[..., Site]):
        self.name = name
        self._gen = generator
        self._sampler = sampler

    @property
    def rng(self) -> np.random.Generator:
        return self._gen.rng

    def sample_site(self, fchans: int, total_tchans: int) -> Site:
        return self._sampler(self._gen, fchans, total_tchans)

    def inject(self, obs_windows: np.ndarray, site: Site, snr: float,
               on_indices: Tuple[int, ...] = ON_INDICES) -> Tuple[np.ndarray, dict]:
        out, info = self._gen.inject_on_only_cadence(
            obs_windows,
            snr=snr,
            drift_rate=site.payload["drift_rate"],
            start_channel=site.payload["start_channel"],
            f_profile=site.payload["f_profile"],
            t_profile_builder=site.payload["t_profile_builder"],
            on_indices=on_indices,
            path_builder=site.payload.get("path_builder"),
        )
        info.update(site.meta)
        info["morphology"] = self.name
        return out, info


# --------------------------------------------------------------------------
# Samplers. One per morphology; each returns a Site with everything frozen
# except amplitude.
# --------------------------------------------------------------------------

def _sample_drifting(gen, fchans: int, total_tchans: int) -> Site:
    """Trick 1 — linear narrowband drift. The historical baseline.

    Delegates verbatim to the existing, validated sampler so this morphology is
    bit-identical to every previous benchmark: the sweeps remain comparable to
    the 79.11/95.56/100.0 numbers rather than merely similar.
    """
    drift_rate, start_channel, f_profile, t_builder, meta = \
        gen.sample_cadence_signal_params(fchans, total_tchans)
    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_builder,
                 "path_builder": None},
        meta={**meta, "path": "constant"},
    )


def _band_and_duration(gen, fchans: int, total_tchans: int) -> Tuple[float, float]:
    """Usable bandwidth (Hz) and absolute cadence duration (s).

    ``total_tchans`` is the whole cadence's time extent, because that is the
    domain setigen paths are evaluated over once ``inject_on_only_cadence``
    shifts each ON frame onto the absolute timeline.
    """
    p = gen.params
    return float(fchans) * p.df, float(total_tchans) * p.dt


def _sample_accelerating(gen, fchans: int, total_tchans: int) -> Site:
    """Trick 2a — quadratically accelerating drift (``squared_path``).

    Physically: a transmitter whose line-of-sight acceleration is non-negligible
    over the cadence, e.g. a close-in planetary rotator. This is the class a
    linear-drift matched filter such as turboSETI degrades on, so it is one of
    the cleanest places to look for a sensitivity gap in either direction.

    Parameterised by total excursion rather than by the coefficient
    ``squared_path`` actually takes — see the module docstring. Given an
    excursion ``df`` over cadence duration ``T``, ``0.5*a*T^2 = df`` gives
    ``a = 2*df/T^2``. The track is monotonic, so placing ``f_start`` at the
    low (or high) edge with a margin keeps the whole sweep in-band.
    """
    band_hz, duration_s = _band_and_duration(gen, fchans, total_tchans)
    frac = float(gen.rng.uniform(0.05, 0.40))
    sign = 1.0 if gen.rng.random() < 0.5 else -1.0
    excursion_hz = frac * band_hz
    accel = 2.0 * excursion_hz / (duration_s ** 2)

    margin = 0.05 * band_hz
    span_chans = excursion_hz / gen.params.df
    lo = margin / gen.params.df
    hi = fchans - lo - span_chans
    if hi <= lo:  # excursion too wide for the band; fall back to centred
        start_channel = int(fchans // 2)
    elif sign > 0:
        start_channel = int(gen.rng.uniform(lo, hi))
    else:
        start_channel = int(gen.rng.uniform(lo + span_chans, fchans - lo))

    # Width follows the MEAN drift over the cadence, not the coefficient: the
    # signal's instantaneous linewidth is set by how fast it sweeps, and
    # `accel` (Hz/s^2) is not a rate.
    mean_drift = sign * excursion_hz / duration_s
    width = gen._calculate_eti_width(abs(mean_drift))
    f_profile, f_name = gen._select_f_profile(width)

    def path_builder(f_start_hz, _drift_rate):
        return stg.squared_path(f_start=f_start_hz,
                                drift_rate=sign * accel * u.Hz / u.s)

    def t_profile_builder(intensity: float, n_bins: int):
        return stg.constant_t_profile(level=intensity)

    return Site(
        payload={"drift_rate": mean_drift, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_profile_builder,
                 "path_builder": path_builder},
        meta={"path": "squared", "accel_hz_s2": sign * accel,
              "excursion_frac": frac, "mean_drift": mean_drift,
              "width": width, "f_profile": f_name, "t_profile": "constant",
              "start_channel": int(start_channel)},
    )


def _sample_sinusoidal(gen, fchans: int, total_tchans: int) -> Site:
    """Trick 2b — sinusoidally modulated drift (``sine_path``).

    Physically: the Doppler signature of a transmitter on a rotating or orbiting
    body, which is periodic rather than monotonic. Unlike the accelerating case
    this track *returns* to earlier frequencies, so a frequency-adjacency
    clustering stage sees it as several disconnected candidates — which makes it
    a direct test of the clustering stage, not only of the scorer.

    Amplitude is a fraction of the band and the period a fraction of the cadence
    (2-5 cycles, so the periodicity is actually visible within the window); the
    centre is placed so ``+/- amplitude`` plus the linear term stays in-band.
    """
    band_hz, duration_s = _band_and_duration(gen, fchans, total_tchans)
    amp_frac = float(gen.rng.uniform(0.05, 0.20))
    amplitude_hz = amp_frac * band_hz
    n_cycles = float(gen.rng.uniform(2.0, 5.0))
    period_s = duration_s / n_cycles

    max_lin = 0.10 * band_hz / duration_s
    drift_rate = float(gen.rng.uniform(-max_lin, max_lin))
    lin_excursion = abs(drift_rate) * duration_s

    half_span_chans = (amplitude_hz + lin_excursion) / gen.params.df
    lo, hi = half_span_chans, fchans - half_span_chans
    start_channel = int(gen.rng.uniform(lo, hi)) if hi > lo else int(fchans // 2)

    # Peak instantaneous drift of the sinusoid, which is what sets linewidth.
    peak_drift = 2.0 * np.pi * amplitude_hz / period_s + abs(drift_rate)
    width = gen._calculate_eti_width(peak_drift)
    f_profile, f_name = gen._select_f_profile(width)

    def path_builder(f_start_hz, dr):
        return stg.sine_path(f_start=f_start_hz, drift_rate=dr * u.Hz / u.s,
                             period=period_s * u.s, amplitude=amplitude_hz * u.Hz)

    def t_profile_builder(intensity: float, n_bins: int):
        return stg.constant_t_profile(level=intensity)

    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_profile_builder,
                 "path_builder": path_builder},
        meta={"path": "sine", "amplitude_hz": amplitude_hz,
              "amplitude_frac": amp_frac, "period_s": period_s,
              "n_cycles": n_cycles, "linear_drift": drift_rate,
              "peak_drift": peak_drift, "width": width,
              "f_profile": f_name, "t_profile": "periodic-none",
              "start_channel": int(start_channel)},
    )


def _sample_pulsed(gen, fchans: int, total_tchans: int) -> Site:
    """Trick 3 — wide-band periodic pulse train (radar/beacon-like).

    Reuses ``WidebandPulsedGenerator``'s frequency-extent and pulse sampling,
    but renders through the shared ON-only cadence path rather than the
    generator's own single-frame ``inject_signal``, so the OFF observations stay
    untouched exactly as for every other morphology here.

    The pulse period is sampled against the FULL cadence time extent, not one
    observation: a train whose period is a large fraction of a single 16-bin
    observation would show one pulse per ON block and be indistinguishable from
    an intermittent constant signal.

    ``snr`` is a per-pulse peak level here, not a frame-integrated SNR — the
    train is off for most of the frame, so integrated SNR is roughly
    ``snr * duty``. See the module docstring on cross-morphology comparison.
    """
    p = gen.params
    frac = float(gen.rng.uniform(p.frac_low, p.frac_high))
    width_hz = max(1.0, frac * fchans) * p.df
    f_profile, f_name = gen._select_f_profile(width_hz)

    period_max = max(p.period_bins_min + 1.0, p.period_bins_max_frac * total_tchans)
    period_bins = float(gen.rng.uniform(p.period_bins_min, period_max))
    pulse_width_bins = float(gen.rng.uniform(1.0, max(1.0, p.duty_max * period_bins)))
    period_s, pulse_width_s = period_bins * p.dt, pulse_width_bins * p.dt

    drift_rate = float(gen.rng.uniform(-p.drift_jitter, p.drift_jitter))
    start_channel = int(gen.rng.integers(1, max(2, fchans - 1)))
    pulse_seed = int(gen.rng.integers(0, 2 ** 31))

    def t_profile_builder(intensity: float, n_bins: int):
        return stg.periodic_gaussian_t_profile(
            pulse_width=pulse_width_s * u.s,
            period=period_s * u.s,
            pulse_direction="up",
            amplitude=intensity,
            level=0.0,
            min_level=0.0,
            seed=pulse_seed,
        )

    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_profile_builder,
                 "path_builder": None},
        meta={"path": "constant", "width": width_hz, "width_frac": frac,
              "f_profile": f_name, "t_profile": "periodic_gaussian",
              "period_s": period_s, "pulse_width_s": pulse_width_s,
              "duty": pulse_width_bins / period_bins,
              "start_channel": int(start_channel), "snr_convention": "pulse_peak"},
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_SETIGEN_MORPHOLOGIES = {
    "narrowband_drift": (NarrowbandParams, NarrowbandDriftingGenerator, _sample_drifting),
    "narrowband_accel": (NarrowbandParams, NarrowbandDriftingGenerator, _sample_accelerating),
    "narrowband_sine": (NarrowbandParams, NarrowbandDriftingGenerator, _sample_sinusoidal),
    "wideband_pulsed": (WidebandParams, WidebandPulsedGenerator, _sample_pulsed),
}

MORPHOLOGIES = tuple(_SETIGEN_MORPHOLOGIES)


def build_morphology(name: str, data_cfg: dict, seed: Optional[int] = None):
    """Construct a morphology injector by name.

    ``data_cfg`` is the parsed product config (e.g. ``configs/data/gbt_fine.yaml``);
    each morphology reads its geometry and sampling ranges from the block its
    params dataclass owns, so frame geometry stays a single source of truth.
    """
    if name not in _SETIGEN_MORPHOLOGIES:
        raise ValueError(
            f"Unknown morphology '{name}'. Available: {', '.join(MORPHOLOGIES)}."
        )
    params_cls, gen_cls, sampler = _SETIGEN_MORPHOLOGIES[name]
    generator = gen_cls(params_cls.from_config(data_cfg), seed=seed)
    return SetigenMorphology(name, generator, sampler)
