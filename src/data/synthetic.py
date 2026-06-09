"""
Synthetic signal generators for the BL-Exotica-AD injection-recovery tests.

This module hosts one generator per GBT data product / signal class:

- ``NarrowbandDriftingGenerator`` (``*.0000.fil`` fine-frequency) — drifting
  narrowband tones (setigen-native: gaussian/sinc2/lorentzian/voigt frequency
  profiles, log-normal drift, AR(1) scintillation). Ported from the RST
  ``SignalGenerator`` ETI path, minus all cadence/RFI/classifier machinery.
- ``WidebandPulsedGenerator`` (``*.0002.fil`` intermediate) — wide-band periodic
  pulse trains (setigen ``periodic_gaussian_t_profile``). Default design pending
  confirmation with Vishal.
- ``BroadbandTransientGenerator`` (``*.0001.fil`` high-time) — the dispersed,
  FRB-like pulse documented below, built directly in numpy.

The two setigen-native generators build on the ``_SetigenInjector`` base, but
"SNR" means something different in each of the three classes, so it is **not**
directly comparable across them:

- narrowband: setigen's frame-integrated SNR (``Frame.get_intensity``), valid
  because the constant/scintillating tone fills every time bin;
- wideband: the same intensity is applied as the *per-pulse peak*. The pulse
  train is "on" only a fraction (duty) of the frame, so its frame-integrated SNR
  is ~snr*duty (lower) — "snr" here is a per-pulse peak level, not a
  frame-integrated SNR (definition to confirm with Vishal);
- broadband: a transient SNR (dedispersed, frequency-averaged peak).

----

Broadband transient (FRB-like) synthetic signal generator.

This is the one signal class setigen does not cover natively: a dispersed
broadband pulse in the raw (un-dedispersed) waterfall — the ``*.0001.fil``
high-time-resolution GBT product (Gajjar et al. 2022). setigen's separable
``path / t_profile / f_profile`` model cannot express a dispersion sweep together
with a frequency-dependent scattering tail and spectral scintillation, and its
``Frame.get_intensity`` SNR is calibrated for a signal present in *all* time bins
— wrong for a transient that occupies ~1 time bin per channel.

We therefore build the pulse directly in numpy (porting the per-channel physics
of Connor & van Leeuwen 2018 / hey-aliens ``SimulatedFRB``, kept dispersed
instead of dedispersed) and scale it to the correct transient SNR: the
**dedispersed, frequency-averaged peak SNR**. setigen is used only as the
Frame / noise / I-O container (see :meth:`synthetic_background`).

A single ``DM`` parameter unifies both regimes:

- ``DM == 0`` → vertical stripe (undispersed broadband pulse),
- ``DM  > 0`` → diagonal sweep (pulse centered at the per-channel arrival time).

The interface mirrors the reference ``SignalGenerator`` in ML-SRT-SETI: a
dataclass of parameters, a seeded ``rng``, and ``inject_signal(data) -> (data, info)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

try:  # setigen/astropy are optional: only the setigen-native generators need them.
    import setigen as stg
    from astropy import units as u
except ImportError:  # pragma: no cover - keeps the module (and broadband) importable
    stg = None
    u = None

__all__ = [
    "BroadbandParams",
    "BroadbandTransientGenerator",
    "K_DM",
    "NarrowbandParams",
    "NarrowbandDriftingGenerator",
    "WidebandParams",
    "WidebandPulsedGenerator",
]

# Dispersion constant: delay (s) = K_DM * DM * (nu^-2 - nu_ref^-2), nu in MHz.
# Equivalently 4.148808 ms per unit DM at 1 GHz.
K_DM = 4.148808e3  # MHz^2 * pc^-1 * cm^3 * s


@dataclass
class BroadbandParams:
    """Frame geometry + broadband-transient sampling ranges.

    Geometry defaults match ``configs/data/gbt_high_time.yaml``.
    """

    # Frame geometry
    df: float = 364583.0       # Hz per channel
    dt: float = 0.000349       # s per time bin (~349 us)
    fch1: float = 1500.0       # MHz, absolute frequency of channel 0
    fchans: int = 512
    tchans: int = 2048      # DM=1000 sweep ≈ 1105 bins; 2048 keeps pulse in-frame with margin
    ascending: bool = True     # channel index increasing => frequency increasing

    # Dispersion measure (pc/cm^3)
    dm_min: float = 100.0
    dm_max: float = 1000.0

    # Intrinsic Gaussian pulse width (in time bins)
    tau_int_min: float = 1.0
    tau_int_max: float = 4.0

    # Scattering timescale at the reference frequency (ms); scales as (nu/f_ref)^-4
    tau_scatter_ms: float = 0.1

    # Spectral scintillation: nscint log-uniform in (1e-3, nscint_max)
    nscint_max: float = 3.0
    scintillate: bool = True

    # Fractional bandwidth occupied by the pulse
    frac_low: float = 0.5
    frac_high: float = 0.9

    # Peak SNR (dedispersed, frequency-averaged), log-normal in [snr_min, snr_max]
    snr_min: float = 6.0
    snr_max: float = 20.0
    snr_sigma: float = 1.0

    # Synthetic chi^2 background mean (only used by synthetic_background)
    noise_mean: float = 5.0

    @classmethod
    def from_config(cls, cfg: dict) -> "BroadbandParams":
        """Build from a merged data-product config dict (e.g. gbt_high_time.yaml).

        Reads ``raw`` (df, dt, fch1), ``frame`` (fchans, tchans) and a
        ``broadband`` block. Missing keys fall back to the dataclass defaults.
        """
        raw = cfg.get("raw", {})
        frame = cfg.get("frame", {})
        bb = cfg.get("broadband", {})
        tau = bb.get("tau_int_bins", [cls.tau_int_min, cls.tau_int_max])
        kw = dict(
            df=raw.get("df", cls.df),
            dt=raw.get("dt", cls.dt),
            fch1=raw.get("fch1", cls.fch1),
            fchans=frame.get("fchans", cls.fchans),
            tchans=frame.get("tchans", cls.tchans),
            dm_min=bb.get("dm_min", cls.dm_min),
            dm_max=bb.get("dm_max", cls.dm_max),
            tau_int_min=tau[0],
            tau_int_max=tau[1],
            tau_scatter_ms=bb.get("tau_scatter_ms", cls.tau_scatter_ms),
            nscint_max=bb.get("nscint_max", cls.nscint_max),
            frac_low=bb.get("frac_low", cls.frac_low),
            frac_high=bb.get("frac_high", cls.frac_high),
            snr_min=bb.get("snr_min", cls.snr_min),
            snr_max=bb.get("snr_max", cls.snr_max),
            snr_sigma=bb.get("snr_sigma", cls.snr_sigma),
        )
        return cls(**kw)


class BroadbandTransientGenerator:
    """Generate and inject dispersed broadband transient signals.

    Parameters
    ----------
    params : BroadbandParams, optional
        Geometry and sampling ranges. Defaults to ``BroadbandParams()``.
    seed : int, optional
        Seed for the internal ``numpy`` generator (reproducibility).
    """

    def __init__(self, params: Optional[BroadbandParams] = None, seed: Optional[int] = None):
        self.params = params or BroadbandParams()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Geometry / physics
    # ------------------------------------------------------------------ #
    @property
    def freqs_mhz(self) -> np.ndarray:
        """Absolute frequency (MHz) of each channel index."""
        p = self.params
        df_mhz = p.df / 1e6
        idx = np.arange(p.fchans)
        return p.fch1 + idx * df_mhz if p.ascending else p.fch1 - idx * df_mhz

    def dispersion_delay(self, freqs_mhz: np.ndarray, DM: float, f_ref_mhz: float) -> np.ndarray:
        """Cold-plasma dispersion delay (s) relative to ``f_ref_mhz``."""
        freqs_mhz = np.asarray(freqs_mhz, dtype=float)
        return K_DM * DM * (freqs_mhz ** -2 - f_ref_mhz ** -2)

    # ------------------------------------------------------------------ #
    # Pulse construction (ported from hey-aliens SimulatedFRB, dispersed)
    # ------------------------------------------------------------------ #
    def _pulse_profile(
        self,
        freqs_mhz: np.ndarray,
        t0_bin: float,
        DM: float,
        tau_int: float,
        tau_scatter_ms: float,
        f_ref_mhz: float,
    ) -> np.ndarray:
        """Per-channel pulse, shape ``(nchan, nt)`` = (freq, time).

        Each channel carries a unit-peak Gaussian (width ``tau_int`` bins)
        centered at its **dispersed arrival time** ``t0_bin + delay``, convolved
        with a causal exponential scattering kernel ``exp(-t/tau_nu)`` whose
        timescale scales as ``(nu/f_ref)^-4``. Pulses are placed by index (no
        ``np.roll``) so bursts near the window edge are not wrapped.
        """
        p = self.params
        nt = p.tchans

        delay_bins = self.dispersion_delay(freqs_mhz, DM, f_ref_mhz) / p.dt
        centers = (t0_bin + delay_bins)[:, None]          # (nchan, 1)
        t = np.arange(nt)[None, :]                         # (1, nt)
        gaus = np.exp(-((t - centers) / float(tau_int)) ** 2)  # (nchan, nt), unit peak

        # Causal scattering kernel per channel, unit area (redistributes, no gain).
        tau_nu_bins = (tau_scatter_ms * 1e-3 / p.dt) * (freqs_mhz / f_ref_mhz) ** -4
        tau_nu_bins = np.maximum(tau_nu_bins, 1e-6)[:, None]
        lag = np.arange(nt)[None, :]
        kernel = np.exp(-lag / tau_nu_bins)
        kernel /= kernel.sum(axis=1, keepdims=True)

        # Row-wise convolution along time, keep causal alignment (first nt samples).
        pulse = fftconvolve(gaus, kernel, axes=1, mode="full")[:, :nt]
        return pulse

    def _scintillate(
        self, pulse: np.ndarray, freqs_mhz: np.ndarray, f_ref_mhz: float
    ) -> Tuple[np.ndarray, float]:
        """Multiply by a cosine spectral envelope (random phase, random nscint)."""
        phi = self.rng.random()
        nscint = float(np.exp(self.rng.uniform(np.log(1e-3), np.log(self.params.nscint_max))))
        if nscint < 1:
            nscint = 0.0
        env = np.cos(2 * np.pi * nscint * (freqs_mhz / f_ref_mhz) ** -2 + phi)
        env[env < 0] = 0.0
        env += 0.1
        return pulse * env[:, None], nscint

    def _fractional_bandwidth(self, pulse: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Keep only a random contiguous channel fraction; zero the rest."""
        p = self.params
        nchan = pulse.shape[0]
        frac = self.rng.uniform(p.frac_low, p.frac_high)
        hi = max(1, int(nchan * (1 - frac)))
        stch = int(self.rng.integers(0, hi))
        stop = min(nchan, stch + int(nchan * frac))
        masked = np.zeros_like(pulse)
        masked[stch:stop] = pulse[stch:stop]
        return masked, (stch, stop)

    def _sample_snr(self) -> float:
        """Log-normal peak SNR, rejection-sampled into ``[snr_min, snr_max]``."""
        p = self.params
        for _ in range(100):
            snr = p.snr_min + self.rng.lognormal(mean=1.0, sigma=p.snr_sigma)
            if snr <= p.snr_max:
                return float(snr)
        return float(p.snr_max)

    # ------------------------------------------------------------------ #
    # Injection
    # ------------------------------------------------------------------ #
    def inject_signal(
        self,
        data: np.ndarray,
        snr: Optional[float] = None,
        DM: Optional[float] = None,
        t0: Optional[float] = None,
        f_ref_mhz: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Inject one broadband transient into ``data``.

        Parameters
        ----------
        data : np.ndarray
            Background of shape ``(tchans, fchans)`` = (time, freq). Real or
            synthetic; used both as the additive background and to estimate the
            noise level for SNR scaling.
        snr, DM, t0, f_ref_mhz : optional
            Override the sampled values. ``t0`` is the burst arrival bin at the
            reference frequency; ``DM`` in pc/cm^3; ``f_ref_mhz`` defaults to the
            top of the band (zero delay there).

        Returns
        -------
        (out, info) : (np.ndarray, dict)
            ``out`` has the same shape as ``data``; ``info`` records the
            parameters actually used.
        """
        p = self.params
        data = np.asarray(data, dtype=float)
        if data.shape != (p.tchans, p.fchans):
            raise ValueError(
                f"data shape {data.shape} != (tchans, fchans) = ({p.tchans}, {p.fchans})"
            )

        freqs = self.freqs_mhz
        if f_ref_mhz is None:
            f_ref_mhz = float(freqs.max())
        if DM is None:
            DM = float(self.rng.uniform(p.dm_min, p.dm_max))
        if t0 is None:
            t0 = float(self.rng.uniform(0, p.tchans))
        tau_int = float(self.rng.uniform(p.tau_int_min, p.tau_int_max))
        if snr is None:
            snr = self._sample_snr()

        # (nchan, nt) pulse, dispersed
        pulse = self._pulse_profile(freqs, t0, DM, tau_int, p.tau_scatter_ms, f_ref_mhz)
        nscint = None
        if p.scintillate:
            pulse, nscint = self._scintillate(pulse, freqs, f_ref_mhz)
        pulse, frac_bw = self._fractional_bandwidth(pulse)

        # --- Transient SNR scaling -------------------------------------- #
        # Dedisperse the pulse at its own DM, frequency-average -> profile(t);
        # set its peak to snr * std(frequency-averaged background time series).
        delay_bins = np.round(self.dispersion_delay(freqs, DM, f_ref_mhz) / p.dt).astype(int)
        aligned = np.stack(
            [np.roll(pulse[i], -delay_bins[i]) for i in range(pulse.shape[0])], axis=0
        )
        peak = float(aligned.mean(axis=0).max())
        noise_std = float(data.mean(axis=1).std())

        if peak <= 1e-12:
            # Pulse fell entirely outside the window (e.g. very high DM): no signal.
            signal_t = np.zeros((p.tchans, p.fchans))
            scale = 0.0
        else:
            scale = (snr * noise_std) / peak
            signal_t = (pulse * scale).T  # -> (tchans, fchans)

        out = data + signal_t
        info = {
            "DM": DM,
            "t0": t0,
            "snr": snr,
            "tau_int": tau_int,
            "tau_scatter_ms": p.tau_scatter_ms,
            "nscint": nscint,
            "frac_bw": frac_bw,
            "f_ref_mhz": f_ref_mhz,
            "scale": scale,
        }
        return out, info

    # ------------------------------------------------------------------ #
    # setigen-backed synthetic background (optional convenience)
    # ------------------------------------------------------------------ #
    def synthetic_background(self) -> np.ndarray:
        """Generate a chi^2 noise background via setigen, shape ``(tchans, fchans)``.

        setigen is imported lazily so the generator (and its tests) work without
        it; only this convenience method requires it.
        """
        import setigen as stg
        from astropy import units as u

        p = self.params
        frame = stg.Frame(
            fchans=p.fchans,
            tchans=p.tchans,
            df=p.df * u.Hz,
            dt=p.dt * u.s,
            fch1=p.fch1 * u.MHz,
            ascending=p.ascending,
            seed=int(self.rng.integers(2 ** 31)),
        )
        frame.add_noise(x_mean=p.noise_mean, noise_type="chi2")
        return np.asarray(frame.data, dtype=float)


# =========================================================================== #
# setigen-native generators (narrowband drifting, wideband pulsed)
# =========================================================================== #
class _SetigenInjector:
    """Shared base for the setigen-native injectors.

    Holds the seeded RNG plus the helpers both setigen generators need: frame
    construction, log-uniform SNR sampling, the stochastic (AR(1)) scintillation
    profile, and a chi^2 synthetic background. Subclasses implement
    ``inject_signal``. The numpy-based :class:`BroadbandTransientGenerator` does
    NOT inherit from this — it has a different (transient) SNR definition and no
    setigen in its hot path.

    SNR convention: setigen's frame-integrated ``Frame.get_intensity(snr)``. The
    RST per-ON rescaling (``sqrt(tchans / tchans_per_obs)``) is intentionally
    dropped: BL-Exotica-AD scores a single frame, not a 6-observation cadence.
    """

    def __init__(self, params, seed: Optional[int] = None):
        if stg is None or u is None:
            raise ImportError(
                "setigen and astropy are required for the setigen-native signal "
                "generators (NarrowbandDriftingGenerator / WidebandPulsedGenerator)."
            )
        self.params = params
        self.rng = np.random.default_rng(seed)

    def _make_frame(self, data: np.ndarray):
        """Build a setigen Frame from a COPY of ``data`` (so the input is intact)."""
        p = self.params
        return stg.Frame.from_data(
            df=p.df * u.Hz,
            dt=p.dt * u.s,
            fch1=p.fch1 * u.MHz,
            ascending=p.ascending,
            data=np.asarray(data, dtype=float).copy(),
        )

    def _sample_snr(self) -> float:
        """Log-uniform SNR in ``[snr_min, snr_max]`` (more weight at low SNR)."""
        p = self.params
        log_min, log_max = np.log10(p.snr_min), np.log10(p.snr_max)
        return float(10 ** self.rng.uniform(log_min, log_max))

    def _make_stochastic_t_profile(self, level: float, n_bins: int):
        """AR(1) red-noise amplitude modulation, unit-mean (ported from RST).

        Scintillation as a correlated log-normal envelope (not a clean sine,
        which the model could latch onto as a synthetic fingerprint).
        E[envelope] = 1, so the time-averaged level — and thus the SNR
        calibration — is preserved.
        """
        p = self.params
        dt = p.dt
        tau = self.rng.uniform(p.scint_timescale_min, p.scint_timescale_max)
        depth = self.rng.uniform(p.scint_depth_min, p.scint_depth_max)
        rho = np.exp(-dt / tau)
        x = np.zeros(n_bins)
        x[0] = self.rng.standard_normal()
        for i in range(1, n_bins):
            x[i] = rho * x[i - 1] + np.sqrt(1.0 - rho ** 2) * self.rng.standard_normal()
        envelope = np.exp(depth * x - depth ** 2 / 2.0)  # log-normal, E[env]=1
        series = level * envelope
        t_grid = np.arange(n_bins) * dt

        def t_profile(t):
            t = np.atleast_1d(np.asarray(t, dtype=float))
            return np.interp(t, t_grid, series)

        return t_profile

    def synthetic_background(self) -> np.ndarray:
        """Generate a chi^2 noise background via setigen, shape ``(tchans, fchans)``."""
        p = self.params
        frame = stg.Frame(
            fchans=p.fchans,
            tchans=p.tchans,
            df=p.df * u.Hz,
            dt=p.dt * u.s,
            fch1=p.fch1 * u.MHz,
            ascending=p.ascending,
            seed=int(self.rng.integers(2 ** 31)),
        )
        frame.add_noise(x_mean=p.noise_mean, noise_type="chi2")
        return np.asarray(frame.data, dtype=float)


@dataclass
class NarrowbandParams:
    """Frame geometry + narrowband-drifting sampling ranges.

    Geometry defaults match ``configs/data/gbt_fine.yaml``; signal ranges follow
    the RST ``SignalParams`` ETI path.
    """

    # Frame geometry
    df: float = 2.7939677238464355   # Hz per channel
    dt: float = 18.25361108          # s per time bin
    fch1: float = 0.0                # MHz; 0 => inject on existing data (relative positions only)
    fchans: int = 1024
    tchans: int = 32
    ascending: bool = False          # index increasing => frequency decreasing (RST convention)

    # Peak SNR, log-uniform in [snr_min, snr_max] (setigen frame-integrated convention)
    snr_min: float = 5.0
    snr_max: float = 50.0

    # Drift magnitude distribution: 'lognormal' (default; concentrates near the
    # Earth+exoplanet rotational scale ~0.3 Hz/s) or 'loguniform' (flat per decade).
    # The upper clip is window-limited, computed per frame from data.shape.
    drift_distribution: str = "lognormal"
    drift_median: float = 0.3        # Hz/s — geometric centre of the log-normal
    drift_log_sigma: float = 0.5     # spread in dex
    min_nonzero_drift: float = 0.01  # Hz/s sampler floor
    zero_drift_prob: float = 0.05    # P(exactly zero drift) — compensated beacon

    # Width: |DR|*dt + U(eti_width_offset_min, eti_width_offset_max), Hz
    eti_width_offset_min: float = 1.0
    eti_width_offset_max: float = 10.0

    # Frequency profiles (lorentzian/voigt add exo-IPM/ISM scattering wings).
    # Profile *weights* are code-level defaults (awkward as YAML tuples).
    freq_profiles: tuple = ("gaussian", "sinc2", "lorentzian", "voigt")
    freq_profile_weights: tuple = (0.55, 0.10, 0.20, 0.15)
    scatter_width_min: float = 3.0   # scattering-wing FWHM range (Hz)
    scatter_width_max: float = 40.0

    # Temporal profiles: constant or 'scintillating' (AR(1) red-noise)
    time_profiles: tuple = ("constant", "scintillating")
    time_profile_weights: tuple = (0.6, 0.4)
    scint_timescale_min: float = 60.0   # s — correlation timescale
    scint_timescale_max: float = 600.0
    scint_depth_min: float = 0.2        # log-amplitude modulation depth
    scint_depth_max: float = 0.6

    # Synthetic chi^2 background mean (synthetic_background only)
    noise_mean: float = 5.0

    @classmethod
    def from_config(cls, cfg: dict) -> "NarrowbandParams":
        """Build from a merged data-product config (e.g. gbt_fine.yaml).

        Reads ``raw`` (df, dt, fch1), ``frame`` (fchans, tchans) and a
        ``narrowband`` block. Missing keys fall back to the dataclass defaults.
        """
        raw = cfg.get("raw", {})
        frame = cfg.get("frame", {})
        nb = cfg.get("narrowband", {})
        kw = dict(
            df=raw.get("df", cls.df),
            dt=raw.get("dt", cls.dt),
            fch1=raw.get("fch1", cls.fch1),
            fchans=frame.get("fchans", cls.fchans),
            tchans=frame.get("tchans", cls.tchans),
            snr_min=nb.get("snr_min", cls.snr_min),
            snr_max=nb.get("snr_max", cls.snr_max),
            drift_distribution=nb.get("drift_distribution", cls.drift_distribution),
            drift_median=nb.get("drift_median", cls.drift_median),
            drift_log_sigma=nb.get("drift_log_sigma", cls.drift_log_sigma),
            min_nonzero_drift=nb.get("min_nonzero_drift", cls.min_nonzero_drift),
            zero_drift_prob=nb.get("zero_drift_prob", cls.zero_drift_prob),
            eti_width_offset_min=nb.get("eti_width_offset_min", cls.eti_width_offset_min),
            eti_width_offset_max=nb.get("eti_width_offset_max", cls.eti_width_offset_max),
            scatter_width_min=nb.get("scatter_width_min", cls.scatter_width_min),
            scatter_width_max=nb.get("scatter_width_max", cls.scatter_width_max),
            scint_timescale_min=nb.get("scint_timescale_min", cls.scint_timescale_min),
            scint_timescale_max=nb.get("scint_timescale_max", cls.scint_timescale_max),
            scint_depth_min=nb.get("scint_depth_min", cls.scint_depth_min),
            scint_depth_max=nb.get("scint_depth_max", cls.scint_depth_max),
            noise_mean=nb.get("noise_mean", cls.noise_mean),
        )
        return cls(**kw)


class NarrowbandDriftingGenerator(_SetigenInjector):
    """Inject narrowband drifting (ETI-like) signals.

    Mirrors the RST ``SignalGenerator.inject_signal`` ETI path: log-uniform SNR,
    signed drift from a log-normal magnitude (+ small zero-drift probability),
    and a randomly selected frequency/temporal profile. The 6 RFI types, the
    cadence/ON-OFF logic, and the supervised-classifier machinery are NOT ported
    (this is a single-frame, unsupervised anomaly-detection pipeline).
    """

    def _max_drift_rate(self, fchans: int, tchans: int) -> float:
        """Window-limited |drift|: traverse at most the full band in one frame."""
        p = self.params
        return (fchans * p.df) / (tchans * p.dt)

    def _slope_from_drift(self, drift_rate: float) -> float:
        """Track slope in (time-bin / freq-bin) space; metadata only."""
        p = self.params
        if abs(drift_rate) < 1e-9:
            return 1e9  # effectively vertical
        return (-1.0 / drift_rate) / (p.dt / p.df)

    def _sample_drift_magnitude(self, max_drift: float) -> float:
        """|drift rate| (Hz/s) from the configured distribution, clipped to range."""
        p = self.params
        lo, hi = p.min_nonzero_drift, max_drift
        if p.drift_distribution == "lognormal":
            log_mag = self.rng.normal(np.log10(p.drift_median), p.drift_log_sigma)
            magnitude = 10 ** log_mag
        elif p.drift_distribution == "loguniform":
            magnitude = 10 ** self.rng.uniform(np.log10(lo), np.log10(hi))
        else:
            raise ValueError(
                f"Unknown drift_distribution: {p.drift_distribution!r}. "
                "Choose 'lognormal' or 'loguniform'."
            )
        return float(np.clip(magnitude, lo, hi))

    def _sample_drift_rate(self, max_drift: float) -> float:
        """Signed drift (Hz/s): zero with prob ``zero_drift_prob``, else mag x sign."""
        p = self.params
        if self.rng.random() < p.zero_drift_prob:
            return 0.0
        magnitude = self._sample_drift_magnitude(max_drift)
        return float(magnitude * self.rng.choice([-1, 1]))

    def _calculate_eti_width(self, drift_rate: float) -> float:
        """ETI width (Hz): |DR|*dt smearing + intrinsic U(offset)."""
        p = self.params
        drift_component = abs(drift_rate) * p.dt
        offset = self.rng.uniform(p.eti_width_offset_min, p.eti_width_offset_max)
        return drift_component + offset

    def _select_f_profile(self, width: float) -> Tuple[object, str]:
        """Weighted choice of frequency profile (gaussian/sinc2/lorentzian/voigt)."""
        p = self.params
        choice = str(self.rng.choice(list(p.freq_profiles), p=list(p.freq_profile_weights)))
        if choice == "sinc2":
            return stg.sinc2_f_profile(width=width * u.Hz), choice
        if choice == "lorentzian":
            scatter = self.rng.uniform(p.scatter_width_min, p.scatter_width_max)
            return stg.lorentzian_f_profile(width=(width + scatter) * u.Hz), choice
        if choice == "voigt":
            scatter = self.rng.uniform(p.scatter_width_min, p.scatter_width_max)
            return stg.voigt_f_profile(g_width=width * u.Hz, l_width=scatter * u.Hz), choice
        return stg.gaussian_f_profile(width=width * u.Hz), "gaussian"

    def _select_t_profile(self, intensity: float, n_bins: int) -> Tuple[object, str]:
        """Weighted choice of temporal profile (constant/scintillating)."""
        p = self.params
        choice = str(self.rng.choice(list(p.time_profiles), p=list(p.time_profile_weights)))
        if choice == "scintillating":
            return self._make_stochastic_t_profile(intensity, n_bins), "scintillating"
        return stg.constant_t_profile(level=intensity), "constant"

    def inject_signal(
        self,
        data: np.ndarray,
        snr: Optional[float] = None,
        start_channel: Optional[int] = None,
        drift_rate: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Inject one narrowband drifting signal into ``data`` (time, freq).

        ``snr`` / ``start_channel`` / ``drift_rate`` override the sampled values
        (used by tests for determinism). When ``start_channel`` is not given it is
        constrained so the track stays in-band for the whole frame.

        Returns ``(out, info)`` with ``out`` the same shape as ``data``.
        """
        p = self.params
        data = np.asarray(data, dtype=float)
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()

        max_drift = self._max_drift_rate(fchans, tchans)
        if drift_rate is None:
            drift_rate = self._sample_drift_rate(max_drift)
        true_slope = self._slope_from_drift(drift_rate)

        # Constrain start_channel for full-frame visibility. setigen renders
        # drift>0 toward HIGHER channel index (verified, both `ascending`), so a
        # positive drift must leave room on the high-index side, and vice-versa.
        if start_channel is None:
            drift_chans = abs(drift_rate) / p.df * tchans * p.dt
            if drift_rate > 0:
                max_start = max(1, int(fchans - 1 - drift_chans))
                start_channel = int(self.rng.integers(1, max_start + 1))
            elif drift_rate < 0:
                min_start = min(fchans - 2, int(drift_chans + 1))
                start_channel = int(self.rng.integers(min_start, fchans - 1))
            else:
                start_channel = int(self.rng.integers(1, fchans - 1))

        width = self._calculate_eti_width(drift_rate)

        frame = self._make_frame(data)
        intensity = frame.get_intensity(snr=snr)  # frame-integrated SNR (no per-ON factor)
        f_profile, f_name = self._select_f_profile(width)
        t_profile, t_name = self._select_t_profile(intensity, tchans)

        frame.add_signal(
            stg.constant_path(
                f_start=frame.get_frequency(index=start_channel),
                drift_rate=drift_rate * u.Hz / u.s,
            ),
            t_profile,
            f_profile,
            stg.constant_bp_profile(level=1),
        )

        info = {
            "snr": snr,
            "drift_rate": drift_rate,
            "start_channel": int(start_channel),
            "width": width,
            "slope": true_slope,
            "f_profile": f_name,
            "t_profile": t_name,
        }
        return frame.data, info


@dataclass
class WidebandParams:
    """Frame geometry + wideband-pulsed sampling ranges.

    Geometry defaults match ``configs/data/gbt_moderate.yaml``. The period/pulse
    are sampled in **time bins** (relative to ``tchans``), then converted to
    seconds — RST's seconds-based ranges assume a 96-bin cadence and would give
    <2 pulses in the shorter intermediate-product frame.

    NOTE: this design is a reasonable default; revisit after the Vishal call.
    """

    # Frame geometry
    df: float = 2861.02294921875     # Hz per channel
    dt: float = 1.0737418239999998   # s per time bin
    fch1: float = 0.0                # MHz; 0 => inject on existing data
    fchans: int = 256
    tchans: int = 64
    ascending: bool = False

    # Peak SNR, log-uniform (setigen frame-integrated convention)
    snr_min: float = 5.0
    snr_max: float = 50.0

    # Frequency extent: width = U(frac_low, frac_high) * fchans channels
    freq_profiles: tuple = ("gaussian", "box")
    freq_profile_weights: tuple = (0.5, 0.5)
    frac_low: float = 0.1
    frac_high: float = 0.5

    # Periodic pulse train (in time bins -> seconds). Max period is a fraction of
    # the frame; pulse width is bounded by duty_max to keep distinct pulses.
    period_bins_min: float = 8.0
    period_bins_max_frac: float = 0.5   # max period = period_bins_max_frac * tchans
    duty_max: float = 0.33              # pulse_width <= duty_max * period

    # Near-stationary frequency drift jitter (Hz/s); negligible at this resolution
    drift_jitter: float = 0.05

    # Synthetic chi^2 background mean (synthetic_background only)
    noise_mean: float = 5.0

    @classmethod
    def from_config(cls, cfg: dict) -> "WidebandParams":
        """Build from a merged data-product config (e.g. gbt_moderate.yaml).

        Reads ``raw`` (df, dt, fch1), ``frame`` (fchans, tchans) and a
        ``wideband`` block. Missing keys fall back to the dataclass defaults.
        """
        raw = cfg.get("raw", {})
        frame = cfg.get("frame", {})
        wb = cfg.get("wideband", {})
        kw = dict(
            df=raw.get("df", cls.df),
            dt=raw.get("dt", cls.dt),
            fch1=raw.get("fch1", cls.fch1),
            fchans=frame.get("fchans", cls.fchans),
            tchans=frame.get("tchans", cls.tchans),
            snr_min=wb.get("snr_min", cls.snr_min),
            snr_max=wb.get("snr_max", cls.snr_max),
            frac_low=wb.get("frac_low", cls.frac_low),
            frac_high=wb.get("frac_high", cls.frac_high),
            period_bins_min=wb.get("period_bins_min", cls.period_bins_min),
            period_bins_max_frac=wb.get("period_bins_max_frac", cls.period_bins_max_frac),
            duty_max=wb.get("duty_max", cls.duty_max),
            drift_jitter=wb.get("drift_jitter", cls.drift_jitter),
            noise_mean=wb.get("noise_mean", cls.noise_mean),
        )
        return cls(**kw)


class WidebandPulsedGenerator(_SetigenInjector):
    """Inject wide-band periodic pulse trains (radar/beacon/pulsar-like).

    Combines a wide frequency extent (gaussian or box, a large fraction of the
    band) with setigen's ``periodic_gaussian_t_profile`` and near-zero drift.
    Default design — revisit after the Vishal call.

    SNR note: ``snr`` sets the *pulse-peak* amplitude via ``get_intensity(snr)``.
    Because the pulse train is "on" only a fraction (duty) of the frame, the
    frame-integrated SNR is ~snr*duty (lower) — so here "snr" is a per-pulse peak
    level, NOT a frame-integrated SNR like narrowband. The exact SNR definition
    is a design question for Vishal.
    """

    def _select_f_profile(self, width_hz: float) -> Tuple[object, str]:
        """Weighted choice of a *wide* frequency profile (gaussian or box)."""
        p = self.params
        choice = str(self.rng.choice(list(p.freq_profiles), p=list(p.freq_profile_weights)))
        if choice == "box":
            return stg.box_f_profile(width=width_hz * u.Hz), choice
        return stg.gaussian_f_profile(width=width_hz * u.Hz), "gaussian"

    def inject_signal(
        self,
        data: np.ndarray,
        snr: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Inject one wide-band pulse train into ``data`` (time, freq).

        Returns ``(out, info)`` with ``out`` the same shape as ``data``.
        """
        p = self.params
        data = np.asarray(data, dtype=float)
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()

        # Wide frequency extent.
        frac = self.rng.uniform(p.frac_low, p.frac_high)
        width_chans = max(1.0, frac * fchans)
        width_hz = width_chans * p.df
        start_channel = int(self.rng.integers(1, fchans - 1))

        # Periodic pulse train, sampled in bins then converted to seconds.
        period_max = max(p.period_bins_min + 1.0, p.period_bins_max_frac * tchans)
        period_bins = float(self.rng.uniform(p.period_bins_min, period_max))
        pulse_width_bins = float(self.rng.uniform(1.0, max(1.0, p.duty_max * period_bins)))
        period_s = period_bins * p.dt
        pulse_width_s = pulse_width_bins * p.dt

        drift_rate = float(self.rng.uniform(-p.drift_jitter, p.drift_jitter))

        frame = self._make_frame(data)
        intensity = frame.get_intensity(snr=snr)
        f_profile, f_name = self._select_f_profile(width_hz)
        t_profile = stg.periodic_gaussian_t_profile(
            pulse_width=pulse_width_s * u.s,
            period=period_s * u.s,
            pulse_direction="up",
            amplitude=intensity,
            level=0.0,
            min_level=0.0,
            seed=int(self.rng.integers(0, 2 ** 31)),
        )

        frame.add_signal(
            stg.constant_path(
                f_start=frame.get_frequency(index=start_channel),
                drift_rate=drift_rate * u.Hz / u.s,
            ),
            t_profile,
            f_profile,
            stg.constant_bp_profile(level=1),
        )

        info = {
            "snr": snr,
            "period_bins": period_bins,
            "pulse_width_bins": pulse_width_bins,
            "width": width_hz,
            "drift_rate": drift_rate,
            "start_channel": start_channel,
            "f_profile": f_name,
        }
        return frame.data, info
