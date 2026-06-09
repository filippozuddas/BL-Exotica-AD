#!/usr/bin/env python
"""Visual sanity check for the setigen-native signal generators.

Generates narrowband drifting and/or wideband pulsed signals on a setigen chi^2
background and saves, for each, a companion panel (top) plus the waterfall (bottom).

Axis conventions follow the physical nature of each signal class:
  narrowband  (0000): x=channel (freq proxy), y=time — drift track is diagonal
  wideband    (0002): x=time, y=channel        — periodic pulse stripes are vertical

The companion panel is chosen to align with the waterfall x-axis:
  narrowband: time-averaged spectrum  (x=channel) — shows signal peak
  wideband:   freq-averaged time series (x=time)  — shows periodic pulse train

Waterfall panel height is computed from the actual y-axis bins so the figure
proportions adapt to any (tchans, fchans) from config.

By default the waterfall shows the *noise-subtracted injected signal* (out - bg).
Pass ``--with-noise`` for the realistic (background-included) view.

Run from the repo root:

    python scripts/preview_signals.py --n 9 --out outputs/preview_signals.png
    python scripts/preview_signals.py --kind narrowband --snr 20
    python scripts/preview_signals.py --kind wideband --n 6 --with-noise
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic import (  # noqa: E402
    NarrowbandParams,
    NarrowbandDriftingGenerator,
    WidebandParams,
    WidebandPulsedGenerator,
)


def _wf_h(displayed_y_bins: int, px_per_inch: float = 50.0) -> float:
    """Waterfall panel height (inches) proportional to y-axis bins, clamped [0.8, 3.0]."""
    return float(np.clip(displayed_y_bins / px_per_inch, 0.8, 3.0))


def _make_panel(p, gen, kind, snr, with_noise):
    """Inject one example; return (img, ts_profile, ts_x, extent, wf_xlabel, wf_ylabel, title).

    img is already in the correct orientation for imshow (origin='lower'):
      narrowband: (tchans, fchans) — rows=time (y), cols=channel (x)
      wideband:   (fchans, tchans) — rows=channel (y), cols=time (x)
    ts_profile aligns with the waterfall x-axis so sharex works correctly.
    """
    bg = gen.synthetic_background()
    out, info = gen.inject_signal(bg, snr=snr)
    raw = out if with_noise else (out - bg)  # always (tchans, fchans)

    t_s = np.arange(p.tchans) * p.dt

    if kind == "narrowband":
        img = raw                          # (tchans, fchans); no transpose
        extent = [0, p.fchans, t_s[-1], t_s[0]]
        wf_xlabel, wf_ylabel = "channel", "time (s)"
        ts_profile = raw.mean(axis=0)      # time-averaged spectrum (fchans,)
        ts_x = np.arange(p.fchans, dtype=float)
        title = (f"NB  drift={info['drift_rate']:+.2f} Hz/s  "
                 f"snr={info['snr']:.1f}  {info['f_profile']}")
    else:  # wideband
        img = raw.T                        # (fchans, tchans); transposed
        extent = [t_s[0], t_s[-1], 0, p.fchans]
        wf_xlabel, wf_ylabel = "time (s)", "channel"
        ts_profile = raw.mean(axis=1)      # freq-averaged time series (tchans,)
        ts_x = t_s
        title = (f"WB  period={info['period_bins']:.0f} bins  "
                 f"snr={info['snr']:.1f}  {info['f_profile']}")

    return img, ts_profile, ts_x, extent, wf_xlabel, wf_ylabel, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["narrowband", "wideband", "both"],
                    default="both", help="signal class(es) to preview")
    ap.add_argument("--n", type=int, default=4, help="examples per signal class")
    ap.add_argument("--snr", type=float, default=None, help="fix SNR (else sampled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-noise", action="store_true",
                    help="show the raw waterfall (default: noise-subtracted signal)")
    ap.add_argument("--out", type=str, default="outputs/preview_signals.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["narrowband", "wideband"] if args.kind == "both" else [args.kind]
    specs = []
    for kind in kinds:
        if kind == "narrowband":
            p = NarrowbandParams()
            gen = NarrowbandDriftingGenerator(p, seed=args.seed)
        else:
            p = WidebandParams()
            gen = WidebandPulsedGenerator(p, seed=args.seed)
        for _ in range(args.n):
            specs.append((kind, gen, p))

    n = len(specs)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    panel_w = 4.5
    ts_h = 0.7
    row_gap = 1.0

    # wf_h per row: max across panels in that row (handles mixed narrowband/wideband rows).
    row_wf_h = []
    for r in range(nrows):
        row_specs = [specs[r * ncols + c] for c in range(ncols) if r * ncols + c < n]
        h = max(
            _wf_h(s[2].tchans if s[0] == "narrowband" else s[2].fchans)
            for s in row_specs
        )
        row_wf_h.append(h)

    total_w = panel_w * ncols
    if nrows == 1:
        row_heights = [ts_h, row_wf_h[0]]
        total_h = ts_h + row_wf_h[0]
    else:
        row_heights = []
        total_h = 0.0
        for i in range(nrows):
            row_heights += [ts_h, row_wf_h[i]]
            total_h += ts_h + row_wf_h[i]
            if i < nrows - 1:
                row_heights.append(row_gap)
                total_h += row_gap

    fig = plt.figure(figsize=(total_w, total_h))
    gs = fig.add_gridspec(
        len(row_heights), ncols, height_ratios=row_heights,
        hspace=0.1, wspace=0.30, left=0.07, right=0.97, top=0.95, bottom=0.07,
    )

    for k, (kind, gen, p) in enumerate(specs):
        r, c = k // ncols, k % ncols
        gs_row_ts = r * 3 if nrows > 1 else r * 2
        ax_ts = fig.add_subplot(gs[gs_row_ts, c])
        ax_wf = fig.add_subplot(gs[gs_row_ts + 1, c], sharex=ax_ts)

        img, ts_profile, ts_x, extent, wf_xlabel, wf_ylabel, title = \
            _make_panel(p, gen, kind, args.snr, args.with_noise)

        ax_ts.plot(ts_x, ts_profile, color="tab:blue", lw=0.8)
        ax_ts.set_title(title, fontsize=8, pad=3)
        ax_ts.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_ts.tick_params(axis="y", which="both", left=False, labelleft=False)
        for spine in ax_ts.spines.values():
            spine.set_linewidth(0.6)

        ax_wf.imshow(img, aspect="auto", origin="upper", extent=extent, cmap="viridis")
        ax_wf.set_xlabel(wf_xlabel, fontsize=8)
        ax_wf.set_ylabel(wf_ylabel, fontsize=8)
        ax_wf.tick_params(labelsize=7)
        for spine in ax_wf.spines.values():
            spine.set_linewidth(0.6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {n} examples ({', '.join(kinds)}) to {out_path}")


if __name__ == "__main__":
    main()
