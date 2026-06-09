#!/usr/bin/env python
"""Visual sanity check for the broadband transient generator.

Generates a few dispersed broadband pulses on a setigen chi^2 background and
saves, for each, the waterfall plus its frequency-averaged time series — the
same view as Gajjar et al. 2022 Fig. A1.

Run from the repo root (needs setigen + matplotlib):

    python scripts/preview_broadband.py --n 9 --out outputs/preview_broadband.png
    python scripts/preview_broadband.py --dm 150 --snr 12   # fix DM / SNR
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic import BroadbandParams, BroadbandTransientGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9, help="number of examples")
    ap.add_argument("--dm", type=float, default=None, help="fix DM (pc/cm^3)")
    ap.add_argument("--snr", type=float, default=None, help="fix peak SNR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/preview_broadband.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = BroadbandParams()
    gen = BroadbandTransientGenerator(p, seed=args.seed)

    freqs = gen.freqs_mhz
    t_ms = np.arange(p.tchans) * p.dt * 1e3
    extent = [t_ms[0], t_ms[-1], freqs.min(), freqs.max()]

    ncols = int(np.ceil(np.sqrt(args.n)))
    nrows = int(np.ceil(args.n / ncols))

    # Gajjar et al. 2022 Fig. A1 layout: x=time, y=freq.
    # wf_h adapts to p.fchans so the figure proportions match the actual data shape.
    panel_w = 5.5
    ts_h = 0.7
    wf_h = float(np.clip(p.fchans / 50.0, 0.8, 3.0))
    row_gap = 1.0

    total_w = panel_w * ncols
    total_h = (ts_h + wf_h) * nrows + row_gap * (nrows - 1)
    fig = plt.figure(figsize=(total_w, total_h))

    # Build a nested GridSpec: outer rows have explicit spacing via row heights.
    # We encode gaps as zero-height spacer rows between row-groups.
    if nrows == 1:
        row_heights = [ts_h, wf_h]
    else:
        row_heights = []
        for i in range(nrows):
            row_heights.append(ts_h)
            row_heights.append(wf_h)
            if i < nrows - 1:
                row_heights.append(row_gap)

    gs = fig.add_gridspec(
        len(row_heights), ncols,
        height_ratios=row_heights,
        hspace=0.1,
        wspace=0.30,
        left=0.07, right=0.97, top=0.97, bottom=0.07,
    )

    for k in range(args.n):
        r, c = k // ncols, k % ncols
        # with spacer rows: each row-group occupies 3 rows (ts, wf, gap) except last (ts, wf)
        gs_row_ts = r * 3 if nrows > 1 else r * 2
        gs_row_wf = gs_row_ts + 1

        bg = gen.synthetic_background()
        out, info = gen.inject_signal(bg, snr=args.snr, DM=args.dm)

        ax_ts = fig.add_subplot(gs[gs_row_ts, c])
        ax_wf = fig.add_subplot(gs[gs_row_wf, c], sharex=ax_ts)

        # Time series: keep the box, hide x-ticks (shared with waterfall) and y-ticks
        ax_ts.plot(t_ms, out.mean(axis=1), color="tab:blue", lw=0.8)
        ax_ts.set_title(f"DM={info['DM']:.0f}  SNR={info['snr']:.1f}", fontsize=8, pad=3)
        ax_ts.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_ts.tick_params(axis="y", which="both", left=False, labelleft=False)
        for spine in ax_ts.spines.values():
            spine.set_linewidth(0.6)

        # Waterfall: auto aspect so it fills the allocated axes area naturally
        ax_wf.imshow(out.T, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        ax_wf.set_xlabel("time (ms)", fontsize=8)
        ax_wf.set_ylabel("freq (MHz)", fontsize=8)
        ax_wf.tick_params(labelsize=7)
        for spine in ax_wf.spines.values():
            spine.set_linewidth(0.6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {args.n} examples to {out_path}")


if __name__ == "__main__":
    main()
