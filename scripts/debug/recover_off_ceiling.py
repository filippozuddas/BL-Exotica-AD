"""Recompute a cadence's ``off_ceiling`` from a saved ``maps.npz``.

``scripts/inference.py`` only prints the per-cadence OFF-noise-core ceiling
(line ~487); it is never written to the candidates CSV, so a run whose logs are
gone leaves no record of the threshold that gated the short list. The ``.npz``
written by ``scripts/recompute_anomaly_maps.py`` stores everything the ceiling
is built from (selected + probe component maps, ``thresh_3``/``thresh_5``), so
it can be reconstructed exactly.

Also dumps the per-row ON/OFF peaks behind ``full_row_hits`` for one candidate,
so a short-list membership decision can be checked cell by cell.

Usage:
    python scripts/debug/recover_off_ceiling.py <maps.npz> [--f_start 43628544]
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.search.candidates import off_noise_ceiling, full_row_hits, on_off_contrast

MIN_OFF_POOL = 30
OFF_ROWS = (1, 3, 5)
ON_ROWS = (0, 2, 4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--f_start", type=int, default=None,
                   help="candidate f_start to break down row by row")
    p.add_argument("--weights", type=float, nargs=3, default=(0.5, 0.5, 0.5),
                   help="UDMA score_weights (w1,w2,w3) for st1/st2/ss -> cob")
    args = p.parse_args()

    d = np.load(args.npz)
    w1, w2, w3 = args.weights
    cob = w1 * d["st1"] + w2 * d["st2"] + w3 * d["ss"]
    probe_cob = w1 * d["probe_st1"] + w2 * d["probe_st2"] + w3 * d["probe_ss"]
    thresh_5 = float(d["thresh_5"])

    off_idx = [r for r in OFF_ROWS if r < cob.shape[1]]
    off_pool = np.concatenate([
        cob[:, off_idx, :].ravel(),
        probe_cob[:, off_idx, :].ravel(),
    ])

    if len(off_pool) >= MIN_OFF_POOL:
        raw = off_noise_ceiling(off_pool)
        off_ceiling = max(raw, thresh_5)
    else:
        raw = float("nan")
        off_ceiling = thresh_5

    print(f"{args.npz}")
    print(f"  selected maps : {cob.shape}   probe maps: {probe_cob.shape}")
    print(f"  OFF pool cells: {len(off_pool)}")
    print(f"  raw off_noise_ceiling = {raw:.4f}")
    print(f"  thresh_3 (Gaussian)   = {float(d['thresh_3']):.4f}")
    print(f"  thresh_5 (floor)      = {thresh_5:.4f}")
    print(f"  -> off_ceiling        = {off_ceiling:.4f}"
          f"   ({'AT THE FLOOR' if off_ceiling == thresh_5 else 'from OFF core'})")

    if args.f_start is None:
        return

    hits = np.nonzero(d["f_start"] == args.f_start)[0]
    if not len(hits):
        near = d["f_start"][np.argsort(np.abs(d["f_start"] - args.f_start))[:5]]
        print(f"\n  f_start {args.f_start} not in this npz; nearest: {near.tolist()}")
        return

    amap = cob[hits[0]]
    print(f"\n  candidate f_start={args.f_start}  score={float(d['score'][hits[0]]):.4f}")
    print(f"  row peaks (max over all 64 cols), threshold={off_ceiling:.4f}:")
    for r in range(amap.shape[0]):
        kind = "ON " if r in ON_ROWS else "OFF"
        peak = float(amap[r].max())
        print(f"    row {r} ({kind}): peak={peak:.4f} col={int(amap[r].argmax()):3d}"
              f"  {'HIT' if peak > off_ceiling else '   '}")
    fr = full_row_hits(amap, threshold=off_ceiling)
    oo = on_off_contrast(amap, threshold=off_ceiling)
    on_ref = min(float(amap[r].max()) for r in ON_ROWS if r < amap.shape[0])
    print(f"  leak magnitude gate: OFF peak must also exceed 0.3 * {on_ref:.4f} "
          f"= {0.3 * on_ref:.4f}")
    print(f"  {fr}")
    print(f"  {oo}")


if __name__ == "__main__":
    main()
