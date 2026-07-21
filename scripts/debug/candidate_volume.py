"""How much candidate volume does relaxing the stage-1 pre-cut actually cost?

``scripts/debug/threshold_regime.py`` found that on ~28% of cadences the
stage-1 relative cut (``far_thresh``, the top 1% of scores) sits *above* the
stage-3 absolute gate (``off_ceiling``) — by up to 88x. Those cadences discard
snippets the ON/OFF filter would have accepted: stage 1, meant only to
deduplicate and cap volume, is silently acting as the strictest judge in the
pipeline.

The principled fix is to forbid that inversion:

    stage-1 threshold  =  min(far_thresh, off_ceiling)

so nothing that could survive stage 3 dies before reaching it. The only cost is
volume — more clusters to carry, plot and (at the tail) eyeball. This script
measures that cost exactly, offline.

Cheap because ``scripts/recompute_anomaly_maps.py`` already cached every
snippet score per cadence under ``<map_dir>/_scores/cadNN.npz`` (built in one
pass over the 2.8 GB ``inference_scores.csv``). No HDD access, no model, and
**all 364 cadences** are available — not just the ones whose anomaly maps have
been recomputed so far.

``off_ceiling`` needs the probe maps, so it is read from
``threshold_regime.csv`` where available. For cadences without maps yet it
falls back to ``thresh_5``, which ``threshold_regime.py`` measured to be the
binding term of ``max(off_noise_ceiling, thresh_5)`` on 100% of cadences
checked — flagged per row via ``ceiling_source`` so the assumption stays
visible.

Usage:
    python scripts/debug/candidate_volume.py \
        --map_dir outputs/inference/exotica_heldout_maps \
        --regime_csv outputs/inference/threshold_regime.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

FAR_QUANTILE = 0.99


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-sigma, matching ``scripts/inference.py``."""
    median = float(np.median(x))
    return median, float(np.median(np.abs(x - median)) * 1.4826)


def n_clusters(f_sorted: np.ndarray, stride: int) -> int:
    """Connected components on the frequency axis, as ``cluster_candidates``."""
    if f_sorted.size == 0:
        return 0
    return int(1 + (np.diff(f_sorted) > stride).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map_dir", type=Path, required=True)
    p.add_argument("--regime_csv", type=Path, default=None)
    p.add_argument("--stride", type=int, default=None,
                   help="override; default reads it from any maps.npz")
    p.add_argument("--out_csv", type=Path, default=None)
    args = p.parse_args()

    score_dir = args.map_dir / "_scores"
    if not score_dir.exists():
        raise SystemExit(f"No score cache at {score_dir}")

    stride = args.stride
    if stride is None:
        any_map = next(iter(sorted(args.map_dir.glob("cad*/maps.npz"))), None)
        if any_map is None:
            raise SystemExit("No maps.npz to read --stride from; pass it explicitly")
        with np.load(any_map) as z:
            stride = int(z["stride"])
    print(f"stride = {stride}")

    ceil_by_idx, label_by_idx = {}, {}
    if args.regime_csv and args.regime_csv.exists():
        reg = pd.read_csv(args.regime_csv)
        ceil_by_idx = dict(zip(reg["cad_idx"], reg["off_ceiling"]))
        label_by_idx = dict(zip(reg["cad_idx"], reg["label"]))

    rows = []
    for npz_path in sorted(score_dir.glob("cad*.npz")):
        cad_idx = int(re.match(r"cad(\d+)", npz_path.name).group(1))
        with np.load(npz_path) as z:
            f_start, score = z["f_start"], z["score"]

        median, mad_sigma = robust_stats(score)
        thresh_5 = median + 5 * mad_sigma
        far = float(np.quantile(score, FAR_QUANTILE))

        measured = ceil_by_idx.get(cad_idx)
        ceiling = float(measured) if measured is not None else float(thresh_5)
        new_thresh = min(far, ceiling)

        rows.append({
            "cad_idx": cad_idx,
            "label": label_by_idx.get(cad_idx, f"cad{cad_idx:02d}"),
            "ceiling_source": "measured" if measured is not None else "thresh_5 (assumed)",
            "n_snippets": int(score.size),
            "far_thresh": far,
            "off_ceiling": ceiling,
            "far_over_ceiling": far / ceiling,
            "n_clusters_now": n_clusters(f_start[score > far], stride),
            "n_clusters_new": n_clusters(f_start[score > new_thresh], stride),
        })

    df = pd.DataFrame(rows).sort_values("cad_idx").reset_index(drop=True)
    df["cluster_multiplier"] = df["n_clusters_new"] / df["n_clusters_now"].clip(lower=1)
    changed = df["far_over_ceiling"] > 1.0

    n = len(df)
    print(f"\n{'=' * 78}\nCANDIDATE VOLUME under stage-1 = min(far_thresh, off_ceiling)"
          f"\n{n} cadences ({(df['ceiling_source'] == 'measured').sum()} with measured "
          f"ceiling)\n{'=' * 78}")

    print(f"\nCadences affected at all (far > ceiling): {changed.sum()} "
          f"({100 * changed.sum() / n:.1f}%)  — the rest are unchanged by construction")

    print(f"\nTotal clusters   now: {df['n_clusters_now'].sum():>9,d}")
    print(f"                 new: {df['n_clusters_new'].sum():>9,d}"
          f"   (x{df['n_clusters_new'].sum() / max(df['n_clusters_now'].sum(), 1):.2f})")

    sub = df[changed]
    if len(sub):
        m = sub["cluster_multiplier"]
        print("\nPer-cadence cluster multiplier, affected cadences only:")
        print("  median {:.1f}   p75 {:.1f}   p90 {:.1f}   max {:.1f}".format(
            m.median(), *np.quantile(m, [0.75, 0.9]), m.max()))
        print("\nNew clusters per affected cadence:")
        c = sub["n_clusters_new"]
        print("  median {:.0f}   p90 {:.0f}   max {:.0f}".format(
            c.median(), np.quantile(c, 0.9), c.max()))

    print("\nHeaviest cadences after the change:")
    cols = ["cad_idx", "label", "far_over_ceiling", "n_clusters_now",
            "n_clusters_new", "cluster_multiplier"]
    print(df.nlargest(12, "n_clusters_new")[cols].to_string(index=False))

    out = args.out_csv or args.map_dir.parent / "candidate_volume.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("\nNote: these are CLUSTERS entering stage 3, not short-list entries. "
          "The ON/OFF filter still applies; on the 364-CSV aggregation it "
          "admitted 2.9% of clusters.")


if __name__ == "__main__":
    main()
