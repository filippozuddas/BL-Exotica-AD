"""How many of the newly-admitted candidates would actually reach the short list?

``scripts/debug/candidate_volume.py`` measured that fixing stage 1 to
``min(far_thresh, off_ceiling)`` grows cluster volume by ~1.47x overall (more
in the affected third of cadences). Cluster counts are a machine number; the
real constraint is the human review budget, which only cares about entries
that pass stage 3 (``full_row_hits``) and land in the short list.

This closes that gap — with one caveat that must stay visible, not silently
averaged away.

**Coverage caveat.** ``scripts/recompute_anomaly_maps.py`` saved full (6, 64)
maps only for snippets scoring at or above the 95th percentile
(``map_thresh``), on the assumption that the new stage-1 threshold would
always be looser than that (``FAR_QUANTILE=0.99`` is stricter than q95, so
``far_thresh``-bound cadences are always covered). But where ``off_ceiling``
is the smaller term of ``min(far_thresh, off_ceiling)`` — the exact cadences
this fix targets — the new threshold can fall *below* ``map_thresh``, and
those newly-admitted clusters have no saved map: ``full_row_hits`` cannot be
computed for them without a fresh forward pass (i.e. touching the HDD again).
This script reports that coverage gap explicitly per cadence rather than
pretending the covered subset is the whole answer.

Reuses the already-validated per-cadence thresholds from
``threshold_regime.csv`` (run ``threshold_regime.py`` first) rather than
recomputing them, so the two scripts never disagree on what "the ceiling" is.

Usage:
    python scripts/debug/shortlist_review_load.py \
        --map_dir outputs/inference/exotica_heldout_maps \
        --regime_csv outputs/inference/threshold_regime.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.search.candidates import cluster_candidates, full_row_hits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map_dir", type=Path, required=True)
    p.add_argument("--regime_csv", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--out_csv", type=Path, default=None)
    args = p.parse_args()

    reg = pd.read_csv(args.regime_csv).set_index("cad_idx")
    score_dir = args.map_dir / "_scores"
    w1, w2, w3 = args.weights

    rows = []
    for map_npz in sorted(args.map_dir.glob("cad*/maps.npz")):
        label = map_npz.parent.name
        cad_idx = int(re.match(r"cad(\d+)", label).group(1))
        if cad_idx not in reg.index:
            continue
        off_ceiling = float(reg.loc[cad_idx, "off_ceiling"])
        far_thresh = float(reg.loc[cad_idx, "far_thresh"])
        new_thresh = min(far_thresh, off_ceiling)

        score_npz = score_dir / f"cad{cad_idx:02d}.npz"
        if not score_npz.exists():
            continue
        with np.load(score_npz) as sc:
            all_f, all_s = sc["f_start"], sc["score"]
        with np.load(map_npz) as mp:
            stride = int(mp["stride"])
            fchans = int(mp["fchans"])
            map_thresh = float(mp["map_thresh"])
            map_f = mp["f_start"]
            map_st1, map_st2, map_ss = mp["st1"], mp["st2"], mp["ss"]

        old_clusters = cluster_candidates(all_f, all_s, threshold=far_thresh,
                                          stride=stride, fchans=fchans)
        new_clusters = cluster_candidates(all_f, all_s, threshold=new_thresh,
                                          stride=stride, fchans=fchans)

        n_new = len(new_clusters)
        n_old = len(old_clusters)
        if n_new == 0:
            rows.append(dict(cad_idx=cad_idx, label=label, n_old=n_old, n_new=0,
                              n_covered=0, n_uncovered=0, n_short_list_covered=0))
            continue

        map_f_to_idx = {int(f): i for i, f in enumerate(map_f)}
        n_covered = n_uncovered = n_short_list = 0
        for f_peak in new_clusters["f_start_peak"]:
            idx = map_f_to_idx.get(int(f_peak))
            if idx is None:
                n_uncovered += 1
                continue
            n_covered += 1
            amap = (w1 * map_st1[idx].astype(np.float32)
                    + w2 * map_st2[idx].astype(np.float32)
                    + w3 * map_ss[idx].astype(np.float32))
            fr = full_row_hits(amap, threshold=off_ceiling)
            if fr["in_short_list"]:
                n_short_list += 1

        rows.append(dict(
            cad_idx=cad_idx, label=label,
            n_old=n_old, n_new=n_new,
            n_covered=n_covered, n_uncovered=n_uncovered,
            coverage_frac=n_covered / n_new,
            n_short_list_covered=n_short_list,
            new_thresh_below_map_thresh=new_thresh < map_thresh,
        ))

    df = pd.DataFrame(rows).sort_values("cad_idx").reset_index(drop=True)
    n = len(df)

    print(f"\n{'=' * 78}\nSHORT-LIST REVIEW LOAD under stage-1 = min(far_thresh, off_ceiling)"
          f"\n{n} cadences\n{'=' * 78}")

    gapped = df.get("new_thresh_below_map_thresh", pd.Series(dtype=bool)).fillna(False)
    print(f"\nCadences where the new threshold falls below the saved-map cutoff "
          f"(q95): {int(gapped.sum())} / {n}")
    if gapped.any():
        cov = df.loc[gapped, "coverage_frac"]
        print(f"  Among those, map coverage of new clusters: "
              f"median {cov.median():.1%}, min {cov.min():.1%}")
        print("  Uncovered clusters cannot be scored for short-list membership "
              "offline — they require a fresh forward pass (HDD access) to settle.")

    total_new = df["n_new"].sum()
    total_old = df["n_old"].sum()
    total_covered = df["n_covered"].sum()
    total_uncovered = df["n_uncovered"].sum()
    total_short = df["n_short_list_covered"].sum()

    print(f"\nClusters:  old {total_old:,}  ->  new {total_new:,}  "
          f"(x{total_new / max(total_old, 1):.2f})")
    print(f"  covered by saved maps:   {total_covered:,} ({100 * total_covered / max(total_new, 1):.1f}%)")
    print(f"  NOT covered (unknown):   {total_uncovered:,} ({100 * total_uncovered / max(total_new, 1):.1f}%)")

    print(f"\nOf the covered new clusters, {total_short:,} "
          f"({100 * total_short / max(total_covered, 1):.1f}%) pass full_row_hits "
          f"and would enter the short list.")
    print("This is a LOWER BOUND on the true increase — the uncovered clusters "
          "are unmeasured, not absent.")

    print("\nHeaviest new short-list load per cadence:")
    cols = ["cad_idx", "label", "n_new", "n_covered", "n_uncovered", "n_short_list_covered"]
    print(df.nlargest(12, "n_short_list_covered")[cols].to_string(index=False))

    out = args.out_csv or args.map_dir.parent / "shortlist_review_load.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
