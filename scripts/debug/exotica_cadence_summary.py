"""
Cadence-level summary of the real Exotica `.0000.h5` batch, from headers only
(no blimpy / data load needed -- runs anywhere `data/processed/exotica_0000_headers.csv`
is available, unlike `exotica_dist_analysis.py` which needs the data host).

Cadence extraction logic (2026-07-06, corrected): group files by
(time-cluster, rounded band, **node**) -- node must be part of the key because
distinct node pairs (e.g. blc17/blc21) can cover the *exact same* nominal band
at the *exact same* time with genuinely different data (confirmed by content
diff, not a rounding artifact) -- then split each group into consecutive
6-file chunks (ON/OFF/ON/OFF/ON/OFF), since a single node/band can carry
multiple distinct targets back-to-back within one observing session. Earlier
session-band-only bucketing (no node, no chunking) undercounted cadences by
~2.5x (358 vs 905) by merging multi-target sequences under one dominant ON
target.

Answers:
  1. How many complete (6-file) cadences are present.
  2. Whether the same ON target is observed across multiple cadences.
  3. Whether the same target is observed at different times (MJD days) and/or
     different frequency bands.
  4. The distribution of cadences as a function of covered frequency.
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"pyproject.toml not found above {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from scan_headers import cluster_by_gap, infer_on_off  # noqa: E402

HEADERS_CSV = REPO_ROOT / "data" / "processed" / "exotica_0000_headers.csv"
GAP_THRESHOLD_S = 700


def band_node_key(row):
    return (round(row["f_start_MHz"], 0), round(row["f_stop_MHz"], 0), row["node"])


def region(f_lo):
    if f_lo < 1200: return "L-low (750-1200)"
    if f_lo < 1900: return "L-high (1200-1900)"
    if f_lo < 3200: return "S (1900-3200)"
    if f_lo < 5000: return "C-low (3200-5000)"
    if f_lo < 8000: return "C-high/X-low (5000-8000)"
    if f_lo < 10000: return "X-mid (8000-10000)"
    return "X-high (10000-12400)"


def build_cadences(good):
    """
    Returns (clusters, cadences, n_leftover) where cadences is a list of dicts:
      {session, band, node, on_target, off_targets, mjd_days, files}
    files is the 6 file paths in time order (matches the ON/OFF/ON/OFF/ON/OFF
    sequence -- the order later written into the cadence manifest).
    """
    rows_sorted = good.to_dict("records")
    clusters = cluster_by_gap(rows_sorted, GAP_THRESHOLD_S)

    cadences = []
    n_leftover = 0
    for session_idx, cluster in enumerate(clusters, 1):
        groups = defaultdict(list)
        for r in cluster:
            groups[band_node_key(r)].append(r)
        for key, rows_in_group in groups.items():
            rows_in_group = sorted(rows_in_group, key=lambda x: x["abs_time_s"])
            n = len(rows_in_group)
            n_full = n // 6
            n_leftover += n % 6
            for chunk_i in range(n_full):
                chunk = rows_in_group[chunk_i * 6:(chunk_i + 1) * 6]
                oi = infer_on_off(chunk)
                cadences.append({
                    "session": session_idx, "band": (key[0], key[1]), "node": key[2],
                    "on_target": oi["on_target"], "off_targets": oi["off_targets"],
                    "mjd_days": sorted(set(r["mjd_day"] for r in chunk)),
                    "files": [r["file"] for r in chunk],
                })
    return clusters, cadences, n_leftover


def main():
    df = pd.read_csv(HEADERS_CSV)
    good = df[df["nchans"].notna()].sort_values("abs_time_s").reset_index(drop=True)

    clusters, cadences, n_leftover = build_cadences(good)

    print("=== 1. Complete cadences ===")
    print(f"Real observing sessions (time-clusters): {len(clusters)}")
    print(f"Complete cadences (6-file ON/OFF/ON/OFF/ON/OFF, node+chunk aware): {len(cadences)}")
    print(f"Leftover stray files (don't form a full 6-cycle): {n_leftover}")

    print("\n=== 2 & 3. Target repetition across cadences, bands, days ===")
    target_cadences = defaultdict(list)
    for e in cadences:
        target_cadences[e["on_target"]].append(e)

    rows_target = []
    for t, es in target_cadences.items():
        bands = sorted(set(e["band"] for e in es))
        days = sorted(set(d for e in es for d in e["mjd_days"]))
        rows_target.append({
            "target": t, "n_complete_cadences": len(es),
            "n_distinct_bands": len(bands), "n_distinct_mjd_days": len(days),
        })
    target_df = pd.DataFrame(rows_target).sort_values("n_complete_cadences", ascending=False)

    n_targets = len(target_df)
    multi_cad = (target_df["n_complete_cadences"] > 1).sum()
    multi_band = (target_df["n_distinct_bands"] > 1).sum()
    multi_day = (target_df["n_distinct_mjd_days"] > 1).sum()
    print(f"Distinct ON (Exotica) targets: {n_targets}")
    print(f"  observed in >1 complete cadence : {multi_cad}/{n_targets}")
    print(f"  observed in >1 distinct band    : {multi_band}/{n_targets}")
    print(f"  observed on >1 distinct MJD day : {multi_day}/{n_targets}")
    print("\nPer-target breakdown:")
    print(target_df.to_string(index=False))

    print("\n=== 4. Cadence distribution vs covered frequency ===")
    comp_df = pd.DataFrame([{"band": e["band"]} for e in cadences])
    comp_df["f_lo"] = comp_df["band"].apply(lambda b: b[0])
    by_band = comp_df.groupby("band").size().reset_index(name="n_complete_cadences")
    by_band["f_lo"] = by_band["band"].apply(lambda b: b[0])
    by_band = by_band.sort_values("f_lo")
    by_band["region"] = by_band["f_lo"].apply(region)
    print(by_band[["band", "n_complete_cadences", "region"]].to_string(index=False))

    print("\nBy coarse frequency region:")
    print(by_band.groupby("region")["n_complete_cadences"].agg(["sum", "count"])
          .rename(columns={"sum": "total_cadences", "count": "n_bands"}))

    out_dir = REPO_ROOT / "data" / "processed" / "dist_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(out_dir / "target_cadence_summary.csv", index=False)
    by_band.to_csv(out_dir / "cadences_by_band_region.csv", index=False)

    # Full cadence manifest (session/band/node/target/day + file list) for the
    # downstream stratified train/val/held-out split.
    manifest_rows = [{
        "session": c["session"], "band_lo": c["band"][0], "band_hi": c["band"][1],
        "node": c["node"], "on_target": c["on_target"],
        "off_targets": ";".join(c["off_targets"]),
        "mjd_days": ";".join(str(d) for d in c["mjd_days"]),
        "files": " ".join(c["files"]),
    } for c in cadences]
    pd.DataFrame(manifest_rows).to_csv(out_dir / "all_complete_cadences.csv", index=False)
    print(f"\nFull cadence manifest ({len(manifest_rows)} rows) -> "
          f"{out_dir / 'all_complete_cadences.csv'}")


if __name__ == "__main__":
    main()
