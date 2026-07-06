#!/usr/bin/env python3
"""
build_gbt_cadence_manifest.py — split the real Exotica `.0000.h5` cadence
census into train / val / held-out manifests for `configs/data/gbt_fine.yaml`.

Decisions (from the 2026-07-06 /grill-me session, see memory
`exotica_0000_dist_analysis`):

  - Scope: `.0000.fil` only.
  - Real GBT data fully replaces SRT/synthetic as the training source.
  - 3 pools, not 4: train, val, held-out. held-out serves BOTH the Phase 2
    injection-recovery SNR sweep and the Phase 3 real search -- `inject_recover.py`
    only reads cadences to probe backgrounds / inject synthetic signals
    on-the-fly, it never mutates files, so one pool can serve both purposes.
  - held-out is mandatory, not "train on everything": training the AE and
    searching the same cadences risks it partially reconstructing a rare real
    signal well, suppressing its own anomaly score exactly where it matters.
  - Split is stratified by band (>=1 cadence/band/pool) AND by target
    (>=1 cadence/target in train and in held-out) -- a plain global shuffle
    risks a thin band or a target (only 22 total) vanishing from held-out,
    meaning it would never actually be searched.
  - Day-level diversity within a band is NOT constrained a priori -- checked
    post-hoc instead (reported as a warning, not enforced) since most bands
    span several MJD days and enforcing it adds allocation complexity for a
    rare failure mode (only 2/61 bands are single-day regardless of split).
  - Per-band cadence counts are NOT reweighted to equalize snippet volume --
    band prevalence in training is accepted as-is.

Input: data/processed/dist_analysis/all_complete_cadences.csv (produced by
`scripts/debug/exotica_cadence_summary.py`), one row per complete 6-file
cadence with columns: session, band_lo, band_hi, node, on_target,
off_targets, mjd_days, files (space-separated paths, already time-ordered).

Output: data/raw/gbt_0000_{train,val,heldout}_cadences.txt -- one cadence
per line, 6 space-separated file paths (same format as the old
`srt_0000_cadences.txt`), directly usable as a `preprocess_cache.py`
pre-split manifest.

Usage:
    python3 scripts/build_gbt_cadence_manifest.py
    python3 scripts/build_gbt_cadence_manifest.py --train-frac 0.5 --val-frac 0.1 --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"pyproject.toml not found above {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "dist_analysis" / "all_complete_cadences.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw"

POOLS = ("train", "val", "heldout")


def allocate(n: int, fracs: tuple, min_required: tuple) -> list:
    """
    Split n items into len(fracs) integer buckets summing to n, using
    Hamilton apportionment (floor + largest-remainder), with a hard floor of
    min_required[i] on bucket i wherever n allows it. val (index 1) is the
    first bucket shaved down if the minimums don't fit -- it's the least
    costly pool to shrink for a thin band/target (see module docstring).
    """
    if n == 0:
        return [0] * len(fracs)

    raw = [n * f for f in fracs]
    counts = [int(np.floor(x)) for x in raw]

    for i, m in enumerate(min_required):
        if counts[i] < m:
            counts[i] = m

    total = sum(counts)
    if total > n:
        deficit = total - n
        reduce_val = min(counts[1], deficit)
        counts[1] -= reduce_val
        deficit -= reduce_val
        while deficit > 0:
            # shave whichever non-val bucket is furthest above its minimum
            candidates = [i for i in range(len(counts)) if i != 1 and counts[i] > min_required[i]]
            if not candidates:
                break  # can't satisfy all minimums simultaneously (shouldn't happen at our n)
            idx = max(candidates, key=lambda i: counts[i] - min_required[i])
            counts[idx] -= 1
            deficit -= 1
    else:
        remainder = n - total
        frac_left = [raw[i] - counts[i] for i in range(len(fracs))]
        order = sorted(range(len(fracs)), key=lambda i: (-frac_left[i], i))
        i = 0
        while remainder > 0:
            counts[order[i % len(fracs)]] += 1
            remainder -= 1
            i += 1

    assert sum(counts) == n
    return counts


def split_by_band(df: pd.DataFrame, fracs: tuple, rng: np.random.Generator) -> pd.Series:
    """
    Assign each cadence row to a pool, stratified by band: shuffle within
    each band group, allocate counts via `allocate` (train and heldout
    floored to >=1 whenever the band has >=2 cadences), assign shuffled rows
    train-first / heldout-second / val-last so the guaranteed minimums land
    on genuinely random cadences, not a biased subset.
    """
    pool = pd.Series(index=df.index, dtype=object)
    for band, group in df.groupby(["band_lo", "band_hi"]):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        min_train = 1 if n >= 1 else 0
        min_heldout = 1 if n >= 2 else 0
        n_train, n_val, n_heldout = allocate(n, fracs, (min_train, 0, min_heldout))
        pool.loc[idx[:n_train]] = "train"
        pool.loc[idx[n_train:n_train + n_heldout]] = "heldout"
        pool.loc[idx[n_train + n_heldout:]] = "val"
    return pool


def repair_target_coverage(df: pd.DataFrame, pool: pd.Series, rng: np.random.Generator) -> list:
    """
    Band-stratified allocation doesn't explicitly guarantee every target
    appears in train and in held-out (only every band does). With 22 targets
    each having >=19 cadences spread over many bands this is expected to
    hold anyway -- this pass verifies it and, if a target is missing from a
    required pool, moves one of its cadences there (stealing from val first,
    the pool anywhere else in this script also treats as least costly).
    Returns a list of human-readable repair log lines.
    """
    log = []
    for required_pool in ("train", "heldout"):
        for target, group in df.groupby("on_target"):
            idx = group.index
            if (pool.loc[idx] == required_pool).any():
                continue
            # steal from val first, else from the other non-required pool
            donor_order = ["val"] + [p for p in POOLS if p not in (required_pool, "val")]
            donor_idx = None
            for donor in donor_order:
                candidates = idx[pool.loc[idx] == donor]
                if len(candidates) > 0:
                    donor_idx = rng.choice(candidates)
                    break
            if donor_idx is None:
                log.append(f"  WARNING: target {target!r} has no cadence to donate to {required_pool} "
                           f"(only {len(idx)} total, all already needed elsewhere)")
                continue
            pool.loc[donor_idx] = required_pool
            log.append(f"  moved 1 cadence of target {target!r} from {donor} -> {required_pool} "
                       f"(row {donor_idx})")
    return log


def check_day_diversity(df: pd.DataFrame, pool: pd.Series) -> list:
    """Post-hoc, informational only (not enforced) -- report band/pool combos backed by a single MJD day."""
    warnings = []
    tmp = df.copy()
    tmp["pool"] = pool
    for (band_lo, band_hi, p), group in tmp.groupby(["band_lo", "band_hi", "pool"]):
        days = set()
        for v in group["mjd_days"]:
            days.update(str(v).split(";"))
        if len(days) == 1:
            warnings.append(f"  band ({band_lo:.0f}-{band_hi:.0f} MHz) / pool={p}: "
                            f"{len(group)} cadence(s), all from MJD day {next(iter(days))}")
    return warnings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--train-frac", type=float, default=0.50)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    heldout_frac = 1.0 - args.train_frac - args.val_frac
    if heldout_frac <= 0:
        raise ValueError(f"train-frac + val-frac must be < 1 (got heldout-frac={heldout_frac:.3f})")
    fracs = (args.train_frac, args.val_frac, heldout_frac)
    print(f"Target proportions: train={fracs[0]:.0%} val={fracs[1]:.0%} heldout={fracs[2]:.0%}")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} complete cadences from {args.input}")

    rng = np.random.default_rng(args.seed)

    pool = split_by_band(df, fracs, rng)

    print("\n=== Band-stratified split (before target-coverage repair) ===")
    print(pool.value_counts())

    repair_log = repair_target_coverage(df, pool, rng)
    if repair_log:
        print(f"\n=== Target-coverage repairs ({len(repair_log)}) ===")
        print("\n".join(repair_log))
    else:
        print("\nNo target-coverage repairs needed -- every target already had >=1 cadence in train and heldout.")

    print("\n=== Final pool sizes ===")
    print(pool.value_counts())

    print("\n=== Per-band coverage (min cadences across the 3 pools, should be >=1 except thin bands) ===")
    tmp = df.copy()
    tmp["pool"] = pool
    band_cov = tmp.groupby(["band_lo", "band_hi", "pool"]).size().unstack(fill_value=0)
    band_cov = band_cov.reindex(columns=list(POOLS), fill_value=0)
    zero_bands = band_cov[(band_cov == 0).any(axis=1)]
    if len(zero_bands):
        print(f"{len(zero_bands)} band(s) with 0 cadences in at least one pool:")
        print(zero_bands)
    else:
        print("Every band has >=1 cadence in every pool.")

    print("\n=== Per-target coverage (train / heldout) ===")
    target_cov = tmp.groupby(["on_target", "pool"]).size().unstack(fill_value=0)
    target_cov = target_cov.reindex(columns=list(POOLS), fill_value=0)
    print(target_cov[["train", "heldout"]].sort_values("heldout"))

    day_warnings = check_day_diversity(df, pool)
    print(f"\n=== Day-diversity check (informational, not enforced): {len(day_warnings)} single-day band/pool combos ===")
    if day_warnings:
        print("\n".join(day_warnings))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pool_name in POOLS:
        out_path = args.output_dir / f"gbt_0000_{pool_name}_cadences.txt"
        rows = df.loc[pool[pool == pool_name].index]
        with open(out_path, "w") as f:
            for files in rows["files"]:
                f.write(files.strip() + "\n")
        print(f"\nWrote {len(rows)} cadences -> {out_path}")


if __name__ == "__main__":
    main()
