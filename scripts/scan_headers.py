#!/usr/bin/env python3
"""
scan_headers.py — read headers from all .h5 files in a directory tree,
summarise frequency coverage, and group observations into cadences.

Naming convention (BL GBT Exotica):
    {node}_guppi_{mjd_day}_{mjd_sec}_{TARGET}_{scan}.rawspec.{product}.h5
    e.g. blc00_guppi_60055_03921_MESSIER67_0055.rawspec.0000.h5

Cadence pattern (inferred from data):
    ON-OFF-ON-OFF-ON-OFF with the SAME Exotica target (ON) and 3 different
    HIP reference stars (OFF, one per cycle).  No explicit ON/OFF label.

Cadence inference algorithm:
    1. Sort by abs_time = mjd_day * 86400 + mjd_sec.
    2. Split into clusters wherever the time gap > --gap-threshold (default 700 s;
       within-cadence gaps are ~325 s, between-cadence ≥ ~855 s in the data).
    3. Within each cluster the most-frequent target = ON (Exotica source);
       targets appearing once each = OFF reference stars.

Usage:
    python scripts/scan_headers.py /path/to/exotica/
    python scripts/scan_headers.py /path/to/exotica/ --product 0002
    python scripts/scan_headers.py /path/to/exotica/ --out headers.csv
    python scripts/scan_headers.py /path/to/exotica/ --gap-threshold 700
"""

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> dict:
    """
    Parse:  {node}_guppi_{mjd_day}_{mjd_sec}_{TARGET...}_{scan}.rawspec.{product}.h5

    Returns a dict with: node, backend, mjd_day, mjd_sec, abs_time_s,
    target, scan_num, product.  On failure returns {"file":..., "parse_error":...}.
    """
    name = path.name

    # Split off .rawspec.{product}.h5
    if ".rawspec." not in name or not name.endswith(".h5"):
        # Fall back: try splitting on last two dots (some older files lack .rawspec.)
        parts = name.rsplit(".", 2)
        if len(parts) == 3 and parts[2] == "h5" and parts[1].isdigit():
            core, product = parts[0], parts[1]
        else:
            return {"file": str(path), "parse_error": f"unrecognised filename: {name}"}
    else:
        idx = name.index(".rawspec.")
        core = name[:idx]                     # blc00_guppi_60055_03921_MESSIER67_0055
        product = name[idx + len(".rawspec.") : -len(".h5")]  # 0000

    fields = core.split("_")
    # Expected minimum: node, guppi, mjd_day, mjd_sec, target, scan  →  6 fields
    if len(fields) < 6:
        return {"file": str(path), "parse_error": f"too few fields ({len(fields)}): {core}"}

    node    = fields[0]
    backend = fields[1]
    try:
        mjd_day = int(fields[2])
        mjd_sec = int(fields[3])
    except ValueError:
        return {"file": str(path), "parse_error": f"non-numeric MJD: {fields[2:4]}"}

    scan_num = fields[-1]                    # last field is always the scan counter
    target   = "_".join(fields[4:-1])        # everything between mjd_sec and scan_num

    return {
        "file":       str(path),
        "node":       node,
        "backend":    backend,
        "mjd_day":    mjd_day,
        "mjd_sec":    mjd_sec,
        "abs_time_s": mjd_day * 86400 + mjd_sec,
        "target":     target,
        "scan_num":   scan_num,
        "product":    product,
    }


# ---------------------------------------------------------------------------
# Header reading (no data load) — h5py first, blimpy as fallback
# ---------------------------------------------------------------------------

def read_header_h5py(path: Path) -> dict:
    import h5py
    with h5py.File(str(path), "r") as f:
        # BL HDF5 files store header attrs on the "data" dataset
        if "data" in f:
            h     = dict(f["data"].attrs)
            shape = f["data"].shape
        else:
            # fallback: root attrs + first dataset shape
            h     = dict(f.attrs)
            ds    = next(iter(f.values()))
            shape = ds.shape
    ntime  = int(shape[0])
    nifs   = int(shape[1]) if len(shape) == 3 else 1
    nchans = int(shape[-1])
    fch1   = float(h.get("fch1",  h.get("fch1_MHz", float("nan"))))
    foff   = float(h.get("foff",  h.get("foff_MHz", float("nan"))))
    tsamp  = float(h.get("tsamp", float("nan")))
    f_lo   = min(fch1, fch1 + foff * nchans)
    f_hi   = max(fch1, fch1 + foff * nchans)
    return {
        "ntime": ntime, "nchans": nchans, "nifs": nifs,
        "fch1_MHz":    round(fch1, 6),
        "foff_Hz":     round(foff * 1e6, 4),
        "f_start_MHz": round(f_lo, 4),
        "f_stop_MHz":  round(f_hi, 4),
        "bw_MHz":      round(abs(foff * nchans), 4),
        "tsamp_s":     round(tsamp, 9),
        "duration_s":  round(tsamp * ntime, 2),
    }


def read_header_blimpy(path: Path) -> dict:
    import blimpy
    wf     = blimpy.Waterfall(str(path), load_data=False)
    h      = wf.header
    nchans = int(h.get("nchans", 0))
    fch1   = float(h.get("fch1",  float("nan")))
    foff   = float(h.get("foff",  float("nan")))
    tsamp  = float(h.get("tsamp", float("nan")))
    nifs   = int(h.get("nifs", 1))
    ns     = h.get("nsamples", None)
    ntime  = int(ns[0] if hasattr(ns, "__len__") else ns) if ns is not None else 0
    f_lo   = min(fch1, fch1 + foff * nchans)
    f_hi   = max(fch1, fch1 + foff * nchans)
    return {
        "ntime": ntime, "nchans": nchans, "nifs": nifs,
        "fch1_MHz":    round(fch1, 6),
        "foff_Hz":     round(foff * 1e6, 4),
        "f_start_MHz": round(f_lo, 4),
        "f_stop_MHz":  round(f_hi, 4),
        "bw_MHz":      round(abs(foff * nchans), 4),
        "tsamp_s":     round(tsamp, 9),
        "duration_s":  round(tsamp * ntime, 2) if ntime else float("nan"),
    }


def read_header(path: Path) -> dict:
    """h5py first (fast), blimpy as fallback."""
    try:
        return read_header_h5py(path)
    except Exception:
        pass
    try:
        return read_header_blimpy(path)
    except Exception as e:
        return {"error": str(e)}


def _process_one(path_str: str) -> dict:
    """Top-level worker function (must be picklable for multiprocessing)."""
    path   = Path(path_str)
    parsed = parse_filename(path)
    header = read_header(path)
    row    = dict(parsed)
    if "error" not in header:
        row.update(header)
    else:
        row["header_error"] = header["error"]
    return row


# ---------------------------------------------------------------------------
# Cadence clustering and ON/OFF inference
# ---------------------------------------------------------------------------

def cluster_by_gap(rows: list, gap_s: int) -> list:
    """Split time-sorted rows into clusters wherever consecutive gap > gap_s."""
    if not rows:
        return []
    clusters, current = [], [rows[0]]
    for row in rows[1:]:
        if row["abs_time_s"] - current[-1]["abs_time_s"] > gap_s:
            clusters.append(current)
            current = [row]
        else:
            current.append(row)
    clusters.append(current)
    return clusters


def infer_on_off(cluster: list) -> dict:
    """
    Within a cluster, the target appearing most often is the Exotica (ON) source.
    Targets appearing once each are the OFF reference stars.
    Returns a dict with: on_target, off_targets (list, time-ordered), n_on, n_off,
    pattern (human-readable string).
    """
    counts    = Counter(r["target"] for r in cluster)
    max_count = counts.most_common(1)[0][1]

    # All targets tied at 1 → can't distinguish ON from OFF
    if max_count == 1:
        return {
            "on_target":  None,
            "off_targets": list(counts.keys()),
            "n_on": 0, "n_off": len(cluster),
            "pattern": "ambiguous (all targets appear once)",
        }

    on_cands = [t for t, c in counts.items() if c == max_count]
    # If somehow two targets tie at max_count > 1, pick the first in time
    on_target = next(r["target"] for r in cluster if r["target"] in on_cands)

    off_targets_ordered = []
    seen = set()
    for r in cluster:
        t = r["target"]
        if t != on_target and t not in seen:
            off_targets_ordered.append(t)
            seen.add(t)

    n_on  = counts[on_target]
    n_off = len(cluster) - n_on

    if not off_targets_ordered:
        pattern = f"ON-only ({n_on} obs, no OFF visible)"
    else:
        pattern = (f"ON×{n_on} / OFF×{n_off} "
                   f"({'fixed' if len(off_targets_ordered)==1 else str(len(off_targets_ordered))+' different'} OFF source{'s' if len(off_targets_ordered)>1 else ''})")

    return {
        "on_target":   on_target,
        "off_targets": off_targets_ordered,
        "n_on": n_on, "n_off": n_off,
        "pattern": pattern,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Root directory to search")
    parser.add_argument("--product", choices=["0000", "0001", "0002"],
                        help="Filter to one product suffix")
    parser.add_argument("--gap-threshold", type=int, default=700,
                        help="Seconds gap that starts a new cadence (default: 700)")
    parser.add_argument("--out", help="Save per-file CSV to this path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-file sequence inside each cadence")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes for header reading (default: 8)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    files = sorted(root.rglob("*.h5"))
    if args.product:
        files = [f for f in files if f".{args.product}.h5" in f.name]

    if not files:
        suffix = f" with product {args.product}" if args.product else ""
        print(f"No .h5 files found under {root}{suffix}")
        sys.exit(0)

    n_workers = min(args.workers, len(files))
    print(f"Found {len(files)} file(s). Reading headers with {n_workers} workers...\n")

    t0 = time.time()
    path_strs = [str(p) for p in files]
    with mp.Pool(processes=n_workers) as pool:
        all_rows = pool.map(_process_one, path_strs, chunksize=max(1, len(files) // (n_workers * 4)))
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed/len(files)*1000:.1f} ms/file)\n")

    for r in all_rows:
        if "parse_error" in r:
            print(f"  [PARSE ERROR]  {Path(r['file']).name}: {r['parse_error']}")
        if "header_error" in r:
            print(f"  [HEADER ERROR] {Path(r['file']).name}: {r['header_error']}")

    good = [r for r in all_rows if "nchans" in r and "abs_time_s" in r]
    bad  = [r for r in all_rows if "nchans" not in r or "abs_time_s" not in r]

    # -----------------------------------------------------------------------
    # Per-file table (sorted by product then time)
    # -----------------------------------------------------------------------
    print("=" * 150)
    hdr_cols = ["product", "mjd_day", "scan_num", "target",
                "f_start_MHz", "f_stop_MHz", "bw_MHz", "nchans", "ntime", "duration_s"]
    print(f"  {'FILE':<55} " + "  ".join(f"{c:<13}" for c in hdr_cols))
    print("-" * 150)
    for r in sorted(good, key=lambda x: (x.get("product",""), x.get("abs_time_s", 0))):
        fname = Path(r["file"]).name
        vals  = [str(r.get(c, "")) for c in hdr_cols]
        print(f"  {fname:<55} " + "  ".join(f"{v:<13}" for v in vals))

    # -----------------------------------------------------------------------
    # Cadence grouping (per product)
    # -----------------------------------------------------------------------
    products = sorted(set(r.get("product","?") for r in good))

    print("\n" + "=" * 100)
    print(f"CADENCE GROUPING  (gap threshold = {args.gap_threshold} s)")
    print("=" * 100)

    for prod in products:
        prod_rows = sorted(
            [r for r in good if r.get("product") == prod],
            key=lambda x: x["abs_time_s"],
        )
        clusters = cluster_by_gap(prod_rows, args.gap_threshold)

        print(f"\n  Product .{prod}  —  {len(prod_rows)} files  →  {len(clusters)} cadence(s)\n")
        print(f"  {'#':>3}  {'mjd_day':>7}  {'t_start':>7}  {'t_end':>7}  "
              f"{'span':>5}  {'n':>4}  {'ON target':<25}  {'OFF sources':<50}  pattern")
        print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*7}  "
              f"{'─'*5}  {'─'*4}  {'─'*25}  {'─'*50}  {'─'*45}")

        for idx, cluster in enumerate(clusters, 1):
            times   = [r["abs_time_s"] for r in cluster]
            days    = sorted(set(r["mjd_day"] for r in cluster))
            day_str = "/".join(str(d) for d in days)
            oi      = infer_on_off(cluster)

            on_str  = oi["on_target"] or "?"
            off_str = ", ".join(oi["off_targets"]) if oi["off_targets"] else "—"
            if len(off_str) > 48:
                off_str = off_str[:45] + "..."

            print(f"  {idx:>3}  {day_str:>7}  {min(times)%86400:>7}  {max(times)%86400:>7}  "
                  f"{max(times)-min(times):>5.0f}s  {len(cluster):>4}  "
                  f"{on_str:<25}  {off_str:<50}  {oi['pattern']}")

            if args.verbose or oi["on_target"] is None:
                for r in cluster:
                    role = "ON " if r["target"] == oi["on_target"] else "OFF"
                    print(f"       mjd_sec={r['mjd_sec']:>6}  scan={r['scan_num']}  "
                          f"[{role}]  {r['target']}")

    # -----------------------------------------------------------------------
    # Exotica target inventory
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("EXOTICA TARGETS OBSERVED")
    print("=" * 100)
    for prod in products:
        prod_rows = sorted(
            [r for r in good if r.get("product") == prod],
            key=lambda x: x["abs_time_s"],
        )
        clusters = cluster_by_gap(prod_rows, args.gap_threshold)
        on_targets = {}  # target → list of (mjd_day, n_cadences)
        for cluster in clusters:
            oi = infer_on_off(cluster)
            t  = oi["on_target"]
            if t:
                days = sorted(set(r["mjd_day"] for r in cluster))
                on_targets.setdefault(t, []).extend(days)

        print(f"\n  Product .{prod}  ({len(on_targets)} unique ON targets):")
        for tgt, days in sorted(on_targets.items()):
            day_counts = Counter(days)
            day_str    = ", ".join(f"MJD{d}×{n}" if n > 1 else f"MJD{d}"
                                   for d, n in sorted(day_counts.items()))
            print(f"    {tgt:<30}  {day_str}")

    # -----------------------------------------------------------------------
    # Frequency coverage summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FREQUENCY COVERAGE")
    print("=" * 100)
    for prod in products:
        prod_rows = [r for r in good if r.get("product") == prod]
        f_starts  = [r["f_start_MHz"] for r in prod_rows]
        f_stops   = [r["f_stop_MHz"]  for r in prod_rows]
        bws       = [r["bw_MHz"]      for r in prod_rows]
        durations = [r["duration_s"]  for r in prod_rows
                     if not math.isnan(r.get("duration_s", float("nan")))]

        print(f"\n  Product .{prod}  ({len(prod_rows)} files)")
        if f_starts:
            all_f = sorted(set(round(v, 1) for v in f_starts + f_stops))
            print(f"    global range : {min(f_starts):.2f} – {max(f_stops):.2f} MHz")
            print(f"    bandwidth    : min={min(bws):.2f}  max={max(bws):.2f}  "
                  f"median={sorted(bws)[len(bws)//2]:.2f} MHz per file")
        if durations:
            print(f"    duration     : min={min(durations):.1f}  max={max(durations):.1f}  "
                  f"median={sorted(durations)[len(durations)//2]:.1f} s")

        bands = sorted(set(
            (round(r["f_start_MHz"], 0), round(r["f_stop_MHz"], 0))
            for r in prod_rows
        ))
        print(f"    unique bands ({len(bands)}):")
        for lo, hi in bands:
            n = sum(1 for r in prod_rows
                    if abs(r["f_start_MHz"] - lo) < 1 and abs(r["f_stop_MHz"] - hi) < 1)
            print(f"      {lo:.0f} – {hi:.0f} MHz   ({n} file{'s' if n>1 else ''})")

    if bad:
        print(f"\n  Files with errors: {len(bad)}")
        for r in bad:
            print(f"    {r['file']}")

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    if args.out:
        csv_cols = [
            "file", "product", "node", "mjd_day", "mjd_sec", "abs_time_s",
            "scan_num", "target",
            "f_start_MHz", "f_stop_MHz", "bw_MHz", "fch1_MHz", "foff_Hz",
            "nchans", "ntime", "nifs", "tsamp_s", "duration_s",
        ]
        out = Path(args.out)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(good)
        print(f"\n  CSV saved → {out}  ({len(good)} rows)")


if __name__ == "__main__":
    main()
