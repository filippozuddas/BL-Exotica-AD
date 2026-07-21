"""
Build cadence manifest files from a directory of GBT HDF5 observations.

Scans one or more directories recursively for .h5 files, groups them into
6-observation ON/OFF cadences, validates the cadence structure, and writes
one manifest file per product type (0000, 0001, 0002) — one cadence per line,
space-separated absolute paths sorted chronologically by timestamp.

GBT GUPPI filename format (both with and without node prefix):
    blc<node>_guppi_<MJD>_<secs>_<subsecs>_<target>_(ON|OFF)_<obs>.<product>.h5
    guppi_<MJD>_<secs>_<subsecs>_<target>_(ON|OFF)_<obs>.<product>.h5

Cadence grouping key: target + MJD-day + product suffix.
Sort within cadence: by the seconds field in the filename.

Usage:
    PYTHONPATH=. python scripts/build_cadence_manifest.py \\
        --scan /path/to/data/ \\
        --output data/raw/ \\
        --product 0000

    # Multiple directories, all products
    PYTHONPATH=. python scripts/build_cadence_manifest.py \\
        --scan /data/gbt/ /data/archive/ \\
        --output data/raw/ \\
        --product all

    # Just print what's found, don't write
    PYTHONPATH=. python scripts/build_cadence_manifest.py \\
        --scan /data/gbt/ --list-only
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GBT filename patterns
# Ordered from most specific to most generic — first match wins.
# ---------------------------------------------------------------------------
_TARGET_PATTERNS = [
    # TIC catalogue (most common in Exotica): TIC368536386
    re.compile(r"(TIC\d+)_(ON|OFF)", re.IGNORECASE),
    # HIP / GJ stellar catalogue
    re.compile(r"(HIP\d+|GJ\d+[A-Za-z]?)_(ON|OFF)", re.IGNORECASE),
    # Generic alphanumeric target
    re.compile(r"([A-Za-z0-9]+)_(ON|OFF)(?:_|\.)"),
]

# GUPPI timestamp: _<5-digit-MJD>_<secs>_
_TIMESTAMP_RE = re.compile(r"_(\d{5})_(\d+)_")

# Product suffix: the last numeric field before .h5 / .hdf5
# e.g. "0001.0000.h5" → product "0000"
_PRODUCT_RE = re.compile(r"\.(\d{4})\.(?:h5|hdf5)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CadenceInfo:
    target: str
    mjd: str
    product: str
    files: List[Path] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.files) == 6

    @property
    def obs_pattern(self) -> List[str]:
        return [_parse_file(f)["obs_type"] for f in self.files]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_file(filepath: Path) -> Optional[Dict]:
    """
    Parse a GBT GUPPI filename into metadata fields.
    Returns None if the file doesn't look like a GUPPI observation.
    """
    name = filepath.name

    product_m = _PRODUCT_RE.search(name)
    if product_m is None:
        return None
    product = product_m.group(1)          # "0000", "0001", "0002"

    target = obs_type = None
    for pat in _TARGET_PATTERNS:
        m = pat.search(name)
        if m:
            target = m.group(1)
            obs_type = m.group(2).upper()
            break
    if target is None:
        return None

    ts_m = _TIMESTAMP_RE.search(name)
    mjd  = ts_m.group(1) if ts_m else "00000"
    secs = int(ts_m.group(2)) if ts_m else 0

    return {
        "target":   target,
        "obs_type": obs_type,
        "mjd":      mjd,
        "secs":     secs,
        "product":  product,
        "path":     filepath,
    }


# ---------------------------------------------------------------------------
# Grouping and validation
# ---------------------------------------------------------------------------

def _group_into_cadences(
    files: List[Path],
    products: List[str],
) -> Dict[str, List[CadenceInfo]]:
    """
    Group parsed files into cadences by (target, MJD-day, product).

    Returns a dict keyed by product suffix, each value a list of CadenceInfo.
    Only complete 6-file cadences with the expected ON/OFF/ON/OFF/ON/OFF
    pattern are included.
    """
    parsed = []
    skipped = 0
    for f in files:
        info = _parse_file(f)
        if info is None:
            skipped += 1
            continue
        if products and info["product"] not in products:
            continue
        parsed.append(info)

    if skipped:
        print(f"  Skipped {skipped} files that don't match GUPPI naming pattern")

    # Group: one bucket per (target, mjd, product)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for info in parsed:
        key = f"{info['target']}_{info['mjd']}_{info['product']}"
        buckets[key].append(info)

    # Sort each bucket chronologically and validate
    cadences_by_product: Dict[str, List[CadenceInfo]] = defaultdict(list)
    n_incomplete = 0
    n_wrong_pattern = 0
    seen_filesets: set = set()
    n_dup = 0

    for key, infos in buckets.items():
        infos.sort(key=lambda x: x["secs"])

        if len(infos) != 6:
            n_incomplete += 1
            continue

        pattern = [i["obs_type"] for i in infos]
        if pattern != ["ON", "OFF", "ON", "OFF", "ON", "OFF"]:
            n_wrong_pattern += 1
            continue

        # Deduplicate by file basename set (same observation under different dirs)
        fileset = tuple(sorted(i["path"].name for i in infos))
        if fileset in seen_filesets:
            n_dup += 1
            continue
        seen_filesets.add(fileset)

        cad = CadenceInfo(
            target=infos[0]["target"],
            mjd=infos[0]["mjd"],
            product=infos[0]["product"],
            files=[i["path"] for i in infos],
        )
        cadences_by_product[cad.product].append(cad)

    if n_incomplete:
        print(f"  Skipped {n_incomplete} groups with ≠6 files")
    if n_wrong_pattern:
        print(f"  Skipped {n_wrong_pattern} groups with wrong ON/OFF pattern")
    if n_dup:
        print(f"  Deduplicated {n_dup} cadences with identical file sets")

    return dict(cadences_by_product)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_PRODUCT_LABELS = {
    "0000": "Fine-freq  (0000) — narrowband  ~3 Hz/chan",
    "0001": "High-time  (0001) — broadband transients",
    "0002": "Wideband   (0002) — ~2.86 kHz/chan",
}


def _print_summary(cadences_by_product: Dict[str, List[CadenceInfo]]) -> None:
    print()
    total = sum(len(v) for v in cadences_by_product.values())
    print(f"Complete cadences found: {total}")
    for prod in sorted(cadences_by_product):
        cads = cadences_by_product[prod]
        label = _PRODUCT_LABELS.get(prod, f"Product {prod}")
        targets = sorted({c.target for c in cads})
        print(f"\n  [{label}]  {len(cads)} cadences")
        print(f"    Targets ({len(targets)}): {', '.join(targets[:10])}"
              + (" ..." if len(targets) > 10 else ""))
        if cads:
            print(f"    MJD range : {min(c.mjd for c in cads)} – {max(c.mjd for c in cads)}")


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def _write_manifest(
    cadences: List[CadenceInfo],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for cad in cadences:
            f.write(" ".join(str(p.resolve()) for p in cad.files) + "\n")
    print(f"  Written: {output_path}  ({len(cadences)} cadences)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Scan GBT HDF5 directories and build cadence manifest files"
    )
    p.add_argument("--scan", "-s", nargs="+", required=True,
                   help="Directories to scan recursively for .h5 files")
    p.add_argument("--output", "-o", default="data/raw/",
                   help="Output directory for manifest files (default: data/raw/)")
    p.add_argument("--product", "-p", default="all",
                   choices=["0000", "0001", "0002", "all"],
                   help="Product type to process (default: all)")
    p.add_argument("--list-only", action="store_true",
                   help="Print summary only, do not write manifest files")
    p.add_argument("--prefix", default="gbt",
                   help="Filename prefix for manifests (default: gbt → gbt_0000_cadences.txt)")
    args = p.parse_args()

    # ------------------------------------------------------------------ scan
    all_files: List[Path] = []
    for directory in args.scan:
        d = Path(directory)
        found = list(d.rglob("*.h5")) + list(d.rglob("*.hdf5"))
        print(f"  {d}: {len(found)} HDF5 files")
        all_files.extend(found)
    print(f"Total HDF5 files: {len(all_files)}")

    # ----------------------------------------------------------- group
    products = [] if args.product == "all" else [args.product]
    cadences_by_product = _group_into_cadences(all_files, products)

    _print_summary(cadences_by_product)

    if args.list_only or not cadences_by_product:
        return

    # ---------------------------------------------------------- write
    print()
    output_dir = Path(args.output)
    for prod, cadences in sorted(cadences_by_product.items()):
        manifest_path = output_dir / f"{args.prefix}_{prod}_cadences.txt"
        _write_manifest(cadences, manifest_path)

    print("\nDone. Update configs/data/gbt_*.yaml to point dataset.cadence_list at these files.")


if __name__ == "__main__":
    main()
