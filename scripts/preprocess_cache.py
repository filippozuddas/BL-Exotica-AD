"""
Offline background extractor — BL Exotica Autoencoder.

Scans a cadence manifest, splits cadences into train / inject-recovery pools,
extracts RAW snippets from the train pool, and saves them as per-split .npy
files for memory-mapped training.  The inject-recovery cadence list is written
to a separate text file so Phase 2 (injection-recovery test) can use those
cadences without any overlap with training data.

Output directory contains:
  train.npy  — (N_train, n_obs, tchans_per_obs, fchans) float32 RAW
  val.npy    — (N_val,   n_obs, tchans_per_obs, fchans) float32 RAW
  meta.json  — extraction metadata

Normalization (bandpass_correct + core_transform) happens in
CachedDataset.__getitem__ so preprocessing hyperparameters can be changed
without re-extracting the cache.  The .npy format enables np.load(mmap_mode='r')
so DataLoader workers share physical memory pages instead of duplicating.

Two-pass extraction: pass 1 validates all file headers and computes exact
snippet counts; pass 2 allocates the .npy via memmap and writes directly to
disk, keeping RAM usage bounded to one cadence buffer at a time.

Usage:
    PYTHONPATH=. python scripts/preprocess_cache.py \\
        --config    configs/training/srt_real.yaml \\
        --output    data/processed/cache_gbt_fine \\
        --snippets-per-cadence 18000 \\
        --max-snippets 1100000 \\
        --train-fraction 0.7 \\
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import h5py
try:
    import hdf5plugin  # noqa: F401 — registers bitshuffle/LZ4 filters for BL HDF5 files
except ImportError:
    pass
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _read_header(path: Path):
    """Return (ntime, nchans) from HDF5 header without loading data."""
    with h5py.File(str(path), "r") as f:
        sh = f["data"].shape   # (ntime, nif, nchans) or (ntime, nchans)
        ntime = sh[0]
        nchans = sh[-1]
    return ntime, nchans


def _validate_cadence(cadence_paths: list[Path]) -> int | None:
    """Check all files in a cadence are readable. Return nchans or None."""
    nchans = None
    for path in cadence_paths:
        try:
            _, nc = _read_header(path)
        except Exception as e:
            print(f"  skip cadence (corrupt {path.name}): {e}")
            return None
        if nchans is None:
            nchans = nc
        elif nc != nchans:
            print(f"  skip cadence (nchans mismatch: {path.name} has {nc}, expected {nchans})")
            return None
    return nchans


def _extract_cadence_snippets(
    cadence_paths: list[Path],
    indices: np.ndarray,
    fchans: int,
    tchans_per_obs: int,
) -> np.ndarray:
    """
    Extract snippets at the given frequency indices from all obs in a cadence.

    Returns (n_snippets, n_obs, tchans_per_obs, fchans) float32 RAW array.
    Each file is opened once and all snippets are read in a single pass.
    """
    n_obs = len(cadence_paths)
    n_snip = len(indices)
    out = np.empty((n_snip, n_obs, tchans_per_obs, fchans), dtype=np.float32)

    for oi, path in enumerate(cadence_paths):
        with h5py.File(str(path), "r", rdcc_nbytes=256 * 1024 * 1024) as hf:
            dset = hf["data"]
            three_d = dset.ndim == 3
            for si, idx in enumerate(indices):
                start = int(idx) * fchans
                if three_d:
                    out[si, oi] = dset[:tchans_per_obs, 0, start : start + fchans]
                else:
                    out[si, oi] = dset[:tchans_per_obs, start : start + fchans]

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract and cache background snippets")
    p.add_argument("--config", type=Path, default=Path("configs/training/srt_real.yaml"))
    p.add_argument("--output", type=Path, default=Path("data/processed/cache_gbt_fine"),
                   help="Output directory (will contain train.npy, val.npy, meta.json)")
    p.add_argument("--snippets-per-cadence", type=int, default=18000,
                   help="Max snippets sampled per cadence (controls diversity)")
    p.add_argument("--max-snippets", type=int, default=1_100_000,
                   help="Global cap on total train snippets")
    p.add_argument("--train-fraction", type=float, default=0.7,
                   help="Fraction of cadences used for training; rest → inject-recovery pool")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of train cadences held out for validation (within train pool)")
    p.add_argument("--exclude-targets", nargs="*", default=None,
                   help="Target names to force into inject-recovery pool (never trained on)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ------------------------------------------------------------------ config
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    frame_cfg = data_cfg["frame"]
    tchans = frame_cfg["tchans"]           # 96 — total cadence height
    fchans = frame_cfg["fchans"]           # 1024
    cfg_preproc = data_cfg["preprocessing"]

    cadence_list_path = Path(data_cfg["dataset"]["cadence_list"])
    all_cadences = [
        [Path(p) for p in line.strip().split()]
        for line in cadence_list_path.read_text().splitlines()
        if line.strip()
    ]

    print(f"Total cadences in manifest: {len(all_cadences)}")

    # ---------------------------------------------------------- cadence split
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    cadences = list(all_cadences)
    rng.shuffle(cadences)

    exclude = set(args.exclude_targets or [])

    def _target_name(paths):
        """Best-effort: filename field before _ON/_OFF."""
        import re
        stem = paths[0].stem
        m = re.search(r"([A-Za-z0-9_]+?)_(ON|OFF)", stem)
        return m.group(1) if m else stem

    excluded = [c for c in cadences if _target_name(c) in exclude]
    cadences  = [c for c in cadences if _target_name(c) not in exclude]

    n_train_total = int(len(cadences) * args.train_fraction)
    train_and_val = cadences[:n_train_total]
    inject_recovery = cadences[n_train_total:] + excluded

    n_val_cad = max(1, int(len(train_and_val) * args.val_fraction))
    val_cadences   = train_and_val[:n_val_cad]
    train_cadences = train_and_val[n_val_cad:]

    print(f"Train cadences : {len(train_cadences)}")
    print(f"Val cadences   : {len(val_cadences)}")
    print(f"Inject-recovery: {len(inject_recovery)}")

    # ----------------------------------- save inject-recovery cadence manifest
    args.output.mkdir(parents=True, exist_ok=True)
    inj_path = args.output / "inject_recovery_cadences.txt"
    with open(inj_path, "w") as f:
        for cad in inject_recovery:
            f.write(" ".join(str(p) for p in cad) + "\n")
    print(f"Inject-recovery cadences saved to: {inj_path}")

    # Determine n_obs and tchans_per_obs from first available cadence
    sample_cad = (train_cadences or val_cadences)[0]
    n_obs = len(sample_cad)
    tchans_per_obs = tchans // n_obs   # e.g. 96 // 6 = 16

    # ====================================================================
    # Pass 1: validate all cadences and compute exact snippet counts
    # ====================================================================
    def _plan_split(split_cadences, n_target, label):
        """Validate headers, compute per-cadence n_take. Returns [(cadence, n_take, n_avail)]."""
        print(f"\n--- Pre-scan {label}: {len(split_cadences)} cadences, target {n_target} ---")
        plan = []
        total = 0
        for cad in split_cadences:
            if total >= n_target:
                break
            nchans = _validate_cadence(cad)
            if nchans is None:
                continue
            n_avail = nchans // fchans
            n_take = min(args.snippets_per_cadence, n_avail, n_target - total)
            if n_take == 0:
                continue
            plan.append((cad, n_take, n_avail))
            total += n_take

        print(f"  {label}: {len(plan)} valid cadences, {total} snippets planned")
        return plan, total

    n_val_target = max(1000, int(args.max_snippets * args.val_fraction / (1 - args.val_fraction)))

    train_plan, n_train = _plan_split(train_cadences, args.max_snippets, "train")
    val_plan, n_val = _plan_split(val_cadences, n_val_target, "val")

    if n_train == 0:
        print("ERROR: no valid train snippets. Aborting.")
        return
    if n_val == 0:
        print("ERROR: no valid val snippets. Aborting.")
        return

    snippet_shape = (n_obs, tchans_per_obs, fchans)
    snippet_bytes = np.dtype(np.float32).itemsize * n_obs * tchans_per_obs * fchans
    train_gb = n_train * snippet_bytes / 1e9
    val_gb = n_val * snippet_bytes / 1e9
    print(f"\nData geometry:")
    print(f"  tchans per obs  : {tchans_per_obs}")
    print(f"  n_obs per cadence: {n_obs}")
    print(f"  snippet shape   : {snippet_shape}")
    print(f"  train: {n_train:,} snippets ({train_gb:.1f} GB)")
    print(f"  val  : {n_val:,} snippets ({val_gb:.1f} GB)")
    print(f"  total: {train_gb + val_gb:.1f} GB")

    # ====================================================================
    # Pass 2: allocate memmap files and extract directly to disk
    # ====================================================================
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    def _extract_to_memmap(plan, n_total, label, out_path):
        """Allocate a .npy memmap and fill it cadence by cadence."""
        shape = (n_total, *snippet_shape)
        print(f"\n=== {label}: allocating {out_path.name} {shape} ===")
        mm = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.float32, shape=shape,
        )

        cursor = 0
        for cad, n_take, n_avail in tqdm(plan, desc=label):
            indices = np_rng.choice(n_avail, size=n_take, replace=False)
            indices.sort()

            try:
                snippets = _extract_cadence_snippets(cad, indices, fchans, tchans_per_obs)
            except Exception as e:
                print(f"\nFATAL: extraction failed for {cad[0].name} "
                      f"(passed header check but failed on data read): {e}")
                print("Aborting — partial file may exist. Re-run after fixing the corrupt file.")
                raise

            mm[cursor : cursor + n_take] = snippets
            cursor += n_take
            mm.flush()

        assert cursor == n_total, f"BUG: wrote {cursor} but allocated {n_total}"
        del mm
        print(f"  {label}: {cursor:,} snippets written to {out_path}")

    train_path = out_dir / "train.npy"
    val_path = out_dir / "val.npy"

    _extract_to_memmap(train_plan, n_train, "train", train_path)
    _extract_to_memmap(val_plan, n_val, "val", val_path)

    # ---------------------------------------------------------- metadata
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "shape_per_snippet": list(snippet_shape),
        "n_train_cadences": len(train_plan),
        "n_val_cadences": len(val_plan),
        "n_inject_recovery_cadences": len(inject_recovery),
        "snippets_per_cadence": args.snippets_per_cadence,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "preprocessing": cfg_preproc,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone.")
    print(f"  {train_path}  ({n_train:,} train snippets)")
    print(f"  {val_path}  ({n_val:,} val snippets)")
    print(f"  {meta_path}")
    print(f"  {inj_path}  ({len(inject_recovery)} cadences for inject-recovery)")
    print(f"\nAdd to configs/data/gbt_fine.yaml:")
    print(f"  dataset:")
    print(f"    cache_dir: {out_dir}")


if __name__ == "__main__":
    main()
