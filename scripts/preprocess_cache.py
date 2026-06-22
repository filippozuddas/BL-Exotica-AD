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

Usage:
    PYTHONPATH=. python scripts/preprocess_cache.py \\
        --config    configs/training/srt_real.yaml \\
        --output    data/processed/cache_gbt_fine \\
        --snippets-per-cadence 3850 \\
        --max-snippets 200000 \\
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


def _read_window(path: Path, f_start: int, fchans: int, tchans_per_obs: int) -> np.ndarray:
    """Read a (tchans_per_obs, fchans) window from an HDF5 file."""
    with h5py.File(str(path), "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        dset = f["data"]
        if dset.ndim == 3:
            raw = dset[:tchans_per_obs, 0, f_start : f_start + fchans]
        else:
            raw = dset[:tchans_per_obs, f_start : f_start + fchans]
    return np.asarray(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-cadence extraction
# ---------------------------------------------------------------------------

def _extract_cadence_snippets(
    cadence_paths: list[Path],
    indices: np.ndarray,
    fchans: int,
    tchans_per_obs: int,
) -> np.ndarray:
    """
    Extract snippets at the given frequency indices from all obs in a cadence.

    Returns (n_snippets, n_obs, tchans_per_obs, fchans) float32 RAW array.
    Each file is opened once and all snippets are read in a single pass —
    avoids the overhead of 3850 × 6 = 23k file open/close per cadence.
    """
    n_obs = len(cadence_paths)
    n_snip = len(indices)
    out = np.empty((n_snip, n_obs, tchans_per_obs, fchans), dtype=np.float32)

    for oi, path in enumerate(cadence_paths):
        # 256 MB chunk cache: sorted indices let adjacent windows reuse
        # already-decompressed chunks (same pattern as RST background_extractor).
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
    p.add_argument("--snippets-per-cadence", type=int, default=3850,
                   help="Max snippets sampled per cadence (controls diversity)")
    p.add_argument("--max-snippets", type=int, default=200_000,
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

    # Check nchans and tchans_per_obs from first cadence
    sample_path = (train_cadences or val_cadences)[0][0]
    ntime_file, nchans_file = _read_header(sample_path)
    n_obs = len((train_cadences or val_cadences)[0])
    tchans_per_obs = tchans // n_obs   # e.g. 96 // 6 = 16

    print(f"\nData geometry:")
    print(f"  nchans per file : {nchans_file:,}")
    print(f"  tchans per obs  : {tchans_per_obs} (file has {ntime_file})")
    print(f"  n_obs per cadence: {n_obs}")
    print(f"  snippets available per cadence: {nchans_file // fchans:,}")

    # --------------------------------------------------------- extract helper
    def _extract_split(split_cadences, n_target, label):
        print(f"\n=== {label}: {len(split_cadences)} cadences, target {n_target} snippets ===")

        all_snippets = []
        total = 0

        for cad in tqdm(split_cadences, desc=label):
            if total >= n_target:
                break
            try:
                _, nchans = _read_header(cad[0])
            except Exception as e:
                print(f"  skip {cad[0].name}: {e}")
                continue

            n_avail = nchans // fchans
            n_take = min(args.snippets_per_cadence, n_avail, n_target - total)
            if n_take == 0:
                continue

            indices = np_rng.choice(n_avail, size=n_take, replace=False)
            indices.sort()

            try:
                snippets = _extract_cadence_snippets(cad, indices, fchans, tchans_per_obs)
            except Exception as e:
                print(f"  error {cad[0].name}: {e}")
                continue

            all_snippets.append(snippets)
            total += len(snippets)

        if not all_snippets:
            return np.empty((0, n_obs, tchans_per_obs, fchans), dtype=np.float32)

        out = np.concatenate(all_snippets, axis=0)
        print(f"  Extracted {len(out)} snippets → shape {out.shape}")
        return out

    n_val_target = max(1000, int(args.max_snippets * args.val_fraction / (1 - args.val_fraction)))

    train_data = _extract_split(train_cadences, args.max_snippets, "train")
    val_data   = _extract_split(val_cadences,   n_val_target,      "val")

    # ---------------------------------------------------------- save .npy files
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.npy"
    val_path   = out_dir / "val.npy"
    size_gb = (train_data.nbytes + val_data.nbytes) / 1e9
    print(f"\nSaving to {out_dir}/  ({size_gb:.1f} GB)...")
    np.save(str(train_path), train_data)
    np.save(str(val_path),   val_data)

    meta = {
        "n_train": int(len(train_data)),
        "n_val": int(len(val_data)),
        "shape_per_snippet": [int(n_obs), int(tchans_per_obs), int(fchans)],
        "n_train_cadences": len(train_cadences),
        "n_val_cadences": len(val_cadences),
        "n_inject_recovery_cadences": len(inject_recovery),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "preprocessing": cfg_preproc,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone.")
    print(f"  {train_path}  ({len(train_data)} train snippets)")
    print(f"  {val_path}  ({len(val_data)} val snippets)")
    print(f"  {meta_path}")
    print(f"  {inj_path}  ({len(inject_recovery)} cadences for inject-recovery)")
    print(f"\nAdd to configs/data/gbt_fine.yaml:")
    print(f"  dataset:")
    print(f"    cache_dir: {out_dir}")


if __name__ == "__main__":
    main()
