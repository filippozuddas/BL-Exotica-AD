"""
RFI composition analysis.

Quantifies what fraction of training snippets contain RFI and characterises
the RFI by type (narrowband / broadband-temporal / bright spikes).

Metrics are computed on fully preprocessed snippets (bandpass_correct +
core_transform), exactly as the model sees them. After core_transform the
noise is approximately Gaussian centred at 0 with MAD=1 (std ~1.48).

Two data sources are supported (auto-detected):
  --cache   path to the NPZ cache (CachedDataset format).
            Shape: (N, n_obs, tchans_per_obs, fchans), raw power values.
            Opened with mmap_mode='r' so only the sampled rows are loaded.
  --cadences / --data_config
            fallback: load from raw HDF5 files via SpectrogramDataset.

Usage
-----
    # preferred — uses existing NPZ cache with memory-mapping
    PYTHONPATH=. python scripts/debug/rfi_composition_analysis.py \\
        --cache /home/acabras/data/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine.npz \\
        --data_config configs/data/gbt_fine.yaml \\
        --split train \\
        --n_samples 3000 \\
        --out_dir outputs/rfi_analysis

    # fallback — loads from raw HDF5 files
    PYTHONPATH=. python scripts/debug/rfi_composition_analysis.py \\
        --cadences data/raw/srt_0000_cadences.txt \\
        --data_config configs/data/gbt_fine.yaml \\
        --max_cadences 5 \\
        --n_samples 3000 \\
        --out_dir outputs/rfi_analysis
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import bandpass_correct, core_transform


# ---------------------------------------------------------------------------
# RFI thresholds (in normalised MAD units after core_transform)
# Noise std ≈ 1.48 MAD units (MAD ≈ 0.674σ for Gaussian).
# THRESHOLD_LO ≈ 3.4σ,  THRESHOLD_HI ≈ 6.8σ.
# ---------------------------------------------------------------------------
THRESHOLD_LO = 5.0
THRESHOLD_HI = 10.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(snippet: np.ndarray) -> dict:
    """
    RFI diagnostics for a single preprocessed (tchans, fchans) snippet.

    Returns keys: max_val, p99, hot_frac_lo, hot_frac_hi,
                  col_mean_std, row_mean_std.
    """
    max_val      = float(snippet.max())
    p99          = float(np.percentile(snippet, 99))
    n_pix        = snippet.size
    hot_frac_lo  = float((snippet > THRESHOLD_LO).sum()) / n_pix
    hot_frac_hi  = float((snippet > THRESHOLD_HI).sum()) / n_pix
    col_mean_std = float(snippet.mean(axis=0).std())   # narrowband indicator
    row_mean_std = float(snippet.mean(axis=1).std())   # temporal RFI indicator
    return dict(max_val=max_val, p99=p99,
                hot_frac_lo=hot_frac_lo, hot_frac_hi=hot_frac_hi,
                col_mean_std=col_mean_std, row_mean_std=row_mean_std)


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

class NpzSampler:
    """
    Random-access sampler over a CachedDataset NPZ.

    Opens the file with mmap_mode='r' so only the rows we actually request
    are paged in from disk — no need to load the full 73 GB into RAM.

    The NPZ stores raw (unpreprocessed) data as
        arr[i]  →  (n_obs, tchans_per_obs, fchans)
    __getitem__ concatenates along the time axis, applies bandpass_correct
    and core_transform, and returns (tchans, fchans) float32.
    """

    def __init__(self, npz_path: Path, split: str, cfg_preproc: dict):
        print(f"Opening NPZ cache (mmap_mode='r'): {npz_path}")
        archive = np.load(str(npz_path), mmap_mode="r")
        self.data = archive[split]          # (N, n_obs, tchans_per_obs, fchans)
        self.cfg_preproc = cfg_preproc
        print(f"  split='{split}'  shape={self.data.shape}  "
              f"dtype={self.data.dtype}  "
              f"~{self.data.nbytes / 1e9:.1f} GB (memory-mapped)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> np.ndarray:
        method      = self.cfg_preproc.get("bandpass_method", "polynomial")
        poly_degree = self.cfg_preproc.get("poly_degree", 3)
        mad_epsilon = self.cfg_preproc.get("mad_epsilon", 1e-6)

        raw    = np.array(self.data[idx])          # force load: (n_obs, tchans_per_obs, fchans)
        result = np.concatenate(raw, axis=0)       # (tchans, fchans)
        result = bandpass_correct(result, method=method, poly_degree=poly_degree)
        result = core_transform(result, mad_epsilon)
        return result.astype(np.float32)


class DatasetSampler:
    """Thin wrapper around SpectrogramDataset for the fallback HDF5 path."""

    def __init__(self, cadences_path: Path, cfg: dict, max_cadences: int):
        from src.data.torch_dataset import SpectrogramDataset
        cadences = []
        with open(cadences_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cadences.append([Path(p) for p in line.split()])
                if len(cadences) >= max_cadences:
                    break
        print(f"Cadences loaded: {len(cadences)}")
        frame   = cfg["frame"]
        preproc = cfg["preprocessing"]
        self._ds = SpectrogramDataset(
            cadence_paths=cadences,
            tchans=frame["tchans"],
            fchans=frame["fchans"],
            stride=frame["stride_train"],
            cfg_preproc=preproc,
            downsample_factor=frame.get("downsample_factor", 1),
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._ds[idx].numpy()[0]   # (tchans, fchans)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_histograms(data: dict, out_path: Path) -> None:
    keys = ["max_val", "p99", "hot_frac_lo", "hot_frac_hi",
            "col_mean_std", "row_mean_std"]
    labels = [
        "Max pixel value (normalised)",
        "99th-percentile pixel value",
        f"Hot fraction (> {THRESHOLD_LO:.0f} MAD units, ~3.4σ)",
        f"Hot fraction (> {THRESHOLD_HI:.0f} MAD units, ~6.8σ)",
        "Std of per-column means  [narrowband indicator]",
        "Std of per-row means  [temporal RFI indicator]",
    ]
    log_x = [False, False, True, True, False, False]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("RFI composition — snippet metrics after core_transform", fontsize=13)
    for ax, key, label, logx in zip(axes.flat, keys, labels, log_x):
        v = data[key]
        if logx:
            v_pos = v[v > 0]
            if len(v_pos):
                bins = np.logspace(np.log10(v_pos.min()), np.log10(v_pos.max()), 50)
                ax.hist(v_pos, bins=bins, color="steelblue", edgecolor="none", alpha=0.8)
                ax.set_xscale("log")
            n_zero = int((v == 0).sum())
            suffix = f"\n({n_zero}/{len(v)} = {100*n_zero/len(v):.1f}% are exactly 0)"
            ax.set_title(label + (suffix if n_zero else ""), fontsize=8)
        else:
            ax.hist(v, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
            ax.set_title(label, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_scatter(data: dict, out_path: Path) -> None:
    max_v = data["max_val"]
    vmin  = max(max_v.min(), 0.1)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(data["col_mean_std"], data["hot_frac_lo"],
                    c=max_v, cmap="plasma", s=6, alpha=0.5,
                    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=max_v.max()))
    plt.colorbar(sc, ax=ax, label="max pixel value")
    ax.set_xlabel("col_mean_std  (narrowband indicator)")
    ax.set_ylabel(f"hot_frac_lo  (fraction > {THRESHOLD_LO:.0f})")
    ax.set_title("Snippet RFI characterisation")
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_examples(sampler, indices: list, records: list, data: dict,
                   is_clean, is_mild, is_strong, out_path: Path) -> None:
    categories = [
        ("Clean noise",  np.where(is_clean)[0]),
        ("Mild RFI",     np.where(is_mild)[0]),
        ("Strong RFI",   np.where(is_strong)[0]),
    ]
    rows = [(lbl, arr) for lbl, arr in categories if len(arr) > 0]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax_row, (label, idx_arr) in zip(axes, rows):
        cat_max = data["max_val"][idx_arr]
        pick    = idx_arr[np.argsort(cat_max)[len(cat_max) // 2]]
        arr     = sampler[indices[pick]]
        m       = records[pick]
        ax_row[0].imshow(arr, aspect="auto", origin="lower", cmap="viridis")
        ax_row[0].set_title(
            f"{label}\nmax={m['max_val']:.1f}  "
            f"hot_lo={m['hot_frac_lo']:.2e}  col_std={m['col_mean_std']:.3f}",
            fontsize=8)
        ax_row[0].set_ylabel("time bin")
        ax_row[0].set_xlabel("freq channel")
        ax_row[1].plot(arr.mean(axis=0), lw=0.8)
        ax_row[1].set_title("Mean over time  (frequency profile)", fontsize=8)
        ax_row[1].set_xlabel("freq channel")
        ax_row[1].set_ylabel("mean value (normalised)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache",        type=Path, default=None,
                   help="Path to NPZ cache (preferred). If set, --cadences is ignored.")
    p.add_argument("--split",        default="train",
                   help="NPZ split to analyse: 'train' or 'val'. Default: train.")
    p.add_argument("--cadences",     type=Path, default=None,
                   help="Fallback: cadence list for SpectrogramDataset.")
    p.add_argument("--data_config",  type=Path,
                   default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--max_cadences", type=int, default=10,
                   help="Cap on cadences to load (fallback path only). Default: 10.")
    p.add_argument("--n_samples",    type=int, default=3000)
    p.add_argument("--out_dir",      type=Path,
                   default=ROOT / "outputs/rfi_analysis")
    p.add_argument("--seed",         type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)
    preproc = cfg["preprocessing"]

    # --- build sampler ---
    if args.cache is not None:
        sampler = NpzSampler(args.cache, args.split, preproc)
    elif args.cadences is not None:
        sampler = DatasetSampler(args.cadences, cfg, args.max_cadences)
    else:
        # try the path in the data config
        cadence_file = ROOT / cfg.get("dataset", {}).get("cadence_list", "")
        if not cadence_file.exists():
            raise ValueError("Provide --cache or --cadences (or set dataset.cadence_list in the data config).")
        sampler = DatasetSampler(cadence_file, cfg, args.max_cadences)

    n_total   = len(sampler)
    n_samples = min(args.n_samples, n_total)
    indices   = rng.choice(n_total, size=n_samples, replace=False).tolist()
    print(f"\nTotal snippets: {n_total}  →  sampling {n_samples}")

    # --- compute metrics ---
    records = []
    for i, idx in enumerate(indices):
        arr = sampler[idx]
        records.append(compute_metrics(arr))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_samples}")

    keys = ["max_val", "p99", "hot_frac_lo", "hot_frac_hi",
            "col_mean_std", "row_mean_std"]
    data = {k: np.array([r[k] for r in records]) for k in keys}

    # --- summary ---
    print("\n=== Metric summary ===")
    for k in keys:
        v = data[k]
        print(f"  {k:20s}:  mean={v.mean():.4f}  p50={np.percentile(v,50):.4f}"
              f"  p95={np.percentile(v,95):.4f}  p99={np.percentile(v,99):.4f}"
              f"  max={v.max():.4f}")

    is_strong = data["hot_frac_hi"] > 0
    is_mild   = (~is_strong) & (data["hot_frac_lo"] > 0)
    is_clean  = ~(is_strong | is_mild)
    print("\n=== RFI composition ===")
    print(f"  Clean  (no pixel > {THRESHOLD_LO:.0f}):                   "
          f"{is_clean.sum():5d}/{n_samples}  ({100*is_clean.mean():.1f}%)")
    print(f"  Mild   (pixel > {THRESHOLD_LO:.0f}, none > {THRESHOLD_HI:.0f}): "
          f"{is_mild.sum():5d}/{n_samples}  ({100*is_mild.mean():.1f}%)")
    print(f"  Strong (pixel > {THRESHOLD_HI:.0f}):                  "
          f"{is_strong.sum():5d}/{n_samples}  ({100*is_strong.mean():.1f}%)")

    col_thresh = np.percentile(data["col_mean_std"], 90)
    row_thresh = np.percentile(data["row_mean_std"], 90)
    print(f"\n  col_mean_std > p90 ({col_thresh:.3f}) [narrowband indicator]: "
          f"{(data['col_mean_std'] > col_thresh).sum()} snippets")
    print(f"  row_mean_std > p90 ({row_thresh:.3f}) [temporal indicator]:   "
          f"{(data['row_mean_std'] > row_thresh).sum()} snippets")

    # --- plots ---
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot_histograms(data, args.out_dir / "metric_histograms.png")
    print(f"\nSaved → {args.out_dir / 'metric_histograms.png'}")

    _plot_scatter(data, args.out_dir / "rfi_scatter.png")
    print(f"Saved → {args.out_dir / 'rfi_scatter.png'}")

    _plot_examples(sampler, indices, records, data,
                   is_clean, is_mild, is_strong,
                   args.out_dir / "example_snippets.png")
    print(f"Saved → {args.out_dir / 'example_snippets.png'}")

    csv_path = args.out_dir / "snippet_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx"] + keys)
        writer.writeheader()
        for idx, r in zip(indices, records):
            writer.writerow({"idx": idx, **r})
    print(f"Saved → {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
