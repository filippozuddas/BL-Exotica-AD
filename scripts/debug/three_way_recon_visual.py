"""Three-way reconstruction comparison: pure noise vs RFI vs injected-ETI.

Simple visual/numeric sanity check requested directly (not an AUC/statistical
test): for one example of each class, plot input / reconstruction / squared-
error map, and report mean MSE + max squared-error side by side. The question
this answers by eye: does the model's max reconstruction error on the ETI
signal actually stand out from its max error on ordinary RFI, or do all three
classes sit at a similar error level (consistent with the matched-energy
findings in encode_separation_test.py / ae_recon_visual.py, where RFI's local
residual was found to meet or exceed the injected line's)?

Works for any backbone exposing forward(x) -> reconstruction of the same
shape (AE / MemAE / ViT-MAE) — reuses ae_recon_visual.py's loader/injector.

Usage:
    python scripts/debug/three_way_recon_visual.py \
        --checkpoint outputs/.../best_model.ckpt \
        --model_config configs/model/memae.yaml \
        --cache /path/to/cache_gbt_fine --split val \
        --inject_snr 25
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.debug.ae_recon_visual import (
    add_obs_boundaries, build_narrowband_generator, inject_narrowband_on_only,
    load_model, preprocess_raw, reconstruct,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--inject_snr", type=float, default=25.0)
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--n_scan", type=int, default=200,
                   help="Pool size to pick the most-quiet / most-RFI-like example from.")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/injection_test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)
    preproc = cfg["preprocessing"]
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)
    generator = build_narrowband_generator(cfg, seed=args.seed)

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    n_total = arr.shape[0]
    n_scan = min(args.n_scan, n_total)
    scan_idx = rng.choice(n_total, size=n_scan, replace=False)
    scan_raw = np.array(arr[scan_idx])
    del arr

    hot_fracs = np.array([
        float((preprocess_raw(scan_raw[i], preproc) > 5.0).sum()) /
        preprocess_raw(scan_raw[i], preproc).size
        for i in range(n_scan)
    ])
    order = np.argsort(hot_fracs)
    quiet_i = order[0]        # lowest hot_frac -> pure noise
    rfi_i = order[-1]         # highest hot_frac -> RFI

    # Row 1: pure noise (quiet, no injection)
    noise_frame = preprocess_raw(scan_raw[quiet_i], preproc)
    # Row 2: RFI (no injection)
    rfi_frame = preprocess_raw(scan_raw[rfi_i], preproc)
    # Row 3: ETI injected into the SAME quiet snippet as row 1
    raw_inj, signal_mask_obs = inject_narrowband_on_only(
        generator, scan_raw[quiet_i], snr=args.inject_snr, drift_rate=args.drift_rate)
    eti_frame = preprocess_raw(raw_inj, preproc)

    rows = [
        ("Pure noise (quiet)", noise_frame),
        ("RFI", rfi_frame),
        (f"ETI injected (SNR={args.inject_snr})", eti_frame),
    ]

    print(f"\n{'='*64}\nMAX/MEAN SQUARED-ERROR COMPARISON\n{'='*64}")
    print(f"  {'class':<26s}  {'mean_MSE':>10s}  {'max_error':>10s}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}")

    # First pass: compute all errors so the shared color scale (col 4) can use
    # the true global max across classes, not just this row's.
    summary = []
    recons, errors = [], []
    for label, frame in rows:
        recon = reconstruct(model, frame, args.device)
        error = (frame - recon) ** 2
        recons.append(recon)
        errors.append(error)
        summary.append((label, float(error.mean()), float(error.max())))
        print(f"  {label:<26s}  {summary[-1][1]:10.4f}  {summary[-1][2]:10.2f}")
    global_err_vmax = max(s[2] for s in summary)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    col_titles = ["Input", "Reconstruction", "Squared error\n(row-wise scale)",
                  "Squared error\n(shared scale)"]

    for row, ((label, frame), recon, error) in enumerate(zip(rows, recons, errors)):
        mean_mse, max_err = summary[row][1], summary[row][2]

        vmin, vmax = np.percentile(frame, [1, 99])
        err_vmax = max(np.percentile(error, 99), 1e-6)

        axes[row][0].imshow(frame, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row][1].imshow(recon, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        im = axes[row][2].imshow(error, aspect="auto", origin="upper",
                                 cmap="hot", vmin=0, vmax=err_vmax)
        fig.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.04)
        im2 = axes[row][3].imshow(error, aspect="auto", origin="upper",
                                  cmap="hot", vmin=0, vmax=global_err_vmax)
        fig.colorbar(im2, ax=axes[row][3], fraction=0.046, pad=0.04)

        axes[row][0].set_ylabel(f"{label}\ntime bin", fontsize=10)
        axes[row][1].set_title(f"mean MSE={mean_mse:.3f}", fontsize=9)
        axes[row][2].set_title(f"max error={max_err:.1f}", fontsize=9)
        axes[row][3].set_title(f"max error={max_err:.1f} (of global {global_err_vmax:.1f})", fontsize=9)
        for col in range(4):
            add_obs_boundaries(axes[row][col])
            axes[row][col].set_xlabel("freq channel")

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(f"{title}\n{axes[0][col].get_title()}", fontsize=10)

    fig.suptitle("Pure noise vs RFI vs injected-ETI — reconstruction & error comparison "
                 "(col 4 uses ONE shared color scale across all 3 rows — directly comparable)",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "three_way_recon.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")

    labels = [s[0] for s in summary]
    maxes = [s[2] for s in summary]
    print(f"\n  VERDICT hint: if max_error for 'ETI injected' is NOT clearly above "
          f"'RFI', the model's error signal does not distinguish the target signal "
          f"from ordinary RFI at the pixel level (consistent with the matched-energy "
          f"AUC findings). Observed max_error: "
          f"{', '.join(f'{l}={m:.1f}' for l, m in zip(labels, maxes))}.")


if __name__ == "__main__":
    main()
