"""
Visualize AE/ViT-MAE reconstruction of ON-only injected ETI signals.

Shows 4 columns per example:
  1. Original quiet snippet (preprocessed)
  2. With ON-only injection (preprocessed)
  3. Model reconstruction of the injected snippet
  4. Per-pixel squared error

Red dashed lines mark ON/OFF observation boundaries.

Usage:
    python scripts/debug/ae_recon_visual.py \
        --checkpoint outputs/.../best_model.ckpt \
        --cache /path/to/cache_gbt_fine \
        --model_config configs/model/convae.yaml \
        --n_examples 4 --inject_snr 25
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

from src.data.preprocessing import bandpass_correct, core_transform
from src.models.autoencoder import build_autoencoder
from scripts.debug.injection_vs_rfi_test import inject_narrowband_on_only


def load_model(checkpoint_path, model_config, device="cpu"):
    model = build_autoencoder((96, 1024, 1), model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def preprocess_raw(raw_snippet, cfg_preproc):
    method = cfg_preproc.get("bandpass_method", "polynomial")
    poly_degree = cfg_preproc.get("poly_degree", 3)
    mad_epsilon = cfg_preproc.get("mad_epsilon", 1e-6)
    frame = np.concatenate(raw_snippet, axis=0)
    frame = bandpass_correct(frame, method=method, poly_degree=poly_degree)
    frame = core_transform(frame, mad_epsilon)
    return frame.astype(np.float32)


def reconstruct(model, snippet, device="cpu"):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze().cpu().numpy()


def add_obs_boundaries(ax, tchans_per_obs=16, n_obs=6):
    for i in range(1, n_obs):
        ax.axhline(i * tchans_per_obs - 0.5, color="red", ls="--", lw=0.8, alpha=0.7)
    on_obs = [0, 2, 4]
    for obs in on_obs:
        y_mid = obs * tchans_per_obs + tchans_per_obs / 2
        ax.text(-0.02, y_mid, "ON", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color="red", fontweight="bold")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True)
    p.add_argument("--data_config", type=Path,
                   default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--n_examples", type=int, default=4)
    p.add_argument("--inject_snr", type=float, default=25.0)
    p.add_argument("--drift_rate", type=float, default=0.3)
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

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    n_total = arr.shape[0]

    # Pick quiet snippets (low hot_frac)
    n_scan = min(200, n_total)
    scan_idx = rng.choice(n_total, size=n_scan, replace=False)
    scan_raw = np.array(arr[scan_idx])
    del arr

    hot_fracs = []
    for i in range(n_scan):
        snip = preprocess_raw(scan_raw[i], preproc)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)
    hot_fracs = np.array(hot_fracs)
    quiet_order = np.argsort(hot_fracs)
    chosen = quiet_order[:args.n_examples]

    print(f"Plotting {args.n_examples} examples...")
    n = len(chosen)
    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Original (quiet)", "ON-only injection",
                   "Reconstruction", "Squared error", "Difference\n(injection − original)"]

    for row, ci in enumerate(chosen):
        raw = scan_raw[ci]
        orig = preprocess_raw(raw, preproc)
        raw_inj = inject_narrowband_on_only(raw, snr=args.inject_snr,
                                            drift_rate=args.drift_rate,
                                            seed=args.seed + row)
        injected = preprocess_raw(raw_inj, preproc)
        recon = reconstruct(model, injected, args.device)
        error = (injected - recon) ** 2
        diff = injected - orig

        vmin, vmax = np.percentile(orig, [1, 99])
        err_vmax = np.percentile(error, 99)
        diff_abs = np.abs(diff)
        diff_vmax = np.percentile(diff_abs[diff_abs > 0], 99) if (diff_abs > 0).any() else 1

        # Col 0: original
        axes[row][0].imshow(orig, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        mse_orig = np.mean((orig - reconstruct(model, orig, args.device)) ** 2)
        axes[row][0].set_title(f"MSE={mse_orig:.3f}", fontsize=9)

        # Col 1: injected (ON-only)
        axes[row][1].imshow(injected, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row][1].set_title(f"SNR={args.inject_snr}", fontsize=9)

        # Col 2: reconstruction
        axes[row][2].imshow(recon, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        mse_inj = np.mean(error)
        axes[row][2].set_title(f"MSE={mse_inj:.3f}", fontsize=9)

        # Col 3: error map
        axes[row][3].imshow(error, aspect="auto", origin="upper",
                            cmap="hot", vmin=0, vmax=err_vmax)
        axes[row][3].set_title(f"max={error.max():.1f}", fontsize=9)

        # Col 4: difference (shows where injection landed)
        axes[row][4].imshow(diff, aspect="auto", origin="upper",
                            cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
        axes[row][4].set_title("ON-only signal trace", fontsize=9)

        for col in range(5):
            add_obs_boundaries(axes[row][col])
            axes[row][col].set_ylabel("time bin")
            axes[row][col].set_xlabel("freq channel")

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(f"{title}\n{axes[0][col].get_title()}", fontsize=9)

    fig.suptitle(
        f"AE reconstruction of ON-only ETI injection (SNR={args.inject_snr}, "
        f"drift={args.drift_rate} Hz/s)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "recon_on_only.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
