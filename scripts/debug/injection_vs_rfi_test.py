"""
Injection vs RFI discrimination test.

Tests whether the current ViT-MAE checkpoint can distinguish injected signals
from real RFI by comparing per-pixel reconstruction error. If injected signals
produce significantly higher error than RFI, the model is already usable for
anomaly detection — even if its average loss sits at the variance floor.

Three snippet categories:
  - "Quiet": low RFI content (baseline reconstruction error)
  - "RFI":   high RFI content (should be reconstructed reasonably if learned)
  - "Injected": quiet snippet + synthetic narrowband signal added to raw data
                before preprocessing (mimics a real technosignature)

Usage:
    PYTHONPATH=. python scripts/debug/injection_vs_rfi_test.py \
        --checkpoint outputs/20260617_134719_dc9d83c/checkpoints/best_model.ckpt \
        --cache /path/to/cache_gbt_fine \
        --data_config configs/data/gbt_fine.yaml \
        --n_samples 50 \
        --out_dir outputs/injection_test
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
from src.data.synthetic import NarrowbandDriftingGenerator, NarrowbandParams
from src.models.autoencoder import build_autoencoder

# Bare dataclass defaults match configs/data/gbt_fine.yaml exactly (geometry +
# narrowband sampling ranges) -- the only product these debug scripts inject
# into -- so no config file needs loading here.
_NB_PARAMS = NarrowbandParams()


def load_model(checkpoint_path: Path, model_config: dict, device: str = "cpu"):
    model = build_autoencoder((96, 1024, 1), model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def preprocess_raw(raw_snippet: np.ndarray, cfg_preproc: dict) -> np.ndarray:
    """(n_obs, tchans_per_obs, fchans) raw -> (tchans, fchans) preprocessed.

    Matches CachedDataset/SpectrogramDataset (src/data/torch_dataset.py,
    commit 4b5660c): bandpass_correct + core_transform applied per
    observation, THEN concatenated -- not on the pre-concatenated frame.
    Concatenating first lets a real ON/OFF power step between observations
    survive as spurious cross-block contrast.
    """
    method = cfg_preproc.get("bandpass_method", "polynomial")
    poly_degree = cfg_preproc.get("poly_degree", 3)
    mad_epsilon = cfg_preproc.get("mad_epsilon", 1e-6)
    normed = [
        core_transform(bandpass_correct(obs, method=method, poly_degree=poly_degree), mad_epsilon)
        for obs in raw_snippet
    ]
    frame = np.concatenate(normed, axis=0)
    return frame.astype(np.float32)


def inject_narrowband(raw_snippet: np.ndarray, snr: float = 25.0,
                      drift_rate: float = 0.3, seed: int = 42) -> np.ndarray:
    """Inject a narrowband drifting signal into raw data (before preprocessing),
    via setigen (``NarrowbandDriftingGenerator.inject_signal`` on the
    concatenated frame) — spans the whole cadence continuously, no ON/OFF
    distinction (see ``inject_narrowband_on_only`` for the ON-only variant that
    mimics a real technosignature)."""
    raw = np.asarray(raw_snippet, dtype=float)
    n_obs, tchans_per_obs, fchans = raw.shape
    frame = np.concatenate(raw, axis=0)  # (n_obs*tchans_per_obs, fchans)

    gen = NarrowbandDriftingGenerator(_NB_PARAMS, seed=seed)
    out_frame, _ = gen.inject_signal(frame, snr=snr, drift_rate=drift_rate)

    result = np.zeros_like(raw)
    for i in range(n_obs):
        result[i] = out_frame[i * tchans_per_obs:(i + 1) * tchans_per_obs]
    return result.astype(np.float32)


def inject_narrowband_on_only(raw_snippet: np.ndarray, snr: float = 25.0,
                              drift_rate: float = 0.3, seed: int = 42) -> np.ndarray:
    """Inject a narrowband signal into ON observations only (obs 0, 2, 4), via
    setigen (``NarrowbandDriftingGenerator.inject_signal_cadence``): a single
    coherent drift track spanning the whole cadence (correct drift phase across
    the real elapsed OFF-observation time too), rendered only into the ON
    frames — OFF frames come back byte-identical, exactly as a real
    technosignature would vanish off-source.
    """
    gen = NarrowbandDriftingGenerator(_NB_PARAMS, seed=seed)
    out, _ = gen.inject_signal_cadence(raw_snippet, snr=snr, drift_rate=drift_rate)
    return out


def compute_error_map(model, snippet: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run model forward pass and return per-pixel squared error."""
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    error = (x - recon).squeeze().cpu().numpy() ** 2
    return error


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True,
                   help="Cache directory path")
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path,
                   default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path,
                   default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--n_samples", type=int, default=50,
                   help="Snippets to sample for RFI/quiet categorisation")
    p.add_argument("--inject_snr", type=float, default=25.0,
                   help="SNR of injected narrowband signal")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "outputs/injection_test")
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

    # --- Load model ---
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)
    print(f"  Model loaded on {args.device}")

    # --- Load snippets from cache ---
    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    n_total = arr.shape[0]
    indices = rng.choice(n_total, size=min(args.n_samples, n_total), replace=False)
    print(f"  Loading {len(indices)} raw snippets...")
    raw_snippets = np.array(arr[indices])  # (n_samples, n_obs, tchans_per_obs, fchans)
    del arr
    print(f"  Shape: {raw_snippets.shape}")

    # --- Preprocess all and compute RFI metrics ---
    print("Preprocessing and computing metrics...")
    preprocessed = []
    hot_fracs = []
    for i in range(len(raw_snippets)):
        snip = preprocess_raw(raw_snippets[i], preproc)
        preprocessed.append(snip)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)

    hot_fracs = np.array(hot_fracs)
    preprocessed = np.array(preprocessed)

    # Split into quiet (bottom 25%) and RFI-rich (top 25%)
    q25, q75 = np.percentile(hot_fracs, [25, 75])
    quiet_mask = hot_fracs <= q25
    rfi_mask = hot_fracs >= q75
    quiet_idx = np.where(quiet_mask)[0]
    rfi_idx = np.where(rfi_mask)[0]
    print(f"  Quiet (hot_frac <= {q25:.6f}): {len(quiet_idx)} snippets")
    print(f"  RFI   (hot_frac >= {q75:.6f}): {len(rfi_idx)} snippets")

    # --- Compute reconstruction errors ---
    print("Computing reconstruction errors...")

    # 1. Quiet snippets (baseline)
    quiet_errors = []
    for i in quiet_idx:
        err = compute_error_map(model, preprocessed[i], args.device)
        quiet_errors.append(err.mean())
    quiet_errors = np.array(quiet_errors)

    # 2. RFI snippets
    rfi_errors = []
    rfi_hot_errors = []  # error only on hot pixels
    for i in rfi_idx:
        err = compute_error_map(model, preprocessed[i], args.device)
        rfi_errors.append(err.mean())
        hot_mask = preprocessed[i] > 5.0
        if hot_mask.any():
            rfi_hot_errors.append(err[hot_mask].mean())
    rfi_errors = np.array(rfi_errors)
    rfi_hot_errors = np.array(rfi_hot_errors) if rfi_hot_errors else np.array([0.0])

    # 3. Injected signals (into quiet snippets, all obs)
    inject_all_errors = []
    inject_all_signal_errors = []  # error on signal pixels only
    n_inject = min(10, len(quiet_idx))
    injected_examples = []
    for j, i in enumerate(quiet_idx[:n_inject]):
        raw_inj = inject_narrowband(raw_snippets[i], snr=args.inject_snr, seed=args.seed + j)
        snip_inj = preprocess_raw(raw_inj, preproc)
        err = compute_error_map(model, snip_inj, args.device)
        inject_all_errors.append(err.mean())
        signal_mask = np.abs(snip_inj - preprocessed[i]) > 1.0
        if signal_mask.any():
            inject_all_signal_errors.append(err[signal_mask].mean())
        if j < 3:
            injected_examples.append((preprocessed[i], snip_inj, err, signal_mask))
    inject_all_errors = np.array(inject_all_errors)
    inject_all_signal_errors = np.array(inject_all_signal_errors) if inject_all_signal_errors else np.array([0.0])

    # 4. Injected signals (ON only — the realistic case)
    inject_on_errors = []
    inject_on_signal_errors = []
    for j, i in enumerate(quiet_idx[:n_inject]):
        raw_inj = inject_narrowband_on_only(raw_snippets[i], snr=args.inject_snr, seed=args.seed + j)
        snip_inj = preprocess_raw(raw_inj, preproc)
        err = compute_error_map(model, snip_inj, args.device)
        inject_on_errors.append(err.mean())
        signal_mask = np.abs(snip_inj - preprocessed[i]) > 1.0
        if signal_mask.any():
            inject_on_signal_errors.append(err[signal_mask].mean())
    inject_on_errors = np.array(inject_on_errors)
    inject_on_signal_errors = np.array(inject_on_signal_errors) if inject_on_signal_errors else np.array([0.0])

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RECONSTRUCTION ERROR COMPARISON")
    print("=" * 60)
    print(f"\n  {'Category':<30s}  {'Mean MSE':>10s}  {'Std':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"  {'Quiet (baseline)':<30s}  {quiet_errors.mean():10.4f}  {quiet_errors.std():10.4f}")
    print(f"  {'RFI (full snippet)':<30s}  {rfi_errors.mean():10.4f}  {rfi_errors.std():10.4f}")
    print(f"  {'RFI (hot pixels only)':<30s}  {rfi_hot_errors.mean():10.4f}  {rfi_hot_errors.std():10.4f}")
    print(f"  {'Injected ALL obs':<30s}  {inject_all_errors.mean():10.4f}  {inject_all_errors.std():10.4f}")
    print(f"  {'Injected ALL (signal px)':<30s}  {inject_all_signal_errors.mean():10.4f}  {inject_all_signal_errors.std():10.4f}")
    print(f"  {'Injected ON-only':<30s}  {inject_on_errors.mean():10.4f}  {inject_on_errors.std():10.4f}")
    print(f"  {'Injected ON-only (signal px)':<30s}  {inject_on_signal_errors.mean():10.4f}  {inject_on_signal_errors.std():10.4f}")

    # Discrimination ratios
    print(f"\n  --- Discrimination ratios ---")
    if quiet_errors.mean() > 0:
        print(f"  RFI / Quiet:              {rfi_errors.mean() / quiet_errors.mean():.3f}x")
        print(f"  Inject-ALL / Quiet:       {inject_all_errors.mean() / quiet_errors.mean():.3f}x")
        print(f"  Inject-ON / Quiet:        {inject_on_errors.mean() / quiet_errors.mean():.3f}x")
    if rfi_hot_errors.mean() > 0 and inject_all_signal_errors.mean() > 0:
        print(f"  Inject-ALL signal / RFI hot: {inject_all_signal_errors.mean() / rfi_hot_errors.mean():.3f}x")
    if rfi_hot_errors.mean() > 0 and inject_on_signal_errors.mean() > 0:
        print(f"  Inject-ON signal / RFI hot:  {inject_on_signal_errors.mean() / rfi_hot_errors.mean():.3f}x")

    # --- Plots ---
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Error distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Injection vs RFI discrimination (inject SNR={args.inject_snr})", fontsize=12)

    ax = axes[0]
    ax.set_title("Full-snippet mean MSE")
    categories = ["Quiet", "RFI", "Inject\nALL obs", "Inject\nON only"]
    means = [quiet_errors.mean(), rfi_errors.mean(),
             inject_all_errors.mean(), inject_on_errors.mean()]
    stds = [quiet_errors.std(), rfi_errors.std(),
            inject_all_errors.std(), inject_on_errors.std()]
    colors = ["steelblue", "orange", "crimson", "darkred"]
    bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean per-pixel MSE")
    ax.axhline(quiet_errors.mean(), ls="--", color="steelblue", alpha=0.5, label="quiet baseline")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_title("Signal/hot-pixel MSE only")
    categories2 = ["RFI\nhot px", "Inject-ALL\nsignal px", "Inject-ON\nsignal px"]
    means2 = [rfi_hot_errors.mean(), inject_all_signal_errors.mean(),
              inject_on_signal_errors.mean()]
    stds2 = [rfi_hot_errors.std(), inject_all_signal_errors.std(),
             inject_on_signal_errors.std()]
    colors2 = ["orange", "crimson", "darkred"]
    ax.bar(categories2, means2, yerr=stds2, color=colors2, alpha=0.8,
           capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean per-pixel MSE (hot/signal pixels)")

    plt.tight_layout()
    plt.savefig(args.out_dir / "error_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved → {args.out_dir / 'error_comparison.png'}")

    # Plot 2: Example error maps
    if injected_examples:
        n_ex = len(injected_examples)
        fig, axes = plt.subplots(n_ex, 4, figsize=(20, 5 * n_ex))
        if n_ex == 1:
            axes = [axes]
        for row, (original, injected, error, sig_mask) in enumerate(injected_examples):
            vmin, vmax = np.percentile(original, [1, 99])
            axes[row][0].imshow(original, aspect="auto", origin="lower",
                                cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row][0].set_title("Original (quiet)", fontsize=9)
            axes[row][1].imshow(injected, aspect="auto", origin="lower",
                                cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row][1].set_title(f"With injection (SNR={args.inject_snr})", fontsize=9)
            axes[row][2].imshow(error, aspect="auto", origin="lower",
                                cmap="hot", vmin=0, vmax=np.percentile(error, 99))
            axes[row][2].set_title(f"Reconstruction error (MSE={error.mean():.3f})", fontsize=9)
            diff = np.abs(injected - original)
            axes[row][3].imshow(diff, aspect="auto", origin="lower",
                                cmap="hot", vmin=0, vmax=np.percentile(diff[diff > 0], 99) if (diff > 0).any() else 1)
            axes[row][3].set_title("Signal mask (|injected - original|)", fontsize=9)
            for ax in axes[row]:
                ax.set_ylabel("time bin")
                ax.set_xlabel("freq channel")
        plt.tight_layout()
        plt.savefig(args.out_dir / "example_error_maps.png", dpi=150)
        plt.close()
        print(f"Saved → {args.out_dir / 'example_error_maps.png'}")

    # Plot 3: RFI example error maps
    n_rfi_ex = min(3, len(rfi_idx))
    if n_rfi_ex > 0:
        fig, axes = plt.subplots(n_rfi_ex, 3, figsize=(15, 5 * n_rfi_ex))
        if n_rfi_ex == 1:
            axes = [axes]
        sorted_rfi = rfi_idx[np.argsort(hot_fracs[rfi_idx])]
        for row, i in enumerate(sorted_rfi[-n_rfi_ex:]):
            snip = preprocessed[i]
            err = compute_error_map(model, snip, args.device)
            vmin, vmax = np.percentile(snip, [1, 99])
            axes[row][0].imshow(snip, aspect="auto", origin="lower",
                                cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row][0].set_title(f"RFI snippet (hot_frac={hot_fracs[i]:.4f})", fontsize=9)
            axes[row][1].imshow(err, aspect="auto", origin="lower",
                                cmap="hot", vmin=0, vmax=np.percentile(err, 99))
            axes[row][1].set_title(f"Reconstruction error (MSE={err.mean():.3f})", fontsize=9)
            hot = snip > 5.0
            overlay = np.zeros((*snip.shape, 3))
            overlay[..., 0] = hot.astype(float)
            axes[row][2].imshow(snip, aspect="auto", origin="lower",
                                cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row][2].imshow(overlay, aspect="auto", origin="lower", alpha=0.4)
            axes[row][2].set_title("Hot pixels overlay (red)", fontsize=9)
            for ax in axes[row]:
                ax.set_ylabel("time bin")
                ax.set_xlabel("freq channel")
        plt.tight_layout()
        plt.savefig(args.out_dir / "rfi_error_maps.png", dpi=150)
        plt.close()
        print(f"Saved → {args.out_dir / 'rfi_error_maps.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
