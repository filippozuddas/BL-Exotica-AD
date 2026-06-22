"""
SNR sweep for SSAST ViT-MAE recon anomaly score.

Evaluates the reconstruction-based anomaly score across a range of SNR values
to characterize detection sensitivity (Phase-2 injection-recovery). Uses the
partitioned reconstruction forward pass — the empirically validated scoring
method (recon separates Inject-ON from Quiet/RFI at SNR=25, embedding does not).

Outputs:
  - sensitivity_curve.png: detection rate vs SNR at various thresholds
  - snr_sweep_distributions.png: score distributions per SNR
  - snr_sweep_results.npz: raw data for further analysis

Usage (run on the server, not the dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/ssast_snr_sweep.py \
        --checkpoint outputs/<run>/checkpoints/last.ckpt \
        --cache /path/to/cache_gbt_fine \
        --out_dir outputs/ssast_snr_sweep
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

from src.models.autoencoder import build_autoencoder
from scripts.debug.injection_vs_rfi_test import (
    preprocess_raw,
    inject_narrowband_on_only,
)

INPUT_SHAPE = (96, 1024, 1)


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def recon_score(model, snip: np.ndarray, device: str) -> float:
    x = torch.from_numpy(snip).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model.anomaly_score(x, method="recon")
    return float(score.item())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/ssast_snr_sweep")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    indices = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[indices])
    del arr
    print(f"  Raw snippets: {raw_snippets.shape}")

    # Preprocess and categorise
    preprocessed, hot_fracs = [], []
    for i in range(len(raw_snippets)):
        snip = preprocess_raw(raw_snippets[i], preproc)
        preprocessed.append(snip)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)
    preprocessed = np.array(preprocessed)
    hot_fracs = np.array(hot_fracs)

    q25 = np.percentile(hot_fracs, 25)
    quiet_idx = np.where(hot_fracs <= q25)[0]
    print(f"  Quiet snippets: {len(quiet_idx)}")

    # Baseline: recon score on quiet snippets (no injection)
    print("Computing baseline scores on quiet snippets...")
    baseline_scores = np.array([recon_score(model, preprocessed[i], args.device)
                                for i in quiet_idx])
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    print(f"  Baseline recon score: {baseline_mean:.4f} ± {baseline_std:.4f}")

    # RFI scores
    q75 = np.percentile(hot_fracs, 75)
    rfi_idx = np.where(hot_fracs >= q75)[0]
    rfi_scores = np.array([recon_score(model, preprocessed[i], args.device)
                           for i in rfi_idx])
    print(f"  RFI recon score: {rfi_scores.mean():.4f} ± {rfi_scores.std():.4f}")

    # SNR sweep: inject ON-only into quiet snippets
    n_inject = len(quiet_idx)
    print(f"\nSNR sweep ({n_inject} injections per SNR): {args.snr_list}")
    print(f"  {'SNR':>5s}  {'mean':>8s}  {'std':>8s}  {'vs base':>8s}  "
          f"{'sigma':>8s}  {'det@3σ':>8s}  {'det@5σ':>8s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    all_inject_scores = {}
    for snr in args.snr_list:
        scores = []
        for j, i in enumerate(quiet_idx):
            raw_inj = inject_narrowband_on_only(
                raw_snippets[i], snr=snr, drift_rate=args.drift_rate,
                seed=args.seed + j)
            snip_inj = preprocess_raw(raw_inj, preproc)
            scores.append(recon_score(model, snip_inj, args.device))
        scores = np.array(scores)
        all_inject_scores[snr] = scores

        mean_s = scores.mean()
        std_s = scores.std()
        ratio = mean_s / baseline_mean
        sigma = (mean_s - baseline_mean) / baseline_std if baseline_std > 0 else 0
        thresh_3s = baseline_mean + 3 * baseline_std
        thresh_5s = baseline_mean + 5 * baseline_std
        det_3s = (scores > thresh_3s).mean() * 100
        det_5s = (scores > thresh_5s).mean() * 100

        print(f"  {snr:5.0f}  {mean_s:8.4f}  {std_s:8.4f}  {ratio:8.3f}x  "
              f"{sigma:8.2f}σ  {det_3s:7.1f}%  {det_5s:7.1f}%")

    # Save raw results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "snr_sweep_results.npz",
             snr_list=np.array(args.snr_list),
             baseline_scores=baseline_scores,
             rfi_scores=rfi_scores,
             **{f"inject_snr_{int(s)}": v for s, v in all_inject_scores.items()})

    # --- Plot 1: Sensitivity curve ---
    snrs = sorted(all_inject_scores.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SSAST ViT-MAE recon score — SNR sensitivity (ON-only injection)", fontsize=12)

    # Left: mean score vs SNR
    means = [all_inject_scores[s].mean() for s in snrs]
    stds = [all_inject_scores[s].std() for s in snrs]
    ax1.errorbar(snrs, means, yerr=stds, marker="o", color="crimson",
                 capsize=3, label="Inject-ON")
    ax1.axhline(baseline_mean, ls="--", color="steelblue",
                label=f"Quiet baseline ({baseline_mean:.2f})")
    ax1.fill_between(snrs, baseline_mean - baseline_std, baseline_mean + baseline_std,
                     alpha=0.15, color="steelblue")
    ax1.axhline(rfi_scores.mean(), ls="--", color="orange",
                label=f"RFI ({rfi_scores.mean():.2f})")
    ax1.set_xlabel("Injection SNR")
    ax1.set_ylabel("Recon anomaly score (MSE)")
    ax1.set_title("Mean recon score vs SNR")
    ax1.legend(fontsize=8)

    # Right: detection rate vs SNR
    for n_sigma, color, ls in [(3, "crimson", "-"), (5, "darkred", "--")]:
        thresh = baseline_mean + n_sigma * baseline_std
        det_rates = [(all_inject_scores[s] > thresh).mean() * 100 for s in snrs]
        ax2.plot(snrs, det_rates, f"{ls}", color=color, marker="o",
                 label=f"Detection @ {n_sigma}σ (thresh={thresh:.2f})")
    ax2.axhline(100, ls=":", color="gray", alpha=0.3)
    ax2.axhline(0, ls=":", color="gray", alpha=0.3)
    ax2.set_xlabel("Injection SNR")
    ax2.set_ylabel("Detection rate (%)")
    ax2.set_title("Detection rate vs SNR")
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out_dir / "sensitivity_curve.png", dpi=150)
    plt.close()
    print(f"\nSaved → {args.out_dir / 'sensitivity_curve.png'}")

    # --- Plot 2: Score distributions per SNR (box plot) ---
    fig, ax = plt.subplots(figsize=(max(10, len(snrs) * 0.8 + 3), 5))
    box_data = [baseline_scores, rfi_scores] + [all_inject_scores[s] for s in snrs]
    box_labels = ["Quiet", "RFI"] + [f"SNR={int(s)}" for s in snrs]
    box_colors = ["steelblue", "orange"] + ["crimson"] * len(snrs)

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(baseline_mean + 3 * baseline_std, ls="--", color="gray",
               alpha=0.5, label=f"3σ threshold ({baseline_mean + 3 * baseline_std:.2f})")
    ax.set_ylabel("Recon anomaly score (MSE)")
    ax.set_title("Score distributions: quiet vs RFI vs injected at various SNR")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "snr_sweep_distributions.png", dpi=150)
    plt.close()
    print(f"Saved → {args.out_dir / 'snr_sweep_distributions.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
