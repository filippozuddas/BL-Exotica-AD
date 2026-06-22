"""
SNR sweep comparing cadence-aware vs partitioned-recon anomaly scores.

Evaluates both scoring methods side-by-side across a range of SNR values to
determine whether cadence-aware masking (mask-ON/reconstruct-from-OFF) improves
detection sensitivity over the standard partitioned reconstruction.

Outputs:
  - cadence_vs_recon_sensitivity.png: detection rate vs SNR for both methods
  - cadence_vs_recon_distributions.png: score distributions per SNR
  - cadence_snr_sweep_results.npz: raw data for further analysis

Usage (run on the server, not the dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/cadence_snr_sweep.py \
        --checkpoint outputs/<run>/checkpoints/last.ckpt \
        --cache /path/to/cache_gbt_fine.npz \
        --out_dir outputs/cadence_snr_sweep
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
METHODS = ["recon", "cadence"]


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def score_snippet(model, snip: np.ndarray, method: str, device: str) -> float:
    x = torch.from_numpy(snip).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        s = model.anomaly_score(x, method=method)
    return float(s.item())


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
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/cadence_snr_sweep")
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

    print(f"Loading NPZ: {args.cache}")
    archive = np.load(str(args.cache), mmap_mode="r")
    arr = archive[args.split]
    indices = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[indices])
    del arr, archive
    print(f"  Raw snippets: {raw_snippets.shape}")

    preprocessed, hot_fracs = [], []
    for i in range(len(raw_snippets)):
        snip = preprocess_raw(raw_snippets[i], preproc)
        preprocessed.append(snip)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)
    preprocessed = np.array(preprocessed)
    hot_fracs = np.array(hot_fracs)

    q25 = np.percentile(hot_fracs, 25)
    quiet_idx = np.where(hot_fracs <= q25)[0]
    q75 = np.percentile(hot_fracs, 75)
    rfi_idx = np.where(hot_fracs >= q75)[0]

    # ---- Baseline scores for both methods ----
    results = {}
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        baseline = np.array([score_snippet(model, preprocessed[i], method, args.device)
                             for i in quiet_idx])
        rfi = np.array([score_snippet(model, preprocessed[i], method, args.device)
                        for i in rfi_idx])
        b_mean, b_std = baseline.mean(), baseline.std()
        print(f"  Quiet baseline: {b_mean:.4f} ± {b_std:.4f}")
        print(f"  RFI:            {rfi.mean():.4f} ± {rfi.std():.4f}")

        print(f"\n  {'SNR':>5s}  {'mean':>8s}  {'std':>8s}  {'vs base':>8s}  "
              f"{'sigma':>8s}  {'det@3σ':>8s}  {'det@5σ':>8s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        inject_scores = {}
        for snr in args.snr_list:
            scores = []
            for j, i in enumerate(quiet_idx):
                raw_inj = inject_narrowband_on_only(
                    raw_snippets[i], snr=snr, drift_rate=args.drift_rate,
                    seed=args.seed + j)
                snip_inj = preprocess_raw(raw_inj, preproc)
                scores.append(score_snippet(model, snip_inj, method, args.device))
            scores = np.array(scores)
            inject_scores[snr] = scores

            mean_s, std_s = scores.mean(), scores.std()
            ratio = mean_s / b_mean if b_mean > 0 else 0
            sigma = (mean_s - b_mean) / b_std if b_std > 0 else 0
            thresh_3s = b_mean + 3 * b_std
            thresh_5s = b_mean + 5 * b_std
            det_3s = (scores > thresh_3s).mean() * 100
            det_5s = (scores > thresh_5s).mean() * 100
            print(f"  {snr:5.0f}  {mean_s:8.4f}  {std_s:8.4f}  {ratio:8.3f}x  "
                  f"{sigma:8.2f}σ  {det_3s:7.1f}%  {det_5s:7.1f}%")

        results[method] = {
            "baseline": baseline, "rfi": rfi, "inject": inject_scores,
            "b_mean": b_mean, "b_std": b_std,
        }

    # ---- Save raw results ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {"snr_list": np.array(args.snr_list)}
    for method in METHODS:
        r = results[method]
        save_dict[f"{method}_baseline"] = r["baseline"]
        save_dict[f"{method}_rfi"] = r["rfi"]
        for snr, v in r["inject"].items():
            save_dict[f"{method}_inject_snr_{int(snr)}"] = v
    np.savez(args.out_dir / "cadence_snr_sweep_results.npz", **save_dict)

    # ---- Plot 1: Sensitivity curves (side-by-side) ----
    snrs = sorted(args.snr_list)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cadence vs Recon scoring — SNR sensitivity (ON-only injection)", fontsize=12)

    colors = {"recon": "steelblue", "cadence": "crimson"}
    for method in METHODS:
        r = results[method]
        means = [r["inject"][s].mean() for s in snrs]
        stds = [r["inject"][s].std() for s in snrs]

        ax = axes[0]
        ax.errorbar(snrs, means, yerr=stds, marker="o", color=colors[method],
                     capsize=3, label=f"{method} inject-ON")
        ax.axhline(r["b_mean"], ls="--", color=colors[method], alpha=0.5,
                    label=f"{method} quiet ({r['b_mean']:.2f})")

    axes[0].set_xlabel("Injection SNR")
    axes[0].set_ylabel("Anomaly score (MSE)")
    axes[0].set_title("Mean score vs SNR")
    axes[0].legend(fontsize=7)

    for method in METHODS:
        r = results[method]
        for n_sigma, ls in [(3, "-"), (5, "--")]:
            thresh = r["b_mean"] + n_sigma * r["b_std"]
            det_rates = [(r["inject"][s] > thresh).mean() * 100 for s in snrs]
            axes[1].plot(snrs, det_rates, ls, color=colors[method], marker="o",
                         label=f"{method} @ {n_sigma}σ", markersize=4)

    axes[1].set_xlabel("Injection SNR")
    axes[1].set_ylabel("Detection rate (%)")
    axes[1].set_title("Detection rate vs SNR")
    axes[1].set_ylim(-5, 105)
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(args.out_dir / "cadence_vs_recon_sensitivity.png", dpi=150)
    plt.close()
    print(f"\nSaved → {args.out_dir / 'cadence_vs_recon_sensitivity.png'}")

    # ---- Plot 2: Score distributions (box plot, both methods) ----
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(max(12, len(snrs) + 4), 5 * len(METHODS)),
                             sharex=True)
    for ax, method in zip(axes, METHODS):
        r = results[method]
        box_data = [r["baseline"], r["rfi"]] + [r["inject"][s] for s in snrs]
        box_labels = ["Quiet", "RFI"] + [f"SNR={int(s)}" for s in snrs]
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.7)
        ax.axhline(r["b_mean"] + 3 * r["b_std"], ls="--", color="gray", alpha=0.5,
                    label=f"3σ ({r['b_mean'] + 3 * r['b_std']:.2f})")
        ax.set_ylabel("Anomaly score (MSE)")
        ax.set_title(f"{method} — score distributions")
        ax.legend(fontsize=8)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "cadence_vs_recon_distributions.png", dpi=150)
    plt.close()
    print(f"Saved → {args.out_dir / 'cadence_vs_recon_distributions.png'}")

    # ---- Summary comparison ----
    print(f"\n{'='*60}")
    print("SUMMARY: cadence vs recon at key SNR thresholds")
    print(f"{'='*60}")
    print(f"  {'SNR':>5s}  {'recon σ':>8s}  {'recon 3σ%':>9s}  "
          f"{'cadence σ':>10s}  {'cadence 3σ%':>11s}  {'winner':>8s}")
    for snr in snrs:
        r_r, r_c = results["recon"], results["cadence"]
        sig_r = (r_r["inject"][snr].mean() - r_r["b_mean"]) / r_r["b_std"] if r_r["b_std"] > 0 else 0
        sig_c = (r_c["inject"][snr].mean() - r_c["b_mean"]) / r_c["b_std"] if r_c["b_std"] > 0 else 0
        det_r = (r_r["inject"][snr] > r_r["b_mean"] + 3 * r_r["b_std"]).mean() * 100
        det_c = (r_c["inject"][snr] > r_c["b_mean"] + 3 * r_c["b_std"]).mean() * 100
        winner = "cadence" if sig_c > sig_r else "recon" if sig_r > sig_c else "tie"
        print(f"  {snr:5.0f}  {sig_r:8.2f}σ  {det_r:8.1f}%  "
              f"{sig_c:10.2f}σ  {det_c:10.1f}%  {winner:>8s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
