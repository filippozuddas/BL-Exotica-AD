"""
Analyze false-positive structure from inference CSV.

Reports:
  1. Per-cadence FP breakdown (which cadences dominate?)
  2. Candidate counts at increasing sigma thresholds (10σ, 20σ, 50σ, 100σ)
  3. ON/OFF peak-column ratio for top candidates (how many survive r>4 cut?)
  4. Score distribution plots per cadence

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/BL-Exotica-AD \
    python scripts/debug/analyze_inference_fp.py \
        --csv outputs/inference/full_8cad_run/inference_scores.csv \
        --checkpoint outputs/training/.../checkpoints/best.ckpt \
        --cadence_list data/processed/cache_gbt_fine/inference_cadences.txt \
        --out_dir outputs/inference/full_8cad_run/fp_analysis
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
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
from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs

INPUT_SHAPE = (96, 1024, 1)
MAD_SCALE = 1.4826


def robust_stats(scores):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    sigma = mad * MAD_SCALE
    return median, sigma


def load_csv(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "cadence_idx": int(r["cadence_idx"]),
                "target": r["target"],
                "f_start": int(r["f_start"]),
                "f_center_mhz": float(r["f_center_mhz"]),
                "recon_score": float(r["recon_score"]),
                "cadence_score": float(r["cadence_score"]),
            })
    return rows


def _score_key(method: str) -> str:
    return f"{method}_score"


def per_cadence_breakdown(rows, out_dir, method="cadence"):
    score_key = _score_key(method)
    print("\n" + "=" * 70)
    print(f"1. PER-CADENCE FALSE-POSITIVE BREAKDOWN  (method={method})")
    print("=" * 70)

    by_cad = defaultdict(list)
    for r in rows:
        by_cad[r["cadence_idx"]].append(r)

    all_scores = np.array([r[score_key] for r in rows])
    global_med, global_sigma = robust_stats(all_scores)

    print(f"\nGlobal: median={global_med:.4f}  MAD_σ={global_sigma:.4f}")
    print(f"{'Cad':>4} {'Target':>20} {'N':>8} {'Med':>8} {'MAD_σ':>8} "
          f"{'3σ_loc':>8} {'5σ_loc':>8} {'3σ_glb':>8} {'5σ_glb':>8}")
    print("-" * 100)

    cad_stats = {}
    for cad_idx in sorted(by_cad.keys()):
        cad_rows = by_cad[cad_idx]
        target = cad_rows[0]["target"]
        scores = np.array([r[score_key] for r in cad_rows])
        med, sig = robust_stats(scores)

        n_3s_local = (scores > med + 3 * sig).sum()
        n_5s_local = (scores > med + 5 * sig).sum()
        n_3s_global = (scores > global_med + 3 * global_sigma).sum()
        n_5s_global = (scores > global_med + 5 * global_sigma).sum()

        cad_stats[cad_idx] = {"median": med, "sigma": sig, "n": len(scores)}

        print(f"{cad_idx:>4} {target:>20} {len(scores):>8} {med:>8.4f} {sig:>8.4f} "
              f"{n_3s_local:>8} {n_5s_local:>8} {n_3s_global:>8} {n_5s_global:>8}")

    return by_cad, cad_stats


def sigma_threshold_sweep(rows, out_dir, method="cadence"):
    score_key = _score_key(method)
    print("\n" + "=" * 70)
    print(f"2. CANDIDATE COUNTS AT INCREASING SIGMA THRESHOLDS  (method={method})")
    print("=" * 70)

    scores = np.array([r[score_key] for r in rows])
    med, sig = robust_stats(scores)
    n_total = len(scores)

    thresholds = [3, 5, 10, 20, 50, 100, 200, 500]
    print(f"\nMedian={med:.4f}  MAD_σ={sig:.4f}  N_total={n_total}")
    print(f"{'σ':>6} {'Threshold':>12} {'N_cand':>10} {'%':>8}")
    print("-" * 40)
    for k in thresholds:
        thresh = med + k * sig
        n = (scores > thresh).sum()
        pct = n / n_total * 100
        print(f"{k:>6} {thresh:>12.4f} {n:>10} {pct:>8.3f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = [(k, (scores > med + k * sig).sum()) for k in range(1, 501)]
    ks, ns = zip(*counts)
    ax.semilogy(ks, ns)
    ax.set_xlabel("Sigma threshold (k)")
    ax.set_ylabel("Number of candidates")
    ax.set_title(f"Candidates vs sigma threshold — {method} (N_total={n_total})")
    ax.axhline(100, color="green", ls="--", alpha=0.5, label="100 candidates")
    ax.axhline(1000, color="orange", ls="--", alpha=0.5, label="1000 candidates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "candidates_vs_sigma.png", dpi=150)
    plt.close()
    print(f"\nSaved -> {out_dir / 'candidates_vs_sigma.png'}")


def per_cadence_distributions(rows, by_cad, out_dir, method="cadence"):
    score_key = _score_key(method)
    print("\n" + "=" * 70)
    print(f"3. PER-CADENCE SCORE DISTRIBUTIONS  (method={method})")
    print("=" * 70)

    n_cads = len(by_cad)
    fig, axes = plt.subplots(2, (n_cads + 1) // 2, figsize=(5 * ((n_cads + 1) // 2), 8))
    axes = axes.flatten()

    for i, cad_idx in enumerate(sorted(by_cad.keys())):
        cad_rows = by_cad[cad_idx]
        scores = np.array([r[score_key] for r in cad_rows])
        target = cad_rows[0]["target"]
        med, sig = robust_stats(scores)

        clipped = scores[scores < np.percentile(scores, 99.5)]
        axes[i].hist(clipped, bins=150, alpha=0.7, edgecolor="black", linewidth=0.2)
        axes[i].axvline(med + 3 * sig, color="orange", ls="--", lw=1, label=f"3σ={med + 3*sig:.3f}")
        axes[i].axvline(med + 10 * sig, color="red", ls="--", lw=1, label=f"10σ={med + 10*sig:.3f}")
        axes[i].set_title(f"cad{cad_idx} {target[:15]}", fontsize=9)
        axes[i].legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{method} score distributions per cadence", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "per_cadence_distributions.png", dpi=150)
    plt.close()
    print(f"Saved -> {out_dir / 'per_cadence_distributions.png'}")


def onoff_ratio_analysis(rows, model, cadence_lines, data_cfg, device, out_dir,
                         n_top=500, top_k_cols=5, method="cadence"):
    score_key = _score_key(method)
    print("\n" + "=" * 70)
    print(f"4. ON/OFF PEAK-COLUMN RATIO ON TOP CANDIDATES  (method={method})")
    print("=" * 70)

    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    tchans = frame["tchans"]
    downsample_factor = frame.get("downsample_factor", 1)
    n_obs = 6
    rows_per_obs = tchans // n_obs  # 16

    scores = np.array([r[score_key] for r in rows])
    med, sig = robust_stats(scores)

    sorted_idx = np.argsort(scores)[::-1][:n_top]
    top_rows = [rows[i] for i in sorted_idx]

    # Group by cadence_idx for efficient loading
    by_cad = defaultdict(list)
    for r in top_rows:
        by_cad[r["cadence_idx"]].append(r)

    # Load obs data per cadence and compute ratios
    ratios = []
    obs_cache = {}

    for cad_idx in sorted(by_cad.keys()):
        cad_candidates = by_cad[cad_idx]

        if cad_idx not in obs_cache:
            obs_paths = [Path(p) for p in cadence_lines[cad_idx]]
            obs_arrays = []
            skip = False
            for obs_path in obs_paths:
                try:
                    arr = _load_full_obs(obs_path, downsample_factor)
                    obs_arrays.append(arr)
                except OSError:
                    skip = True
                    break
            if skip:
                continue
            obs_cache[cad_idx] = obs_arrays

        obs_arrays = obs_cache[cad_idx]

        for r in cad_candidates:
            f_start = r["f_start"]
            # Build snippet
            frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
            stacked = np.concatenate(frames, axis=0)[:tchans, :]
            method = preproc.get("bandpass_method", "polynomial")
            poly_degree = preproc.get("poly_degree", 3)
            mad_eps = preproc.get("mad_epsilon", 1e-6)
            stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
            stacked = core_transform(stacked, mad_eps)

            # Get reconstruction
            x = torch.from_numpy(stacked).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                recon = model(x).squeeze(0).squeeze(0).cpu().numpy()

            residual = np.abs(stacked - recon)

            # Peak-column ratio: top_k_cols by residual energy
            col_energy = residual.sum(axis=0)
            peak_cols = np.argsort(col_energy)[-top_k_cols:]

            on_rows = []
            off_rows = []
            for obs_i in range(n_obs):
                start = obs_i * rows_per_obs
                end = start + rows_per_obs
                if obs_i in (0, 2, 4):
                    on_rows.extend(range(start, end))
                else:
                    off_rows.extend(range(start, end))

            res_on = residual[on_rows][:, peak_cols]
            res_off = residual[off_rows][:, peak_cols]

            mse_on = np.mean(res_on ** 2)
            mse_off = np.mean(res_off ** 2) + 1e-10
            ratio = mse_on / mse_off

            ratios.append({
                "cadence_idx": r["cadence_idx"],
                "target": r["target"],
                "f_start": r["f_start"],
                "score": r[score_key],
                "sigma": (r[score_key] - med) / sig,
                "ratio": ratio,
            })

        # Free memory
        del obs_cache[cad_idx]

    ratios_arr = np.array([r["ratio"] for r in ratios])
    sigmas_arr = np.array([r["sigma"] for r in ratios])

    print(f"\nAnalyzed {len(ratios)} top candidates")
    print(f"Ratio distribution: min={ratios_arr.min():.2f}  median={np.median(ratios_arr):.2f}  "
          f"max={ratios_arr.max():.2f}")

    # Survival counts at different ratio cuts
    print(f"\n{'r_cut':>8} {'Survive':>10} {'Rejected':>10} {'%_rejected':>12}")
    print("-" * 45)
    for r_cut in [1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        survive = (ratios_arr > r_cut).sum()
        rejected = len(ratios_arr) - survive
        pct = rejected / len(ratios_arr) * 100
        print(f"{r_cut:>8.1f} {survive:>10} {rejected:>10} {pct:>11.1f}%")

    # Scatter: sigma vs ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sigmas_arr, ratios_arr, s=8, alpha=0.5)
    ax.axhline(4.0, color="red", ls="--", lw=1, label="r=4 cut")
    ax.axhline(2.0, color="orange", ls="--", lw=1, label="r=2 cut")
    ax.set_xlabel(f"{method} score (σ above median)")
    ax.set_ylabel("ON/OFF peak-column ratio")
    ax.set_title(f"Top {len(ratios)} candidates ({method}): sigma vs ON/OFF ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sigma_vs_ratio_scatter.png", dpi=150)
    plt.close()
    print(f"\nSaved -> {out_dir / 'sigma_vs_ratio_scatter.png'}")

    # Ratio histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios_arr[ratios_arr < np.percentile(ratios_arr, 99)], bins=100,
            alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(4.0, color="red", ls="--", lw=1.5, label="r=4 cut")
    ax.set_xlabel("ON/OFF peak-column ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"Ratio distribution (top {len(ratios)} candidates)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_histogram.png", dpi=150)
    plt.close()
    print(f"Saved -> {out_dir / 'ratio_histogram.png'}")

    # Save CSV
    csv_path = out_dir / "top_candidates_with_ratio.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cadence_idx", "target", "f_start",
                                                "score", "sigma", "ratio"])
        writer.writeheader()
        writer.writerows(ratios)
    print(f"Saved -> {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True,
                   help="inference_scores.csv from inference.py")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Model checkpoint (needed for ON/OFF ratio analysis)")
    p.add_argument("--cadence_list", type=Path, default=None,
                   help="Cadence list file (needed for ON/OFF ratio analysis)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_top", type=int, default=500,
                   help="Number of top candidates to analyze for ON/OFF ratio")
    p.add_argument("--method", default="cadence", choices=["recon", "cadence"],
                   help="Which score column to analyze (default: cadence)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    method = args.method
    print(f"Loading scores from {args.csv}")
    print(f"Scoring method: {method}")
    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} snippets")

    # 1. Per-cadence breakdown
    by_cad, cad_stats = per_cadence_breakdown(rows, args.out_dir, method=method)

    # 2. Sigma threshold sweep
    sigma_threshold_sweep(rows, args.out_dir, method=method)

    # 3. Per-cadence distributions
    per_cadence_distributions(rows, by_cad, args.out_dir, method=method)

    # 4. ON/OFF ratio analysis (if checkpoint + cadence_list provided)
    if args.checkpoint and args.cadence_list:
        with open(args.data_config) as f:
            data_cfg = yaml.safe_load(f)
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)

        print(f"\nLoading model from {args.checkpoint}")
        model = build_autoencoder(INPUT_SHAPE, model_cfg, loss="mse")
        ckpt = torch.load(str(args.checkpoint), map_location=args.device, weights_only=False)
        state = {k.replace("model.", "", 1): v
                 for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state)
        model.eval().to(args.device)

        cadence_lines = [
            line.strip().split()
            for line in args.cadence_list.read_text().splitlines()
            if line.strip()
        ]

        onoff_ratio_analysis(rows, model, cadence_lines, data_cfg, args.device,
                             args.out_dir, n_top=args.n_top, method=method)
    else:
        print("\nSkipping ON/OFF ratio analysis (no --checkpoint / --cadence_list)")

    print("\nDone.")


if __name__ == "__main__":
    main()
