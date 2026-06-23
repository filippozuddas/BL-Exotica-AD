"""
Run inference on real cadences from a cadence list file.

Loads each cadence's .h5 files fully into RAM, slides a (tchans, fchans)
window across the frequency axis at stride_infer, preprocesses each snippet,
computes cadence + recon anomaly scores, and reports candidates above threshold.

Each cadence's 6 observations are loaded once (~4 GB/obs for 67M channels);
snippets are sliced from the in-memory arrays (no per-snippet I/O).

Outputs:
  - inference_scores.csv: per-snippet scores
  - inference_score_distributions.png: global histograms
  - inference_cadence_scores_by_freq.png: per-cadence frequency profiles

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/acabras/data/filippo/BL-Exotica-AD \
    python scripts/inference.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --cadence_list data/processed/inference_cadences.txt \
        --out_dir outputs/inference/run_name
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import build_autoencoder
from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs, _read_nchans

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


def preprocess_snippet(stacked: np.ndarray, preproc_cfg: dict) -> np.ndarray:
    method = preproc_cfg.get("bandpass_method", "polynomial")
    poly_degree = preproc_cfg.get("poly_degree", 3)
    mad_epsilon = preproc_cfg.get("mad_epsilon", 1e-6)
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    stacked = core_transform(stacked, mad_epsilon)
    return stacked


def score_batch(model, snippets: list, method: str, device: str) -> np.ndarray:
    x = torch.from_numpy(np.array(snippets)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        s = model.anomaly_score(x, method=method)
    return s.cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True,
                   help="Text file with one cadence per line (6 space-separated .h5 paths)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=None,
                   help="Limit number of cadences to process (default: all)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of snippets to batch for GPU scoring")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    tchans = frame["tchans"]
    stride = frame.get("stride_infer", fchans // 2)
    downsample_factor = frame.get("downsample_factor", 1)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    print(f"Cadences: {len(cadence_lines)}")
    print(f"Frame: {tchans}x{fchans}, stride={stride}, downsample={downsample_factor}")
    print(f"Batch size: {args.batch_size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_recon = []
    all_cadence = []
    all_rows = []

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]
        target_name = "unknown"
        for p in obs_paths:
            parts = p.stem.split("_")
            for part in parts:
                if part.startswith("TIC"):
                    target_name = part
                    break
            if target_name != "unknown":
                break

        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx}: {target_name} ({len(obs_paths)} obs)")
        print(f"{'='*70}")

        # Load all observations into RAM once
        t_load = time.time()
        obs_arrays = []
        for i, obs_path in enumerate(obs_paths):
            arr = _load_full_obs(obs_path, downsample_factor)
            obs_arrays.append(arr)
            print(f"  Loaded obs {i}: {obs_path.name} → {arr.shape}")
        load_time = time.time() - t_load

        nchans = obs_arrays[0].shape[1]
        n_snippets = max(0, (nchans - fchans) // stride + 1)
        mem_gb = sum(a.nbytes for a in obs_arrays) / 1e9
        print(f"  {mem_gb:.1f} GB in RAM, loaded in {load_time:.1f}s")
        print(f"  nchans={nchans} -> {n_snippets} snippets (stride={stride})")

        t0 = time.time()
        batch_snippets = []
        batch_fstarts = []

        for snip_idx in range(n_snippets):
            f_start = snip_idx * stride
            f_end = f_start + fchans

            # Slice from in-memory arrays and stack
            frames = [obs[:, f_start:f_end] for obs in obs_arrays]
            stacked = np.concatenate(frames, axis=0)
            snip = preprocess_snippet(stacked, preproc)

            batch_snippets.append(snip)
            batch_fstarts.append(f_start)

            if len(batch_snippets) == args.batch_size or snip_idx == n_snippets - 1:
                recon_scores = score_batch(model, batch_snippets, "recon", args.device)
                cadence_scores = score_batch(model, batch_snippets, "cadence", args.device)

                for j in range(len(batch_snippets)):
                    all_recon.append(recon_scores[j])
                    all_cadence.append(cadence_scores[j])
                    all_rows.append({
                        "cadence_idx": cad_idx,
                        "target": target_name,
                        "f_start": batch_fstarts[j],
                        "f_center_mhz": batch_fstarts[j] * data_cfg["raw"]["df"] / 1e6,
                        "recon_score": float(recon_scores[j]),
                        "cadence_score": float(cadence_scores[j]),
                    })
                batch_snippets = []
                batch_fstarts = []

            if (snip_idx + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (snip_idx + 1) / elapsed
                eta = (n_snippets - snip_idx - 1) / rate
                print(f"  {snip_idx+1}/{n_snippets} snippets  "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)")

        elapsed = time.time() - t0
        print(f"  Scored {n_snippets} snippets in {elapsed:.1f}s "
              f"({n_snippets/max(elapsed,1):.0f}/s)")

        # Free memory before next cadence
        del obs_arrays

    # Global analysis
    all_recon = np.array(all_recon)
    all_cadence = np.array(all_cadence)
    n_total = len(all_recon)

    # Save CSV
    csv_path = args.out_dir / "inference_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cadence_idx", "target", "f_start",
                                                "f_center_mhz", "recon_score", "cadence_score"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved scores -> {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY -- {n_total} snippets across {len(cadence_lines)} cadences")
    print(f"{'='*70}")

    for name, scores in [("recon", all_recon), ("cadence", all_cadence)]:
        mean = scores.mean()
        std = scores.std()
        median = np.median(scores)
        thresh_3s = mean + 3 * std
        thresh_5s = mean + 5 * std
        n_3s = (scores > thresh_3s).sum()
        n_5s = (scores > thresh_5s).sum()

        print(f"\n  {name}:")
        print(f"    mean={mean:.4f}  std={std:.4f}  median={median:.4f}")
        print(f"    min={scores.min():.4f}  max={scores.max():.4f}")
        print(f"    3s={thresh_3s:.4f}  -> {n_3s} candidates ({n_3s/n_total*100:.3f}%)")
        print(f"    5s={thresh_5s:.4f}  -> {n_5s} candidates ({n_5s/n_total*100:.3f}%)")

        if n_3s > 0 and n_3s <= 50:
            idx_above = np.where(scores > thresh_3s)[0]
            idx_sorted = idx_above[np.argsort(scores[idx_above])[::-1]]
            print(f"    Top candidates:")
            for idx in idx_sorted[:20]:
                row = all_rows[idx]
                sigma = (scores[idx] - mean) / std
                print(f"      cad={row['cadence_idx']} ({row['target']})  "
                      f"f_start={row['f_start']:>8d}  "
                      f"score={scores[idx]:.4f} ({sigma:.1f}s)")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, scores) in zip(axes, [("recon", all_recon), ("cadence", all_cadence)]):
        mean, std = scores.mean(), scores.std()
        ax.hist(scores, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(mean + 3 * std, color="orange", ls="--", lw=1.5,
                   label=f"3s = {mean + 3*std:.3f}")
        ax.axvline(mean + 5 * std, color="red", ls="--", lw=1.5,
                   label=f"5s = {mean + 5*std:.3f}")
        n_3s = (scores > mean + 3 * std).sum()
        ax.set_xlabel("Anomaly score (MSE)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} -- {n_3s} candidates > 3s")
        ax.legend()

    plt.suptitle(f"Score distribution -- {n_total} snippets, {len(cadence_lines)} cadences",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inference_score_distributions.png", dpi=150)
    plt.close()
    print(f"\nSaved -> {args.out_dir / 'inference_score_distributions.png'}")

    # Per-cadence frequency profile
    fig, axes = plt.subplots(len(cadence_lines), 1,
                             figsize=(16, 3 * len(cadence_lines)), squeeze=False)
    offset = 0
    for cad_idx in range(len(cadence_lines)):
        cad_rows = [r for r in all_rows if r["cadence_idx"] == cad_idx]
        n_cad = len(cad_rows)
        cad_scores = all_cadence[offset:offset + n_cad]
        fstarts = [r["f_start"] for r in cad_rows]

        ax = axes[cad_idx, 0]
        ax.plot(fstarts, cad_scores, linewidth=0.3, alpha=0.7)
        mean, std = all_cadence.mean(), all_cadence.std()
        ax.axhline(mean + 3 * std, color="orange", ls="--", lw=1, alpha=0.7, label="3s")
        ax.axhline(mean + 5 * std, color="red", ls="--", lw=1, alpha=0.7, label="5s")
        ax.set_ylabel("Cadence score")
        ax.set_title(f"Cadence {cad_idx}: {cad_rows[0]['target']}")
        ax.legend(fontsize=7)
        offset += n_cad

    axes[-1, 0].set_xlabel("Frequency channel (f_start)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "inference_cadence_scores_by_freq.png", dpi=150)
    plt.close()
    print(f"Saved -> {args.out_dir / 'inference_cadence_scores_by_freq.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
