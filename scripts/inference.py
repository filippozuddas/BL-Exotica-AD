"""
Run inference on real cadences from a cadence list file.

Loads each cadence's .h5 files fully into RAM, slides a (tchans, fchans)
window across the frequency axis at stride_infer, preprocesses each snippet
in parallel across CPU cores, and scores batches on GPU.

Candidate snippets above the robust threshold (median + k*MAD) are saved
with per-candidate plots showing original | reconstruction | error map.

Outputs:
  - inference_scores.csv: per-snippet scores
  - inference_score_distributions.png: global histograms
  - inference_cadence_scores_by_freq.png: per-cadence frequency profiles
  - candidates/: per-candidate 3-panel plots

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/inference.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --cadence_list data/processed/inference_cadences.txt \
        --out_dir outputs/inference/run_name \
        --num_workers 32
"""

import argparse
import csv
import multiprocessing as mp
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
from src.data.torch_dataset import _load_full_obs

INPUT_SHAPE = (96, 1024, 1)
METHODS = ["recon", "cadence"]
MAD_SCALE = 1.4826  # MAD-to-sigma conversion for Gaussian

# Shared state for worker processes (inherited via fork COW)
_shared_obs = None
_shared_fchans = None
_shared_preproc = None


def _preprocess_at(f_start):
    """Worker: slice snippet from shared obs arrays and preprocess."""
    frames = [obs[:, f_start:f_start + _shared_fchans] for obs in _shared_obs]
    stacked = np.concatenate(frames, axis=0)
    method = _shared_preproc.get("bandpass_method", "polynomial")
    poly_degree = _shared_preproc.get("poly_degree", 3)
    mad_epsilon = _shared_preproc.get("mad_epsilon", 1e-6)
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    stacked = core_transform(stacked, mad_epsilon)
    return stacked


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def score_batch(model, snippets: list, method: str, device: str) -> np.ndarray:
    x = torch.from_numpy(np.array(snippets)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        s = model.anomaly_score(x, method=method)
    return s.cpu().numpy()


def reconstruct_batch(model, snippets: list, device: str) -> np.ndarray:
    x = torch.from_numpy(np.array(snippets)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze(1).cpu().numpy()


def robust_stats(scores: np.ndarray):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    sigma = mad * MAD_SCALE
    return median, sigma


def plot_candidate(original, reconstruction, score, sigma, method, cad_idx,
                   target, f_start, df, out_path):
    error = np.abs(original - reconstruction)
    vmin, vmax = np.percentile(original, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(original, aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Original")
    axes[0].set_ylabel("Time bin")
    axes[0].set_xlabel("Freq channel")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(reconstruction, aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Reconstruction")
    axes[1].set_xlabel("Freq channel")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(error, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title("Residual |orig - recon|")
    axes[2].set_xlabel("Freq channel")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    f_center_mhz = f_start * df / 1e6
    fig.suptitle(
        f"Candidate: cad={cad_idx} ({target})  f_start={f_start}  "
        f"f~{f_center_mhz:.4f} MHz\n"
        f"{method} score={score:.4f} ({sigma:.1f}s)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


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
    p.add_argument("--max_cadences", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--top_k", type=int, default=30,
                   help="Number of top candidates to plot per method")
    return p.parse_args()


def main():
    global _shared_obs, _shared_fchans, _shared_preproc
    args = parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    tchans = frame["tchans"]
    stride = frame.get("stride_infer", fchans // 2)
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]

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
    print(f"Batch size: {args.batch_size}, workers: {args.num_workers}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_recon = []
    all_cadence = []
    all_rows = []
    # Keep top-K candidate snippets for plotting (heap by negative score)
    import heapq
    top_recon_heap = []   # (score, global_idx, snippet_array)
    top_cadence_heap = []

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]
        target_name = "unknown"
        for p in obs_paths:
            for part in p.stem.split("_"):
                if part.startswith("TIC"):
                    target_name = part
                    break
            if target_name != "unknown":
                break

        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx}: {target_name} ({len(obs_paths)} obs)")
        print(f"{'='*70}")

        t_load = time.time()
        obs_arrays = []
        for i, obs_path in enumerate(obs_paths):
            arr = _load_full_obs(obs_path, downsample_factor)
            obs_arrays.append(arr)
            print(f"  Loaded obs {i}: {obs_path.name} -> {arr.shape}")
        load_time = time.time() - t_load

        nchans = obs_arrays[0].shape[1]
        n_snippets = max(0, (nchans - fchans) // stride + 1)
        mem_gb = sum(a.nbytes for a in obs_arrays) / 1e9
        print(f"  {mem_gb:.1f} GB in RAM, loaded in {load_time:.1f}s")
        print(f"  nchans={nchans} -> {n_snippets} snippets (stride={stride})")

        _shared_obs = obs_arrays
        _shared_fchans = fchans
        _shared_preproc = preproc

        f_starts = [i * stride for i in range(n_snippets)]

        t0 = time.time()
        batch_snippets = []
        batch_fstarts = []
        processed_count = 0

        chunksize = max(1, min(256, n_snippets // (args.num_workers * 4)))
        print(f"  Pool: {args.num_workers} workers, chunksize={chunksize}")

        with mp.Pool(args.num_workers) as pool:
            for f_start, snippet in zip(f_starts, pool.imap(_preprocess_at, f_starts,
                                                             chunksize=chunksize)):
                batch_snippets.append(snippet)
                batch_fstarts.append(f_start)

                if len(batch_snippets) == args.batch_size or processed_count == n_snippets - 1:
                    recon_scores = score_batch(model, batch_snippets, "recon", args.device)
                    cadence_scores = score_batch(model, batch_snippets, "cadence", args.device)

                    for j in range(len(batch_snippets)):
                        global_idx = len(all_recon)
                        r_score = float(recon_scores[j])
                        c_score = float(cadence_scores[j])
                        all_recon.append(r_score)
                        all_cadence.append(c_score)
                        all_rows.append({
                            "cadence_idx": cad_idx,
                            "target": target_name,
                            "f_start": batch_fstarts[j],
                            "f_center_mhz": batch_fstarts[j] * df / 1e6,
                            "recon_score": r_score,
                            "cadence_score": c_score,
                        })

                        snip_copy = batch_snippets[j].copy()
                        if len(top_recon_heap) < args.top_k:
                            heapq.heappush(top_recon_heap, (r_score, global_idx, snip_copy))
                        elif r_score > top_recon_heap[0][0]:
                            heapq.heapreplace(top_recon_heap, (r_score, global_idx, snip_copy))

                        if len(top_cadence_heap) < args.top_k:
                            heapq.heappush(top_cadence_heap, (c_score, global_idx, snip_copy))
                        elif c_score > top_cadence_heap[0][0]:
                            heapq.heapreplace(top_cadence_heap, (c_score, global_idx, snip_copy))

                    batch_snippets = []
                    batch_fstarts = []

                processed_count += 1
                if processed_count % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = processed_count / elapsed
                    eta = (n_snippets - processed_count) / rate
                    print(f"  {processed_count}/{n_snippets} snippets  "
                          f"({rate:.0f}/s, ETA {eta:.0f}s)")

        elapsed = time.time() - t0
        print(f"  Scored {n_snippets} snippets in {elapsed:.1f}s "
              f"({n_snippets/max(elapsed,1):.0f}/s)")

        del obs_arrays
        _shared_obs = None

    # ---- Global analysis with robust statistics ----
    all_recon = np.array(all_recon)
    all_cadence = np.array(all_cadence)
    n_total = len(all_recon)

    csv_path = args.out_dir / "inference_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cadence_idx", "target", "f_start",
                                                "f_center_mhz", "recon_score", "cadence_score"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved scores -> {csv_path}")

    print(f"\n{'='*70}")
    print(f"SUMMARY -- {n_total} snippets across {len(cadence_lines)} cadences")
    print(f"{'='*70}")

    for name, scores in [("recon", all_recon), ("cadence", all_cadence)]:
        median, mad_sigma = robust_stats(scores)
        mean, std = scores.mean(), scores.std()
        thresh_3s = median + 3 * mad_sigma
        thresh_5s = median + 5 * mad_sigma
        n_3s = (scores > thresh_3s).sum()
        n_5s = (scores > thresh_5s).sum()

        print(f"\n  {name}:")
        print(f"    mean={mean:.4f}  std={std:.4f}")
        print(f"    median={median:.4f}  MAD_sigma={mad_sigma:.4f}")
        print(f"    min={scores.min():.4f}  max={scores.max():.4f}")
        print(f"    3s (robust)={thresh_3s:.4f}  -> {n_3s} candidates ({n_3s/n_total*100:.3f}%)")
        print(f"    5s (robust)={thresh_5s:.4f}  -> {n_5s} candidates ({n_5s/n_total*100:.3f}%)")

        if n_3s > 0 and n_3s <= 200:
            idx_above = np.where(scores > thresh_3s)[0]
            idx_sorted = idx_above[np.argsort(scores[idx_above])[::-1]]
            print(f"    Top candidates (robust 3s):")
            for idx in idx_sorted[:20]:
                row = all_rows[idx]
                sigma = (scores[idx] - median) / mad_sigma
                print(f"      cad={row['cadence_idx']} ({row['target']})  "
                      f"f_start={row['f_start']:>8d}  "
                      f"score={scores[idx]:.4f} ({sigma:.1f}s)")

    # ---- Plot candidate snippets ----
    cand_dir = args.out_dir / "candidates"
    cand_dir.mkdir(exist_ok=True)

    for method, heap in [("recon", top_recon_heap), ("cadence", top_cadence_heap)]:
        scores_arr = all_recon if method == "recon" else all_cadence
        median, mad_sigma = robust_stats(scores_arr)
        candidates = sorted(heap, key=lambda x: -x[0])

        print(f"\nPlotting top {len(candidates)} {method} candidates...")
        for rank, (score, global_idx, snippet) in enumerate(candidates):
            row = all_rows[global_idx]
            sigma = (score - median) / mad_sigma

            recon = reconstruct_batch(model, [snippet], args.device)

            out_path = cand_dir / f"{method}_rank{rank:02d}_cad{row['cadence_idx']}_f{row['f_start']}.png"
            plot_candidate(
                original=snippet,
                reconstruction=recon[0],
                score=score,
                sigma=sigma,
                method=method,
                cad_idx=row["cadence_idx"],
                target=row["target"],
                f_start=row["f_start"],
                df=df,
                out_path=out_path,
            )
    print(f"Saved candidate plots -> {cand_dir}")

    # ---- Plot histograms (robust thresholds) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, scores) in zip(axes, [("recon", all_recon), ("cadence", all_cadence)]):
        median, mad_sigma = robust_stats(scores)
        thresh_3 = median + 3 * mad_sigma
        thresh_5 = median + 5 * mad_sigma
        n_3s = (scores > thresh_3).sum()

        clipped = scores[scores < np.percentile(scores, 99.5)]
        ax.hist(clipped, bins=200, alpha=0.7, edgecolor="black", linewidth=0.2)
        ax.axvline(thresh_3, color="orange", ls="--", lw=1.5,
                   label=f"3s = {thresh_3:.3f}")
        ax.axvline(thresh_5, color="red", ls="--", lw=1.5,
                   label=f"5s = {thresh_5:.3f}")
        ax.set_xlabel("Anomaly score (MSE)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} -- {n_3s} candidates > 3s (robust)")
        ax.legend()

    plt.suptitle(f"Score distribution -- {n_total} snippets, {len(cadence_lines)} cadences\n"
                 f"Thresholds: median + k * MAD * 1.4826",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inference_score_distributions.png", dpi=150)
    plt.close()
    print(f"\nSaved -> {args.out_dir / 'inference_score_distributions.png'}")

    # ---- Per-cadence frequency profile ----
    fig, axes = plt.subplots(len(cadence_lines), 1,
                             figsize=(16, 3 * len(cadence_lines)), squeeze=False)
    cad_median, cad_mad_sigma = robust_stats(all_cadence)
    offset = 0
    for cad_idx in range(len(cadence_lines)):
        cad_rows = [r for r in all_rows if r["cadence_idx"] == cad_idx]
        n_cad = len(cad_rows)
        cad_scores = all_cadence[offset:offset + n_cad]
        fstarts = [r["f_start"] for r in cad_rows]

        ax = axes[cad_idx, 0]
        ax.plot(fstarts, cad_scores, linewidth=0.3, alpha=0.7)
        ax.axhline(cad_median + 3 * cad_mad_sigma, color="orange", ls="--", lw=1,
                   alpha=0.7, label="3s")
        ax.axhline(cad_median + 5 * cad_mad_sigma, color="red", ls="--", lw=1,
                   alpha=0.7, label="5s")
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
