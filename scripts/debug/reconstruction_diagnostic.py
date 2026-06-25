"""
Reconstruction quality diagnostic: is the model reconstructing structure or
collapsing to the mean?

For quiet, mild-RFI, and strong-RFI snippets, reports:
  - Input vs reconstruction pixel statistics (mean, std, max, energy)
  - Energy ratio ||recon||² / ||input||²
  - Visual comparison (input | reconstruction | residual) with shared colorscale
  - Per-sample scatter: input energy vs reconstruction energy

Usage (on server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/reconstruction_diagnostic.py \
        --checkpoint outputs/training/.../checkpoints/best.ckpt \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \
        --split val \
        --out_dir outputs/debug/recon_diagnostic
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
from src.data.preprocessing import bandpass_correct, core_transform

INPUT_SHAPE = (96, 1024, 1)


def load_model(checkpoint_path, model_config, device):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def preprocess_raw(raw_snippet, preproc):
    stacked = raw_snippet.reshape(-1, raw_snippet.shape[-1])
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    stacked = core_transform(stacked, mad_epsilon)
    return stacked.astype(np.float32)


def reconstruct(model, snippet, device):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze().cpu().numpy()


def energy(arr):
    return float(np.mean(arr ** 2))


def plot_comparison(snippets, recons, labels, out_path, suptitle):
    n = len(snippets)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        inp = snippets[i]
        rec = recons[i]
        res = np.abs(inp - rec)

        vmin, vmax = np.percentile(inp, [1, 99])

        axes[i, 0].imshow(inp, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[i, 0].set_title(f"{labels[i]} — Input")
        axes[i, 0].set_ylabel("Time bin")

        axes[i, 1].imshow(rec, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
        e_ratio = energy(rec) / max(energy(inp), 1e-10)
        axes[i, 1].set_title(f"Reconstruction (energy ratio: {e_ratio:.4f})")

        im = axes[i, 2].imshow(res, aspect="auto", origin="upper", cmap="hot")
        axes[i, 2].set_title(f"Residual |inp - rec|")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    for ax in axes[-1]:
        ax.set_xlabel("Freq channel")

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/debug/recon_diagnostic")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    npy_path = args.cache / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    indices = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[indices])
    del arr
    print(f"  Loaded {len(raw_snippets)} snippets, shape {raw_snippets.shape}")

    # Preprocess all
    preprocessed = np.array([preprocess_raw(s, preproc) for s in raw_snippets])

    # Classify by RFI content: fraction of pixels > 5 MAD
    hot_fracs = np.array([float((s > 5.0).sum()) / s.size for s in preprocessed])
    q10 = np.percentile(hot_fracs, 10)
    q50 = np.percentile(hot_fracs, 50)
    q95 = np.percentile(hot_fracs, 95)

    quiet_idx = np.where(hot_fracs <= q10)[0]
    mild_idx = np.where((hot_fracs > q50 - 0.01) & (hot_fracs < q50 + 0.01))[0]
    strong_idx = np.where(hot_fracs >= q95)[0]

    print(f"\n  Categories (by hot_frac = fraction of pixels > 5 MAD):")
    print(f"    Quiet  (≤p10={q10:.4f}): {len(quiet_idx)} snippets")
    print(f"    Mild   (~p50={q50:.4f}): {len(mild_idx)} snippets")
    print(f"    Strong (≥p95={q95:.4f}): {len(strong_idx)} snippets")

    # ---- 1. Per-category statistics ----
    print(f"\n{'='*80}")
    print(f"1. RECONSTRUCTION STATISTICS BY CATEGORY")
    print(f"{'='*80}")

    categories = [
        ("quiet", quiet_idx),
        ("mild_rfi", mild_idx),
        ("strong_rfi", strong_idx),
    ]

    cat_stats = {}
    for cat_name, cat_idx in categories:
        if len(cat_idx) == 0:
            print(f"\n  {cat_name}: no snippets")
            continue

        sample = cat_idx[:min(50, len(cat_idx))]
        inp_energies = []
        rec_energies = []
        ratios = []
        inp_maxes = []
        rec_maxes = []

        for i in sample:
            inp = preprocessed[i]
            rec = reconstruct(model, inp, args.device)
            e_inp = energy(inp)
            e_rec = energy(rec)
            inp_energies.append(e_inp)
            rec_energies.append(e_rec)
            ratios.append(e_rec / max(e_inp, 1e-10))
            inp_maxes.append(float(np.max(np.abs(inp))))
            rec_maxes.append(float(np.max(np.abs(rec))))

        inp_energies = np.array(inp_energies)
        rec_energies = np.array(rec_energies)
        ratios = np.array(ratios)
        inp_maxes = np.array(inp_maxes)
        rec_maxes = np.array(rec_maxes)

        cat_stats[cat_name] = {
            "inp_energies": inp_energies,
            "rec_energies": rec_energies,
            "ratios": ratios,
        }

        print(f"\n  {cat_name} (n={len(sample)}):")
        print(f"    Input  energy: {inp_energies.mean():.4f} ± {inp_energies.std():.4f}  "
              f"max_pixel: {inp_maxes.mean():.2f} ± {inp_maxes.std():.2f}")
        print(f"    Recon  energy: {rec_energies.mean():.6f} ± {rec_energies.std():.6f}  "
              f"max_pixel: {rec_maxes.mean():.4f} ± {rec_maxes.std():.4f}")
        print(f"    Energy ratio:  {ratios.mean():.6f} ± {ratios.std():.6f}  "
              f"(min={ratios.min():.6f}, max={ratios.max():.6f})")

    # ---- 2. Visual comparison: pick 1 representative per category ----
    print(f"\n{'='*80}")
    print(f"2. VISUAL COMPARISON")
    print(f"{'='*80}")

    vis_snippets = []
    vis_recons = []
    vis_labels = []

    for cat_name, cat_idx in categories:
        if len(cat_idx) == 0:
            continue
        # Pick the median-energy snippet as representative
        energies = np.array([energy(preprocessed[i]) for i in cat_idx[:50]])
        median_idx = cat_idx[np.argsort(energies)[len(energies) // 2]]
        inp = preprocessed[median_idx]
        rec = reconstruct(model, inp, args.device)
        vis_snippets.append(inp)
        vis_recons.append(rec)
        vis_labels.append(f"{cat_name} (hot_frac={hot_fracs[median_idx]:.4f})")

    plot_comparison(vis_snippets, vis_recons, vis_labels,
                    args.out_dir / "reconstruction_comparison.png",
                    "Reconstruction diagnostic: quiet vs mild RFI vs strong RFI")
    print(f"  Saved -> {args.out_dir / 'reconstruction_comparison.png'}")

    # Also plot the most extreme strong-RFI snippet
    if len(strong_idx) > 0:
        extreme_i = strong_idx[np.argmax([energy(preprocessed[i]) for i in strong_idx[:50]])]
        inp = preprocessed[extreme_i]
        rec = reconstruct(model, inp, args.device)
        plot_comparison([inp], [rec],
                        [f"Most extreme RFI (hot_frac={hot_fracs[extreme_i]:.4f})"],
                        args.out_dir / "extreme_rfi.png",
                        "Most extreme RFI snippet: does the model reconstruct anything?")
        print(f"  Saved -> {args.out_dir / 'extreme_rfi.png'}")

    # ---- 3. Scatter: input energy vs reconstruction energy ----
    print(f"\n{'='*80}")
    print(f"3. ENERGY SCATTER")
    print(f"{'='*80}")

    all_inp_e = []
    all_rec_e = []
    all_cats = []

    for cat_name, cat_idx in categories:
        if cat_name in cat_stats:
            all_inp_e.extend(cat_stats[cat_name]["inp_energies"])
            all_rec_e.extend(cat_stats[cat_name]["rec_energies"])
            all_cats.extend([cat_name] * len(cat_stats[cat_name]["inp_energies"]))

    all_inp_e = np.array(all_inp_e)
    all_rec_e = np.array(all_rec_e)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"quiet": "steelblue", "mild_rfi": "orange", "strong_rfi": "crimson"}
    for cat_name in ["quiet", "mild_rfi", "strong_rfi"]:
        mask = np.array([c == cat_name for c in all_cats])
        if mask.sum() == 0:
            continue
        axes[0].scatter(all_inp_e[mask], all_rec_e[mask], s=15, alpha=0.6,
                        color=colors[cat_name], label=cat_name)

    axes[0].set_xlabel("Input energy (mean squared pixel)")
    axes[0].set_ylabel("Reconstruction energy")
    axes[0].set_title("Input vs Reconstruction energy")
    axes[0].legend()
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    # Reference line: perfect reconstruction
    lims = [min(all_inp_e.min(), all_rec_e.min()), max(all_inp_e.max(), all_rec_e.max())]
    axes[0].plot(lims, lims, "k--", alpha=0.3, label="y=x (perfect)")
    axes[0].grid(True, alpha=0.3)

    # Right panel: energy ratio histogram
    for cat_name in ["quiet", "mild_rfi", "strong_rfi"]:
        if cat_name in cat_stats:
            axes[1].hist(cat_stats[cat_name]["ratios"], bins=30, alpha=0.5,
                         color=colors[cat_name], label=cat_name, edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Energy ratio (||recon||² / ||input||²)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Reconstruction energy ratio by category")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Does the model reconstruct structure or collapse to zero?", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out_dir / "energy_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved -> {args.out_dir / 'energy_scatter.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
