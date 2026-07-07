"""Visualize the MemAE memory bank, following Qi et al. 2024 (UDMA paper) Fig. 14.

Only the **standalone** ``MemAE`` (Gong et al. 2019, ``src/models/autoencoder.py``)
can be visualized this way: it has a real pixel decoder (``self.decoder(z_hat)``).
UDMA's memory-augmented student (``src/models/udma.py: FeatureStudent(memory=True)``)
has no pixel decoder at all (its "decoder" is a projection head onto the frozen
ViT-MAE teacher's token grid) — its memory items cannot be decoded to spectrograms.
This script therefore visualizes a proxy, not the production UDMA scorer's memory.

Method (adapted from the paper, which concatenates each item with a zero vector
before decoding because Qi's decoder takes ``[query || retrieved]``; this repo's
MemAE decodes ``z_hat`` directly, so no concatenation is needed):
1. Take a memory item ``m`` (dim = ``latent_dim``, the spatial bottleneck channel
   count).
2. Broadcast it uniformly across every spatial position of the bottleneck grid
   to build a synthetic ``z_hat`` of shape ``(1, latent_dim, H', W')``.
3. Decode it with ``model.decoder`` to obtain a full-size pixel-space spectrogram.

This is a qualitative probe: the decoder never saw a spatially-uniform ``z_hat``
during training (only convex combinations that vary per position), so the
decoded image is not a literal "what this slot reconstructs" but an indication
of the prototype's dominant spectro-temporal pattern.

Slot selection: with N=500 items, hard-shrinkage sparsity, and the entropy loss,
addressing is sparse and many slots may be rarely or never used. Passing
``--cache``/``--split`` computes each slot's argmax-addressing frequency over a
sample of real data and (a) plots the full usage histogram, (b) decodes the
top-K most-used items instead of a random/sequential slice. Without a cache,
the script falls back to the first ``--n_items`` slots (index order only, no
usage information).

Usage:
    python scripts/debug/visualize_memae_memory.py \
        --checkpoint outputs/.../best_model.ckpt \
        --model_config configs/model/memae.yaml \
        --cache /path/to/cache_gbt_fine --split val --n_scan 2000 \
        --n_items 16
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

from scripts.debug.ae_recon_visual import load_model, preprocess_raw


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True,
                    help="Must have memory: true (e.g. configs/model/memae.yaml).")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--cache", type=Path, default=None,
                    help="Optional: cache dir with {split}.npy, used to compute slot "
                         "usage frequency. Without it, items are shown in index order.")
    p.add_argument("--split", default="val")
    p.add_argument("--n_scan", type=int, default=2000,
                    help="Number of cached snippets to run through the encoder for usage stats.")
    p.add_argument("--n_items", type=int, default=16,
                    help="Number of memory items to decode and plot.")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/memae_memory_viz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_usage_from_frames(model, frames, device, batch_size=64):
    """Argmax-addressing frequency per memory slot over a list/array of (already
    preprocessed) 2D frames.
    """
    mem_slots = model.memory.mem_slots
    counts = np.zeros(mem_slots, dtype=np.int64)
    n = len(frames)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = frames[start:start + batch_size]
            x = torch.from_numpy(np.stack(batch)).float().unsqueeze(1).to(device)  # (B,1,H,W)
            z = model.encoder(x)
            _, att = model.memory(z)  # att: (B*H'*W', N)
            winners = att.argmax(dim=1).cpu().numpy()
            u, c = np.unique(winners, return_counts=True)
            counts[u] += c
    return counts


def decode_item(model, item, grid_hw, device):
    """Broadcast a single memory item across the (H', W') bottleneck grid and
    decode it to a full pixel-space spectrogram.
    """
    h, w = grid_hw
    c = item.shape[0]
    z_hat = item.view(1, c, 1, 1).expand(1, c, h, w).contiguous().to(device)
    with torch.no_grad():
        recon = model.decoder(z_hat)
    return recon.squeeze().cpu().numpy()


def bottleneck_grid(model, input_shape, device):
    """Infer (H', W') by running a dummy input through the encoder."""
    with torch.no_grad():
        dummy = torch.zeros(1, 1, input_shape[0], input_shape[1], device=device)
        z = model.encoder(dummy)
    return z.shape[-2], z.shape[-1]


def main():
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    if not model_cfg.get("memory", False):
        raise ValueError(
            f"{args.model_config} does not have memory: true — this script needs the "
            f"standalone MemAE (Gong et al.), not a plain AE/MAE/VAE."
        )
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame_cfg = data_cfg["frame"]
    input_shape = (frame_cfg["tchans"], frame_cfg["fchans"], 1)

    print(f"Loading MemAE from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)
    grid_hw = bottleneck_grid(model, input_shape, args.device)
    mem_slots = model.memory.mem_slots
    latent_dim = model.memory.feature_dim
    print(f"Bottleneck grid: {grid_hw}, mem_slots={mem_slots}, latent_dim={latent_dim}")

    counts = None
    if args.cache is not None:
        npy_path = Path(args.cache) / f"{args.split}.npy"
        print(f"Loading cache for usage stats: {npy_path}")
        arr = np.load(str(npy_path), mmap_mode="r")
        n_scan = min(args.n_scan, arr.shape[0])
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(arr.shape[0], size=n_scan, replace=False)
        frames = [preprocess_raw(np.array(arr[i]), preproc) for i in idx]
        del arr
        counts = compute_usage_from_frames(model, frames, args.device)
        n_dead = int((counts == 0).sum())
        print(f"Slot usage over {n_scan} snippets ({n_scan * grid_hw[0] * grid_hw[1]} positions): "
              f"{n_dead}/{mem_slots} slots never won an addressing argmax.")

    if counts is not None:
        order = np.argsort(-counts)
    else:
        order = np.arange(mem_slots)
    top_idx = order[:args.n_items]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Decoded memory items.
    n_cols = 4
    n_rows = int(np.ceil(args.n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    memory = model.memory.memory.detach()
    for i, slot in enumerate(top_idx):
        row, col = divmod(i, n_cols)
        img = decode_item(model, memory[slot], grid_hw, args.device)
        vmin, vmax = np.percentile(img, [1, 99])
        axes[row][col].imshow(img, aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        title = f"slot {slot}"
        if counts is not None:
            title += f"\nusage={counts[slot]}"
        axes[row][col].set_title(title, fontsize=9)
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    for i in range(args.n_items, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")

    fig.suptitle(
        "MemAE memory items decoded to pixel space (Qi et al. 2024, Fig. 14 method)\n"
        "Each panel: one memory prototype broadcast uniformly across the bottleneck "
        "grid, then decoded — a qualitative probe, not a literal reconstruction "
        "(the decoder never saw a spatially-uniform latent during training).",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = args.out_dir / "memory_items_decoded.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    # Usage histogram (only if we had a cache).
    if counts is not None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(np.arange(mem_slots), counts, width=1.0)
        ax.set_xlabel("memory slot index")
        ax.set_ylabel("argmax-addressing count")
        ax.set_title(
            f"MemAE addressing usage over {n_scan} snippets "
            f"({n_dead}/{mem_slots} slots unused) — collapsed vs. diverse memory diagnostic"
        )
        plt.tight_layout()
        hist_path = args.out_dir / "memory_usage_histogram.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {hist_path}")


if __name__ == "__main__":
    main()
