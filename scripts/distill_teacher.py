"""Distill T (CNN teacher) from P (frozen ImageNet ResNet-18) — Fase 2 of
docs/2026-07-14_paper_alignment_plan.md (5.2), the paper-faithful teacher
route (Qi et al. 2024, Eq. 2 / Bergmann "Uninformed Students"): a small CNN
is trained to reproduce a frozen, out-of-domain generic backbone's features
on in-domain data. Spectrum data enters only as distillation input — P is
never trained on it, so its feature space stays anchored outside the domain
by construction (unlike the retired in-domain self-supervised ViT-MAE
teacher, whose too-learnable target collapsed student disagreement).

P: scripts/debug/resnet_teacher.py's ResNetTeacher (gate PASSED 2026-07-15,
see memory: udma_resnet_teacher_gate) — frozen, forward through layer3,
256ch on the (6,64) grid.

T: TeacherCNN's trunk (src/models/udma.py) — build_encoder with the same
parametrisation as the UDMA students (filters [32,64,128,256],
convs_per_block 2), latent_dim=128 (D8). Training itself lives in
src/training/distill.py; this script is a thin CLI wrapper (loads data/P,
calls distill_teacher(), saves the result).

Usage (server, not dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \\
    python scripts/distill_teacher.py \\
        --config configs/training/udma_gbt_fine_control_bs256.yaml \\
        --epochs 2 --lr 1e-3 \\
        --out outputs/udma_teacher_distill/cnn_distilled_resnet18.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.debug.resnet_teacher import ResNetTeacher
from src.data.torch_dataset import build_datasets
from src.training.distill import distill_teacher
from src.utils.config import load_config

DEFAULT_FILTERS = [32, 64, 128, 256]
T_LATENT_DIM = 128
T_KERNEL_SIZE = (3, 3)
T_ACTIVATION = "relu"
T_USE_BATCHNORM = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs/training/udma_gbt_fine_control_bs256.yaml",
                   help="Training config referencing configs/data (cadence_list/frame/preprocessing) "
                        "and hardware (num_gpus, used to pick --device by default). Model architecture "
                        "is fixed below (distillation doesn't touch the student/memory config).")
    p.add_argument("--epochs", type=int, default=2,
                   help="1-2 epochs over the full train cache (plan default).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Default: use --config's training.batch_size.")
    p.add_argument("--filters", type=int, nargs="+", default=DEFAULT_FILTERS)
    p.add_argument("--convs_per_block", type=int, default=2)
    p.add_argument("--out", type=Path, required=True,
                   help="Output checkpoint path for T's trunk (no D, no optimizer state).")
    p.add_argument("--device", default=None,
                   help="Default: config-driven from --config's hardware.num_gpus "
                        "('cuda' if > 0, else 'cpu').")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    cfg = load_config(args.config)
    device = args.device or ("cuda" if cfg["hardware"].get("num_gpus", 0) > 0 else "cpu")

    data_cfg = cfg["data"]
    frame = data_cfg["frame"]
    input_shape = (frame["tchans"], frame["fchans"], 1)

    cadence_list_path = Path(data_cfg["dataset"]["cadence_list"])
    cadence_list = [
        line.strip().split()
        for line in cadence_list_path.read_text().splitlines()
        if line.strip()
    ]
    train_ds, val_ds = build_datasets(
        cadence_list, data_cfg, val_fraction=0.15, seed=cfg["training"]["seed"],
    )
    if hasattr(val_ds, "close"):
        val_ds.close()
    print(f"Distilling on {len(train_ds)} train snippets, input_shape={input_shape}, device={device}")

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg["hardware"].get("num_workers", 4), drop_last=True,
    )

    print(f"Loading P (frozen ResNet-18, ImageNet, layer3) on {device}")
    P = ResNetTeacher().to(device)
    P.eval()

    print(f"Distilling T (filters={args.filters}, latent_dim={T_LATENT_DIM}) "
          f"+ D (1x1 {T_LATENT_DIM}->{P.channels}, distillation-only)")
    T, losses = distill_teacher(
        P, loader, input_shape, device,
        filters=args.filters, latent_dim=T_LATENT_DIM, p_channels=P.channels,
        kernel_size=T_KERNEL_SIZE, activation=T_ACTIVATION, use_batchnorm=T_USE_BATCHNORM,
        convs_per_block=args.convs_per_block,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        log_every_n_steps=args.log_every_n_steps,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trunk_state_dict": T.state_dict(),
        "filters": args.filters,
        "latent_dim": T_LATENT_DIM,
        "kernel_size": T_KERNEL_SIZE,
        "activation": T_ACTIVATION,
        "use_batchnorm": T_USE_BATCHNORM,
        "convs_per_block": args.convs_per_block,
        "final_loss": losses[-1],
    }, args.out)
    print(f"Saved T (trunk only, no D) -> {args.out}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE(D(T(x)), P(x))")
    ax.set_title("Distillation loss")
    plt.tight_layout()
    loss_plot_path = args.out.with_suffix(".loss.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss curve -> {loss_plot_path}")
    print(f"\nNext: point configs/model/*.yaml's teacher.type: cnn_distilled, "
          f"teacher.checkpoint: {args.out} then run scripts/fit_udma_teacher_norm.py.")


if __name__ == "__main__":
    main()
