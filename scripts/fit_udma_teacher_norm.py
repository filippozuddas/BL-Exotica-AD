"""One-shot: fit the UDMA teacher's per-channel token normalization (Q2/Q7).

Checklist step 3 (docs/2026-07-05_udma_design_spec.md) — MUST run before any
UDMA training launch. ``TeacherViT.mu``/``sigma`` default to identity (0/1);
without this step the students would regress raw, unnormalized teacher
features, which violates Q2 and invalidates the run.

Usage (server, not dev machine):
    PYTHONPATH=/content/filippo/BL-Exotica-AD python scripts/fit_udma_teacher_norm.py \\
        --config configs/training/udma_gbt_fine.yaml \\
        --out outputs/udma_teacher_norm/gbt_fine_block3.pt

Then point ``configs/model/udma.yaml``'s ``teacher.norm_stats`` at ``--out``
before starting ``scripts/train.py``.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import build_datasets
from src.models.autoencoder import build_autoencoder
from src.utils.config import load_config


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/training/udma_gbt_fine.yaml"),
                    help="Training config referencing configs/model/udma.yaml")
    p.add_argument("--out", type=Path, required=True, help="Output path for {mu, sigma} tensors")
    p.add_argument("--max_batches", type=int, default=None,
                    help="Cap the number of train batches (debugging only; full train set by default)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    architecture = cfg["model"].get("architecture")
    if architecture != "udma":
        raise SystemExit(f"--config's model must be architecture: udma, got '{architecture}'.")

    data_cfg = cfg["data"]
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
    print(f"Fitting teacher normalization on {len(train_ds)} train snippets.")

    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["hardware"].get("num_workers", 4),
    )

    frame = data_cfg["frame"]
    input_shape = (frame["tchans"], frame["fchans"], 1)
    model = build_autoencoder(
        input_shape=input_shape,
        model_config=cfg["model"],
        loss=cfg["training"]["loss"],
        learning_rate=cfg["training"]["learning_rate"],
    )
    model.to(args.device)

    teacher_layer = getattr(model.teacher, "teacher_layer", None)
    layer_info = f"layer {teacher_layer}" if teacher_layer is not None else type(model.teacher).__name__
    print(f"Teacher: {layer_info}, channels: {model.teacher.channels}, grid: {model.teacher.grid_size}")
    model.teacher.fit_normalization(loader, args.device, max_batches=args.max_batches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": model.teacher.mu.cpu(), "sigma": model.teacher.sigma.cpu()}, args.out)
    print(f"Saved -> {args.out}")
    print(f"Add to configs/model/udma.yaml: teacher.norm_stats: {args.out}")


if __name__ == "__main__":
    main()
