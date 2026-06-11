"""Training entry point for BL-Exotica autoencoder.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training/default.yaml
    python scripts/train.py --config configs/training/default.yaml \\
                            --data   configs/data/gbt_fine.yaml \\
                            --model  configs/model/convae.yaml

The script merges the training config with the referenced data and model
configs, builds the dataset, model, and Lightning trainer, and runs training.
Outputs (checkpoints, logs, snapshots) are written to outputs/<run_id>/.
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.models.autoencoder import build_autoencoder
from src.data.torch_dataset import build_datasets
from src.training.trainer import AELightningModule, build_trainer, make_run_dir
from src.training.callbacks import build_callbacks
from src.utils.config import load_config


def _parse_args():
    p = argparse.ArgumentParser(description="Train BL-Exotica autoencoder")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/default.yaml"),
        help="Path to training config YAML",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override data config path (e.g. configs/data/gbt_fine.yaml)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override model config path",
    )
    return p.parse_args()


def main():
    # Suppress Lightning's per-run INFO messages (GPU/DDP setup, tensor-core
    # tip, etc.) - the EpochSummary callback prints the per-epoch progress.
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    args = _parse_args()

    cfg = load_config(args.config)

    if args.data is not None:
        with open(args.data) as f:
            cfg["data"] = yaml.safe_load(f)
    if args.model is not None:
        with open(args.model) as f:
            cfg["model"] = yaml.safe_load(f)

    pl.seed_everything(cfg["training"]["seed"], workers=True)

    # ------------------------------------------------------------------ data
    data_cfg = cfg["data"]
    file_list_path = Path(data_cfg["dataset"]["file_list"])
    file_list = [p.strip() for p in file_list_path.read_text().splitlines() if p.strip()]

    train_ds, val_ds = build_datasets(
        file_list,
        data_cfg,
        val_fraction=0.15,
        seed=cfg["training"]["seed"],
    )
    print(f"Dataset: {len(train_ds)} train snippets, {len(val_ds)} val snippets")

    num_gpus = int(cfg["hardware"].get("num_gpus", 0))
    pin_memory = num_gpus > 0
    num_workers = 4 if num_gpus > 0 else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # ----------------------------------------------------------------- model
    frame = data_cfg["frame"]
    input_shape = (frame["tchans"], frame["fchans"], 1)
    model = build_autoencoder(
        input_shape=input_shape,
        model_config=cfg["model"],
        loss=cfg["training"]["loss"],
        learning_rate=cfg["training"]["learning_rate"],
    )

    # --------------------------------------------------------- output / train
    # Lightning's DDP subprocess launcher re-executes this script per GPU rank,
    # so without sharing the path each rank would create its own timestamped
    # run directory. Rank 0 picks the directory and exports it for the other
    # ranks via the environment, which the launcher copies at trainer.fit().
    if "BL_EXOTICA_RUN_DIR" in os.environ:
        run_dir = Path(os.environ["BL_EXOTICA_RUN_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg.get("output_dir", "outputs"))
        os.environ["BL_EXOTICA_RUN_DIR"] = str(run_dir)
    print(f"Run directory: {run_dir}")

    val_sample = val_ds[0].unsqueeze(0)
    callbacks = build_callbacks(cfg, run_dir, val_sample=val_sample)

    module = AELightningModule(model, cfg)
    trainer = build_trainer(cfg, run_dir, callbacks=callbacks)

    trainer.fit(module, train_loader, val_loader)
    print(f"Training complete. Best checkpoint: {callbacks[1].best_model_path}")


if __name__ == "__main__":
    main()
