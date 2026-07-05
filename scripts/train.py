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

import torch
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
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    p.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Existing run directory to resume into (logs and checkpoints append)",
    )
    return p.parse_args()


def _print_config_summary(cfg, args):
    model_cfg = cfg["model"]
    if model_cfg.get("architecture") == "vit_mae":
        backbone = "ViT-MAE"
    elif model_cfg.get("architecture") == "udma":
        backbone = "UDMA"
    elif model_cfg.get("mae"):
        backbone = "MAE"
    elif model_cfg.get("variational"):
        backbone = "VAE"
    elif model_cfg.get("memory"):
        backbone = "MemAE"
    else:
        backbone = "AE"

    train_cfg = cfg["training"]
    hw_cfg = cfg["hardware"]
    frame = cfg["data"]["frame"]

    print("=" * 60)
    print("Config summary")
    print(f"  training config : {args.config}")
    print(f"  data config      : {args.data or '(from training config)'}")
    print(f"  model config     : {args.model or '(from training config)'}")
    print(f"  backbone         : {backbone} (architecture={model_cfg.get('architecture')})")
    print(f"  frame shape      : (tchans={frame['tchans']}, fchans={frame['fchans']})")
    print(f"  loss             : {train_cfg['loss']}")
    print(f"  optimizer        : {train_cfg['optimizer']}")
    print(f"  learning_rate    : {train_cfg['learning_rate']}")
    print(f"  batch_size       : {train_cfg['batch_size']}")
    print(f"  epochs           : {train_cfg['epochs']} (patience={train_cfg['patience']})")
    print(f"  seed             : {train_cfg['seed']}")
    print(f"  hardware         : num_gpus={hw_cfg.get('num_gpus', 0)}, "
          f"mixed_precision={hw_cfg.get('mixed_precision', False)}")
    print("=" * 60)


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

    _print_config_summary(cfg, args)

    pl.seed_everything(cfg["training"]["seed"], workers=True)

    # ------------------------------------------------------------------ data
    data_cfg = cfg["data"]
    cadence_list_path = Path(data_cfg["dataset"]["cadence_list"])
    cadence_list = [
        line.strip().split()
        for line in cadence_list_path.read_text().splitlines()
        if line.strip()
    ]

    train_ds, val_ds = build_datasets(
        cadence_list,
        data_cfg,
        val_fraction=0.15,
        seed=cfg["training"]["seed"],
    )
    print(f"Dataset: {len(train_ds)} train snippets, {len(val_ds)} val snippets")

    num_gpus = int(cfg["hardware"].get("num_gpus", 0))
    pin_memory = num_gpus > 0
    # Lazy dataset loads from disk per __getitem__ — workers hide I/O latency
    # regardless of GPU count.
    num_workers = cfg["hardware"].get("num_workers", 4)

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
        beta=float(cfg["model"].get("beta", 1.0)),
    )

    if cfg["model"].get("architecture") == "udma":
        teacher = model.teacher
        if torch.allclose(teacher.mu, torch.zeros_like(teacher.mu)) and \
           torch.allclose(teacher.sigma, torch.ones_like(teacher.sigma)):
            raise SystemExit(
                "UDMA teacher normalization (Q2) is unfit (mu=0/sigma=1, the identity "
                "default) — training would regress raw, unnormalized teacher features. "
                "Run scripts/fit_udma_teacher_norm.py and set teacher.norm_stats in "
                "configs/model/udma.yaml before training."
            )

    # --------------------------------------------------------- output / train
    # Lightning's DDP subprocess launcher re-executes this script per GPU rank,
    # so without sharing the path each rank would create its own timestamped
    # run directory. Rank 0 picks the directory and exports it for the other
    # ranks via the environment, which the launcher copies at trainer.fit().
    if args.run_dir is not None:
        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    elif "BL_EXOTICA_RUN_DIR" in os.environ:
        run_dir = Path(os.environ["BL_EXOTICA_RUN_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg.get("output_dir", "outputs"))
    os.environ["BL_EXOTICA_RUN_DIR"] = str(run_dir)
    print(f"Run directory: {run_dir}")

    n_snap = min(4, len(val_ds))
    if n_snap >= 4:
        variances = torch.tensor(
            [val_ds[i].var().item() for i in range(len(val_ds))]
        )
        top_idx = int(variances.argmax())
        rng = torch.Generator().manual_seed(42)
        others = torch.randperm(len(val_ds), generator=rng)
        others = [int(i) for i in others if int(i) != top_idx][:n_snap - 1]
        snap_indices = [top_idx] + others
    else:
        snap_indices = list(range(n_snap))
    val_samples = torch.stack([val_ds[i] for i in snap_indices])
    if hasattr(val_ds, "close"):
        val_ds.close()
    # UDMA has no pixel decoder (forward() raises) — ReconstructionSnapshot
    # calls model(x) expecting a pixel reconstruction, so skip it for this
    # backbone rather than passing a val_sample it can't use.
    snapshot_val_sample = None if cfg["model"].get("architecture") == "udma" else val_samples
    callbacks = build_callbacks(cfg, run_dir, val_sample=snapshot_val_sample)

    module = AELightningModule(model, cfg)
    trainer = build_trainer(cfg, run_dir, callbacks=callbacks)

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
    print(f"Training complete. Best checkpoint: {callbacks[1].best_model_path}")


if __name__ == "__main__":
    main()
