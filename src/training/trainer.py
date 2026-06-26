import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy


class AELightningModule(pl.LightningModule):
    """Lightning wrapper for Autoencoder / MAE / VAE.

    Handles training/validation steps and the optimiser schedule. The wrapped
    model exposes ``compute_loss(x) -> Tensor`` (AE/MAE) or
    ``compute_loss(x) -> (Tensor, dict)`` (VAE) — both paths are handled here.
    """

    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        loss_out = self.model.compute_loss(batch)

        if isinstance(loss_out, tuple):
            # VAE path: (total_loss, {"reconstruction_loss": ..., "kl_loss": ...})
            loss, components = loss_out
            self.log_dict(
                {f"{stage}_{k}": v for k, v in components.items()},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            loss = loss_out

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        train_cfg = self.cfg["training"]
        opt_name = str(train_cfg.get("optimizer", "adam")).lower()
        wd = float(train_cfg.get("weight_decay", 0.0))
        if opt_name == "adamw":
            opt = AdamW(self.parameters(), lr=self.model.learning_rate, weight_decay=wd)
        else:
            opt = Adam(self.parameters(), lr=self.model.learning_rate, weight_decay=wd)

        total_epochs = train_cfg["epochs"]
        warmup_epochs = train_cfg.get("warmup_epochs", 0)

        if warmup_epochs > 0:
            warmup = LinearLR(
                opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = CosineAnnealingLR(
                opt, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            )
            scheduler = SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-6)

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _run_id() -> str:
    """Generate a timestamped run ID with the short git hash."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "nogit"
    return f"{ts}_{git_hash}"


def build_trainer(
    cfg: Dict[str, Any],
    run_dir: Path,
    callbacks=None,
) -> pl.Trainer:
    """Build a Lightning Trainer from the merged config.

    Hardware selection is fully config-driven:
    - ``hardware.num_gpus: 0``  → CPU training
    - ``hardware.num_gpus: 1``  → single GPU
    - ``hardware.num_gpus: 2``  → DDP across 2 GPUs (strategy="auto")
    - ``hardware.mixed_precision: true``  → bf16-mixed (native on Ada/4090)
    - ``hardware.mixed_precision: false`` → 32-true

    Args:
        cfg: merged config dict from src/utils/config.py.
        run_dir: run-specific output directory (outputs/<run_id>).
        callbacks: list of Lightning callbacks (from build_callbacks).

    Returns:
        Configured pl.Trainer.
    """
    hw = cfg["hardware"]
    num_gpus = int(hw.get("num_gpus", 0))
    mixed_precision = bool(hw.get("mixed_precision", False))

    if num_gpus == 0:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "gpu"
        devices = num_gpus
        # find_unused_parameters=True: some loss modes (e.g. denoising) don't use
        # mask_token in the forward pass; DDP requires this flag when parameters are
        # conditionally unused across loss modes in the same ViTMAE class.
        strategy = DDPStrategy(find_unused_parameters=True) if num_gpus > 1 else "auto"

    precision = "bf16-mixed" if (mixed_precision and num_gpus > 0) else "32-true"

    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="", version="")

    gradient_clip_val = cfg["training"].get("gradient_clip_val", None)

    return pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks or [],
        logger=logger,
        log_every_n_steps=10,
        enable_model_summary=False,
        enable_progress_bar=False,
        gradient_clip_val=gradient_clip_val,
    )


def make_run_dir(output_root: Union[str, Path]) -> Path:
    """Create and return a timestamped run directory under output_root."""
    run_dir = Path(output_root) / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
