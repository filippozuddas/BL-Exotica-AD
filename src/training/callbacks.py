from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_callbacks(
    cfg: Dict[str, Any],
    run_dir: Path,
    val_sample: Optional[torch.Tensor] = None,
) -> List[pl.Callback]:
    """Build the standard callback stack for AE training.

    Args:
        cfg: merged config dict (see src/utils/config.py).
        run_dir: run-specific output directory (e.g. outputs/<run_id>).
        val_sample: a single (1, C, H, W) tensor used by ReconstructionSnapshot.
            Pass None to skip snapshot logging.

    Returns:
        List of Lightning callbacks.
    """
    train_cfg = cfg["training"]

    callbacks: List[pl.Callback] = [
        EarlyStopping(
            monitor=train_cfg.get("monitor", "val_loss"),
            patience=train_cfg.get("patience", 15),
            min_delta=train_cfg.get("min_delta", 1e-5),
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="epoch={epoch:03d}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if val_sample is not None:
        callbacks.append(
            ReconstructionSnapshot(
                val_sample=val_sample,
                output_dir=run_dir / "snapshots",
                every_n_epochs=cfg["training"].get("snapshot_every", 10),
            )
        )

    return callbacks


class ReconstructionSnapshot(pl.Callback):
    """Save a side-by-side input / reconstruction figure every N epochs.

    Useful for visually verifying that the model is learning the noise
    distribution rather than collapsing or copying the input trivially.
    """

    def __init__(
        self,
        val_sample: torch.Tensor,
        output_dir: Path,
        every_n_epochs: int = 10,
    ):
        super().__init__()
        # Register as buffer-free attribute; will be moved to device in hook
        self._val_sample = val_sample
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        x = self._val_sample.to(pl_module.device)
        with torch.no_grad():
            recon = pl_module.model(x)

        # Convert to numpy for plotting — squeeze batch and channel dims
        inp = x[0, 0].cpu().numpy()       # (H, W)
        rec = recon[0, 0].cpu().numpy()   # (H, W)
        err = np.abs(inp - rec)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, img, title in zip(
            axes,
            [inp, rec, err],
            ["Input", "Reconstruction", "|Error|"],
        ):
            im = ax.imshow(img, aspect="auto", origin="lower", interpolation="nearest")
            ax.set_title(f"{title}  (epoch {epoch + 1})")
            ax.set_xlabel("Frequency channel")
            ax.set_ylabel("Time bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch + 1:04d}.png", dpi=100)
        plt.close(fig)
