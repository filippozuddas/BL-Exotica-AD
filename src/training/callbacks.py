import time
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

    callbacks.append(TrainProgressBar())
    callbacks.append(EpochSummary())

    return callbacks


class TrainProgressBar(pl.Callback):
    """In-place, single-line progress bar for non-interactive stdout.

    Lightning's TQDMProgressBar prints a new line per refresh when stdout
    isn't a TTY (e.g. ``!python`` in a Colab cell). This writes ``\\r`` and
    overwrites the same line instead, then yields to EpochSummary's final
    per-epoch line.
    """

    def __init__(self, refresh_every: int = 10):
        super().__init__()
        self.refresh_every = refresh_every
        self._epoch_start = 0.0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._epoch_start = time.monotonic()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if not trainer.is_global_zero:
            return

        total = trainer.num_training_batches
        step = batch_idx + 1
        if step % self.refresh_every != 0 and step != total:
            return

        frac = step / total
        elapsed = time.monotonic() - self._epoch_start
        eta = elapsed / frac - elapsed if frac > 0 else 0.0

        bar_len = 30
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)

        loss = trainer.callback_metrics.get("train_loss")
        loss_str = f" loss={loss:.4f}" if loss is not None else ""

        print(
            f"\repoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"[{bar}] {step}/{total}{loss_str} ({elapsed:.0f}s, eta {eta:.0f}s)",
            end="",
            flush=True,
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        print()  # finalize the progress line before EpochSummary's recap


class EpochSummary(pl.Callback):
    """Print one compact line per epoch instead of a per-step progress bar.

    Useful when stdout isn't a TTY (e.g. ``!python`` in a notebook), where
    Lightning's progress bar prints a new line per refresh.
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        parts = [f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        parts.append(f"lr={lr:.2e}")
        print(" - ".join(parts), flush=True)


class ReconstructionSnapshot(pl.Callback):
    """Save input / reconstruction / error grid every N epochs.

    Accepts ``(B, C, H, W)`` with ``B >= 1`` samples.  The first sample is
    typically the highest-variance snippet (most structure); the rest provide
    diversity.  Each sample gets one row of three panels: Input, Reconstruction,
    |Error|.
    """

    def __init__(
        self,
        val_sample: torch.Tensor,
        output_dir: Path,
        every_n_epochs: int = 10,
    ):
        super().__init__()
        if val_sample.dim() == 3:
            val_sample = val_sample.unsqueeze(0)
        self._val_samples = val_sample
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        x = self._val_samples.to(pl_module.device)
        with torch.no_grad():
            recon = pl_module.model(x)

        n = x.shape[0]
        fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)
        for row in range(n):
            inp = x[row, 0].cpu().numpy()
            rec = recon[row, 0].cpu().numpy()
            err = np.abs(inp - rec)
            label = "highest-var" if row == 0 else f"sample {row}"
            for col, (img, title) in enumerate(
                zip([inp, rec, err], ["Input", "Reconstruction", "|Error|"])
            ):
                ax = axes[row, col]
                im = ax.imshow(img, aspect="auto", origin="lower", interpolation="nearest")
                ax.set_title(f"{title}  (epoch {epoch + 1}, {label})")
                ax.set_xlabel("Frequency channel")
                ax.set_ylabel("Time bin")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch + 1:04d}.png", dpi=100)
        plt.close(fig)
