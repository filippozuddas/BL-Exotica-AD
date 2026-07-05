"""
Full anomaly-detection model: encoder + bottleneck + decoder.

Four backbones, selected by config flags. The first three share the same CNN
conv stack:

- ``Autoencoder`` (default, ``mae=false``, ``variational=false``): deterministic
  AE. Loss = reconstruction error on the full input. Comparison baseline.
- ``MAE`` (``mae=true``, primary): CNN Masked Autoencoder. During training, ~75%
  of patches are zeroed and the loss is computed only on those masked positions,
  forcing the model to predict occluded regions from context. At inference, no
  masking — scored by full reconstruction error like the plain AE. Addresses the
  "too-good reconstruction" failure mode for locally-regular signals.
- ``VAE`` (``variational=true``): variational backbone. Loss = reconstruction +
  ``beta`` * KL. Optional, fairly-comparable variant for empirical comparison.
- ``MemAE`` (``memory=true``): memory-augmented AE (Gong et al. 2019). Inserts a
  content-addressable memory of learned normal prototypes between encoder and
  decoder (see ``memory.py``); the decoder reconstructs only from normal
  prototypes, so anomalies (e.g. a narrowband line the plain AE would copy) are
  redrawn as normal and surface as reconstruction residual. Loss =
  reconstruction + ``entropy_weight`` * addressing-entropy. Scored by
  reconstruction error.

The 4th and 5th backbones are selected by ``architecture`` rather than the CNN flags:

- ``ViTMAE`` (``architecture: vit_mae``, see ``vit_mae.py``): He et al.-style
  Vision-Transformer Masked Autoencoder — Conv2d patch tokeniser +
  ``nn.TransformerEncoder`` stacks, token-removal masking during training,
  deterministic partitioned reconstruction at inference. A ViT alternative to
  the CNN ``MAE`` (which is inherently CNN-based), bypassing the
  encoder/bottleneck/decoder config sections entirely.
- ``UDMA`` (``architecture: udma``, see ``udma.py``): Qi et al. 2024 teacher-
  student distillation + memory. A frozen ``ViTMAE`` teacher (self-supervised,
  read at an intermediate block) supplies a token-feature target that two CNN
  "students" (one plain, one memory-augmented) are trained to regress; the
  anomaly score is their prediction disagreement on the teacher's feature grid,
  not pixel reconstruction. No pixel decoder. See
  ``docs/2026-07-05_udma_design_spec.md``.

All five are pure ``nn.Module`` wrappers exposing ``compute_loss`` (training)
and ``anomaly_score`` (search-time scoring); the first four also expose
``forward`` as a pixel reconstruction (UDMA has no pixel decoder — its
"reconstruction" lives entirely in ``anomaly_score``/``anomaly_map``). A
PyTorch Lightning ``LightningModule`` wraps all five in
``src/training/trainer.py`` for the optimiser/checkpoint/multi-GPU loop. None
trains a classifier — unlike the
ContrastiveVAE + Random Forest of Ma et al., we drop the supervised
contrastive/cadence head, enabling search across a broader morphology space.
See CLAUDE.md ("Relationship to Ma et al.").
"""

import torch
from torch import nn
from typing import Dict, List, Tuple

from .encoder import build_encoder
from .decoder import build_decoder
from .losses import reconstruction_loss, kl_divergence, _masked_mse, topk_mse
from .memory import MemoryUnit
from .vit_mae import build_vit_mae
from .udma import build_udma

__all__ = ["Autoencoder", "MAE", "VAE", "MemAE", "build_autoencoder"]


class Autoencoder(nn.Module):
    """Deterministic convolutional autoencoder scored by reconstruction error."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_fn):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar training loss: mean per-sample reconstruction error."""
        return self.loss_fn(x, self(x)).mean()

    def anomaly_score(self, x: torch.Tensor, method: str = "recon", topk_frac: float = 0.02, **kwargs) -> torch.Tensor:
        if method not in ("recon", "topk"):
            raise ValueError(f"Autoencoder only supports method='recon'/'topk', got '{method}'.")
        recon = self.forward(x)
        if method == "topk":
            return topk_mse(x, recon, frac=topk_frac)
        return ((x - recon) ** 2).mean(dim=(1, 2, 3))


class MAE(nn.Module):
    """CNN Masked Autoencoder.

    Training: randomly masks ``mask_ratio`` fraction of non-overlapping patches
    by zeroing them, then reconstructs the full image. Loss is computed only
    over the masked pixel positions, forcing the model to predict occluded
    regions from surrounding context.

    Inference (``forward``): no masking — standard forward pass. Anomaly score
    is the full-image reconstruction error, identical to the plain
    ``Autoencoder``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn,
        patch_size: Tuple[int, int] = (4, 4),
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def _make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a pixel-level binary mask: 1 = masked, 0 = visible.

        Patches are sampled independently per batch item at the given ratio;
        the exact number of masked patches varies slightly across items, which
        acts as an implicit data-augmentation. Shape ``(B, 1, H, W)`` —
        broadcasts over channels.
        """
        ph, pw = self.patch_size
        b, _, h, w = x.shape
        nh, nw = h // ph, w // pw
        scores = torch.rand(b, nh * nw, device=x.device, dtype=x.dtype)
        mask_patches = (scores < self.mask_ratio).to(x.dtype).view(b, 1, nh, nw)
        return mask_patches.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)

    def _masked_mse(
        self, x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error averaged over masked pixel positions only."""
        return _masked_mse(x, reconstruction, mask)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar training loss: masked-position reconstruction error."""
        mask = self._make_mask(x)
        reconstruction = self(x * (1.0 - mask))
        return self._masked_mse(x, reconstruction, mask)

    def anomaly_score(self, x: torch.Tensor, method: str = "recon", topk_frac: float = 0.02, **kwargs) -> torch.Tensor:
        if method not in ("recon", "topk"):
            raise ValueError(f"MAE only supports method='recon'/'topk', got '{method}'.")
        recon = self.forward(x)
        if method == "topk":
            return topk_mse(x, recon, frac=topk_frac)
        return ((x - recon) ** 2).mean(dim=(1, 2, 3))


class VAE(nn.Module):
    """Variational autoencoder: reconstruction + ``beta`` * KL."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn,
        beta: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.beta = beta
        self.beta_target = beta  # annealing endpoint; beta is mutated by the trainer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, z = self.encoder(x)
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return ``(total, {"reconstruction_loss":..., "kl_loss":...})``.

        The component dict preserves the three Keras metric trackers as
        loggable scalars for the Lightning trainer.
        """
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        recon_loss = self.loss_fn(x, reconstruction).mean()
        kl_loss = kl_divergence(z_mean, z_log_var).mean()
        total = recon_loss + self.beta * kl_loss
        return total, {
            "reconstruction_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return z_mean (B, latent_dim) — deterministic embedding for one-class scoring."""
        z_mean, _, _ = self.encoder(x)
        return z_mean

    def anomaly_score(self, x: torch.Tensor, method: str = "recon", topk_frac: float = 0.02, **kwargs) -> torch.Tensor:
        if method not in ("recon", "topk"):
            raise ValueError(f"VAE only supports method='recon'/'topk', got '{method}'.")
        recon = self.forward(x)
        if method == "topk":
            return topk_mse(x, recon, frac=topk_frac)
        return ((x - recon) ** 2).mean(dim=(1, 2, 3))


class MemAE(nn.Module):
    """Memory-augmented autoencoder (Gong et al. 2019).

    Wraps the deterministic CNN encoder/decoder with a :class:`MemoryUnit`
    addressed per spatial position of the latent feature map. Trained on normal
    data only, the memory records prototypical normal patterns; an anomalous
    encoding is replaced by the nearest normal prototypes before decoding, so the
    decoder cannot redraw the anomaly and its reconstruction error is amplified.
    This breaks the "copy" failure mode of the plain :class:`Autoencoder`, whose
    spatial bottleneck lets a locally-regular signal (e.g. a narrowband line)
    pass straight through to the decoder.

    Scored at search time by reconstruction error (``method='recon'``), exactly
    like the plain AE — the memory acts during reconstruction, not on the score.
    ``encode`` is provided only for interface compatibility with the
    embedding-based eval harness / ``OneClassScorer``; it is **not** the intended
    anomaly score for this model.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        memory: MemoryUnit,
        loss_fn,
        entropy_weight: float = 2.0e-4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.memory = memory
        self.loss_fn = loss_fn
        self.entropy_weight = entropy_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat, _ = self.memory(self.encoder(x))
        return self.decoder(z_hat)

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return ``(total, {"reconstruction_loss":..., "entropy_loss":...})``.

        ``total = reconstruction + entropy_weight * entropy(addressing_weights)``
        (Gong Eq. 10). The entropy term, together with the hard shrinkage inside
        the memory unit, promotes sparse memory addressing.
        """
        z_hat, att = self.memory(self.encoder(x))
        reconstruction = self.decoder(z_hat)
        recon_loss = self.loss_fn(x, reconstruction).mean()
        entropy_loss = self.memory.entropy(att)
        total = recon_loss + self.entropy_weight * entropy_loss
        return total, {
            "reconstruction_loss": recon_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled encoder output ``(B, latent_dim)``.

        Interface shim for the embedding-based harness/``OneClassScorer`` only —
        MemAE's anomaly score is reconstruction error, not this embedding.
        """
        return self.encoder(x).mean(dim=(2, 3))

    def latent_residual_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-spatial-position squared distance between the encoder query and the
        memory-addressed reconstruction, ``((z - z_hat) ** 2).mean(dim=channel)``.

        This is the memory's own per-position "how far from any normal prototype"
        signal, measured in latent space *before* the decoder — unlike pixel-space
        recon error, it is never diluted by the ~98k incompressible noise pixels of
        the frame, since it is computed on the ``(H', W')`` bottleneck grid, not the
        full input resolution. ``encode()`` (GAP-pooled) throws this map away; use
        this method directly to keep spatial resolution for anomaly scoring.
        """
        z = self.encoder(x)
        z_hat, _ = self.memory(z)
        return ((z - z_hat) ** 2).mean(dim=1)  # (B, H', W')

    def anomaly_score(self, x: torch.Tensor, method: str = "recon", topk_frac: float = 0.02, **kwargs) -> torch.Tensor:
        if method not in ("recon", "topk", "latent_max", "latent_topk"):
            raise ValueError(
                f"MemAE supports method='recon'/'topk'/'latent_max'/'latent_topk', got '{method}'."
            )
        if method in ("latent_max", "latent_topk"):
            resid = self.latent_residual_map(x).flatten(1)  # (B, H'*W')
            if method == "latent_max":
                return resid.max(dim=1).values
            k = max(1, int(round(topk_frac * resid.shape[1])))
            return resid.topk(k, dim=1).values.mean(dim=1)
        recon = self.forward(x)
        if method == "topk":
            return topk_mse(x, recon, frac=topk_frac)
        return ((x - recon) ** 2).mean(dim=(1, 2, 3))


def build_autoencoder(
    input_shape: Tuple[int, int, int],
    model_config: Dict,
    loss: str = "mse",
    learning_rate: float = 1.0e-3,
    beta: float = 1.0,
) -> nn.Module:
    """Build the anomaly-detection model from a merged config.

    Returns a bare ``nn.Module`` (``Autoencoder`` | ``MAE`` | ``VAE`` |
    ``ViTMAE``); the optimiser/training loop lives in the Lightning trainer,
    which reads ``model.learning_rate`` set here.

    Args:
        input_shape: ``(tchans, fchans, 1)`` snippet shape. Both spatial dims
            must be divisible by ``2 ** len(encoder.filters)`` (and by
            ``patch_size`` when ``mae`` is set). For ``architecture: vit_mae``
            both dims must instead be divisible by ``patch_size`` only.
        model_config: parsed ``configs/model/*.yaml`` (encoder/bottleneck/
            decoder sections + optional ``mae`` / ``variational`` flags; or,
            for ``architecture: vit_mae``, the ViT-MAE hyperparameters
            consumed by ``build_vit_mae``).
        loss: reconstruction loss name (``mse`` | ``ssim`` | ``mse+ssim``).
        learning_rate: Adam learning rate, attached to the returned module.
        beta: KL weight (VAE path only).

    Returns:
        An ``Autoencoder``, ``MAE``, ``VAE``, or ``ViTMAE`` module.
    """
    architecture = model_config.get("architecture", "convae")

    if architecture == "vit_mae":
        model: nn.Module = build_vit_mae(input_shape, model_config, loss=loss, learning_rate=learning_rate)
    elif architecture == "udma":
        model = build_udma(input_shape, model_config)
    else:
        enc_cfg = model_config["encoder"]
        dec_cfg = model_config.get("decoder", {})
        bottleneck = model_config["bottleneck"]

        filters: List[int] = list(enc_cfg["filters"])
        kernel_size = tuple(enc_cfg.get("kernel_size", (3, 3)))
        activation = enc_cfg.get("activation", "relu")
        use_batchnorm = enc_cfg.get("use_batchnorm", True)
        convs_per_block = int(enc_cfg.get("convs_per_block", 2))
        latent_dim = bottleneck["latent_dim"]
        output_activation = dec_cfg.get("output_activation", "sigmoid")
        variational = bool(model_config.get("variational", False))
        mae = bool(model_config.get("mae", False))
        memory = bool(model_config.get("memory", False))
        patch_size = tuple(model_config.get("patch_size", (4, 4)))
        mask_ratio = float(model_config.get("mask_ratio", 0.75))
        mem_slots = int(model_config.get("mem_slots", 500))
        shrink_threshold = model_config.get("shrink_threshold", None)
        entropy_weight = float(model_config.get("entropy_weight", 2.0e-4))

        n_blocks = len(filters)
        factor = 2 ** n_blocks
        th, fw, _ = input_shape
        if th % factor or fw % factor:
            raise ValueError(
                f"Input spatial dims {(th, fw)} must be divisible by {factor} "
                f"(2 ** {n_blocks} downsampling blocks). Pad/crop snippets upstream."
            )
        if mae:
            ph, pw = patch_size
            if th % ph or fw % pw:
                raise ValueError(
                    f"Input spatial dims {(th, fw)} must be divisible by patch_size "
                    f"{patch_size} for MAE patch tokenisation."
                )
        spatial_shape = (th // factor, fw // factor)

        encoder = build_encoder(
            input_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            variational=variational,
        )
        decoder = build_decoder(
            output_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            spatial_shape=spatial_shape,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            output_activation=output_activation,
            convs_per_block=convs_per_block,
            variational=variational,
        )

        loss_fn = reconstruction_loss(loss)
        if mae:
            model = MAE(encoder, decoder, loss_fn, patch_size=patch_size, mask_ratio=mask_ratio)
        elif variational:
            model = VAE(encoder, decoder, loss_fn, beta=beta)
        elif memory:
            mem = MemoryUnit(mem_slots, latent_dim, shrink_threshold=shrink_threshold)
            model = MemAE(encoder, decoder, mem, loss_fn, entropy_weight=entropy_weight)
        else:
            model = Autoencoder(encoder, decoder, loss_fn)

    # Read by the Lightning trainer's configure_optimizers (replaces compile()).
    model.learning_rate = learning_rate
    return model
