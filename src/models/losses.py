"""
Reconstruction loss variants and the VAE KL term.

The autoencoder is the anomaly scorer: training minimises a reconstruction
loss over real GBT noise/RFI, and at search time the same per-sample error is
the anomaly score. All reconstruction losses here return a *per-sample* tensor
of shape ``(batch,)`` so the model can average them (and so callers can rank
snippets directly).

Tensors are NCHW (``(batch, channels, height, width)``) throughout, the PyTorch
convention; reductions over the image are therefore over dims ``(1, 2, 3)``.
"""

import torch

__all__ = [
    "mse_loss",
    "ssim_loss",
    "mse_ssim_loss",
    "reconstruction_loss",
    "kl_divergence",
    "_masked_mse",
]


def mse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared error, averaged over (C, H, W)."""
    return ((y_true - y_pred) ** 2).mean(dim=(1, 2, 3))


def ssim_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
              data_range: float | None = None) -> torch.Tensor:
    """Per-sample structural dissimilarity ``1 - SSIM``.

    ``pytorch_msssim.ssim`` returns higher-is-better similarity; with
    ``size_average=False`` it is per-sample, so the loss is ``1 - SSIM``.

    ``data_range``: value range of the inputs, used to set SSIM stability
    constants c1/c2. If None, computed per-batch from ``y_true``. Pass
    ``data_range=1.0`` only when inputs are genuinely in ``[0, 1]``; for
    MAD-normalised spectrograms leave it None so the constants scale correctly.

    ``pytorch-msssim`` is imported lazily so the module (and the default MSE
    path) loads with only ``torch`` installed.
    """
    from pytorch_msssim import ssim

    if data_range is None:
        data_range = float((y_true.detach().max() - y_true.detach().min()).clamp_min(1e-6).item())
    return 1.0 - ssim(y_true, y_pred, data_range=data_range, size_average=False)


def mse_ssim_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.84
) -> torch.Tensor:
    """Convex blend of MSE and SSIM losses (``alpha`` weights the SSIM term)."""
    return (1.0 - alpha) * mse_loss(y_true, y_pred) + alpha * ssim_loss(
        y_true, y_pred
    )


def reconstruction_loss(name: str = "mse"):
    """Return a per-sample reconstruction-loss callable by name.

    Args:
        name: one of ``"mse"``, ``"ssim"``, ``"mse+ssim"`` (matches the
            ``training.loss`` field in the training config).
    """
    table = {
        "mse": mse_loss,
        "ssim": ssim_loss,
        "mse+ssim": mse_ssim_loss,
    }
    key = name.lower().strip()
    if key not in table:
        raise ValueError(
            f"Unknown reconstruction loss '{name}'. "
            f"Expected one of {sorted(table)}."
        )
    return table[key]


def _masked_mse(x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error averaged over masked pixel positions only.

    ``mask``: binary, broadcastable to ``x``/``reconstruction`` (typically
    ``(B, 1, H, W)``, 1 = masked/scored, 0 = visible/ignored). Shared by CNN
    ``MAE`` (pixel-zeroing mask) and ``ViTMAE`` (token-removal mask, expanded
    to pixel space via repeat_interleave) — identical arithmetic.
    """
    per_pixel = (x - reconstruction) ** 2 * mask
    return per_pixel.sum() / (mask.sum() + 1e-8)


def kl_divergence(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
    """Per-sample KL divergence between ``N(z_mean, exp(z_log_var))`` and the
    unit Gaussian, summed over latent dimensions. Shape ``(batch,)``.

    Only used on the variational (VAE) path; the deterministic autoencoder
    never calls this.
    """
    kl = -0.5 * (1.0 + z_log_var - z_mean ** 2 - torch.exp(z_log_var))
    return kl.sum(dim=1)
