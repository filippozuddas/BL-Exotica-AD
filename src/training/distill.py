"""Distillation training loop: P (frozen, out-of-domain) -> T (trainable CNN)
+ D (discardable 1x1 channel-alignment head).

Business logic for ``scripts/distill_teacher.py`` (thin CLI wrapper) — Fase 2
of ``docs/2026-07-14_paper_alignment_plan.md`` (5.2), the paper-faithful
teacher route (Qi et al. 2024, Eq. 2 / Bergmann "Uninformed Students"): a
small CNN is trained to reproduce a frozen, out-of-domain generic backbone's
features on in-domain data.

Loss: ``L = ||D(T(x)) - P(x)||^2``, MSE over the full feature grid, no
normalization (raw feature values on both sides).
"""

import time
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import build_encoder

__all__ = ["build_trainable_trunk", "distill_teacher"]


def build_trainable_trunk(
    input_shape: Tuple[int, int, int],
    filters,
    latent_dim: int,
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    convs_per_block: int = 2,
):
    """T's trunk — same parametrisation :class:`~src.models.udma.TeacherCNN`
    uses to load it later, kept as a plain trainable :class:`~src.models.encoder.Encoder`
    here (``TeacherCNN`` itself freezes + pins eval mode, wrong for training)."""
    return build_encoder(
        input_shape=input_shape,
        filters=filters,
        latent_dim=latent_dim,
        kernel_size=kernel_size,
        activation=activation,
        use_batchnorm=use_batchnorm,
        convs_per_block=convs_per_block,
        variational=False,
    )


def distill_teacher(
    P: nn.Module,
    loader,
    input_shape: Tuple[int, int, int],
    device: str,
    filters,
    latent_dim: int,
    p_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    convs_per_block: int = 2,
    epochs: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    log_every_n_steps: int = 50,
    log_fn: Callable[[str], None] = print,
) -> Tuple[nn.Module, List[float]]:
    """Train T (+ a discardable D) to reproduce frozen ``P``'s features.

    Args:
        P: frozen teacher candidate, callable as ``P(x) -> (B, p_channels, nh, nw)``.
        loader: yields ``(B, 1, tchans, fchans)`` batches (no labels).
        input_shape: ``(tchans, fchans, 1)`` — used only to validate T's grid
            matches ``P.grid_size`` before training starts.
        p_channels: ``P``'s output channel count (``D`` projects T's
            ``latent_dim`` up to this for the loss only).

    Returns:
        ``(T, losses)`` — the trained trunk (``D`` is discarded) and the
        per-step loss history (for a curve plot / sanity check).

    Raises:
        ValueError: if T's trunk grid (from ``input_shape``/``filters``)
            doesn't match ``P.grid_size``.
        RuntimeError: if ``loader`` yields zero batches (e.g. ``batch_size``
            larger than the dataset with ``drop_last=True``) — training would
            otherwise silently produce an untrained, unsaved T.
    """
    n_blocks = len(filters)
    t_grid = (input_shape[0] // (2 ** n_blocks), input_shape[1] // (2 ** n_blocks))
    if t_grid != P.grid_size:
        raise ValueError(
            f"T's trunk grid {t_grid} (input {input_shape[:2]} / {2 ** n_blocks}) must match "
            f"P's grid {P.grid_size} -- adjust filters."
        )

    T = build_trainable_trunk(
        input_shape, filters, latent_dim, kernel_size, activation, use_batchnorm, convs_per_block,
    ).to(device)
    D = nn.Conv2d(latent_dim, p_channels, 1).to(device)

    optimizer = torch.optim.AdamW(
        list(T.parameters()) + list(D.parameters()), lr=lr, weight_decay=weight_decay,
    )

    # Accumulate detached GPU scalars (no host sync) rather than calling
    # .item() every step -- forcing a CUDA device-to-host sync on every
    # iteration of the hot loop stalls the async queue. .item() is only
    # called at log_every_n_steps and once, batched, at the very end.
    loss_tensors: List[torch.Tensor] = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                target = P(x)

            pred = D(T(x))
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tensors.append(loss.detach())
            if step % log_every_n_steps == 0:
                elapsed = time.time() - t0
                log_fn(f"  epoch {epoch} step {step}  loss={loss.item():.4f}  "
                       f"({elapsed:.0f}s elapsed)")
            step += 1

        if not loss_tensors:
            raise RuntimeError(
                "Distillation loader yielded zero batches -- check --batch_size against "
                "the dataset size (drop_last=True silently drops an only/too-small batch)."
            )
        log_fn(f"Epoch {epoch} done, {step} steps, last loss={loss_tensors[-1].item():.4f}")

    losses = torch.stack(loss_tensors).cpu().tolist()
    return T, losses
