"""
ViT-MAE: Vision-Transformer Masked Autoencoder (He et al.-style).

4th ``build_autoencoder`` backbone (``architecture: vit_mae``), alongside the
CNN ``Autoencoder`` / ``MAE`` / ``VAE`` in ``autoencoder.py``. A Conv2d
``PatchEmbed`` tokenises the spectrogram into non-overlapping patches; an
``nn.TransformerEncoder`` processes a random subset of "visible" tokens; a
lightweight decoder reconstructs every patch from the visible tokens plus a
shared learnable mask token. Training loss is masked-patch MSE, computed via
the same ``losses._masked_mse`` used by the CNN ``MAE``.

``input_shape = (H, W, C)`` matches the ``build_autoencoder``/``build_encoder``
convention (NOT ``(C, H, W)``); tensors passed to ``forward``/``compute_loss``
are NCHW.

Patch index ``k = i*nw + j`` (``i`` = time/tchans row, ``j`` = freq/fchans
col) is the single source of truth: it underlies ``PatchEmbed``'s
Conv2d-flatten, ``patchify``/``unpatchify``, both positional embeddings, the
mask ``.view(B, 1, nh, nw)``, and ``_partition_ids``'s row slicing. This is
what makes the patch-mask <-> pixel-mask equivalence hold.
"""

import torch
from torch import nn
from typing import Dict, Optional, Tuple

from .losses import _masked_mse

__all__ = ["PatchEmbed", "ViTMAE", "build_vit_mae", "patchify", "unpatchify"]


class PatchEmbed(nn.Module):
    """Conv2d patch tokeniser: ``(B,C,H,W) -> (B, nh*nw, embed_dim)``."""

    def __init__(self, patch_size: Tuple[int, int] = (4, 64), in_chans: int = 1, embed_dim: int = 128):
        super().__init__()
        ph, pw = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def patchify(x: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """``(B,C,H,W) -> (B, N, ph*pw*C)``.

    Patch ``k = i*nw + j`` holds pixel block
    ``x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]`` flattened in ``(ph, pw, C)``
    order. Pure reshape/permute — bit-exact inverse of ``unpatchify``.
    """
    ph, pw = patch_size
    b, c, h, w = x.shape
    nh, nw = h // ph, w // pw
    x = x.reshape(b, c, nh, ph, nw, pw)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, nh, nw, ph, pw, C)
    return x.reshape(b, nh * nw, ph * pw * c)


def unpatchify(patches: torch.Tensor, patch_size: Tuple[int, int], shape: Tuple[int, int, int, int]) -> torch.Tensor:
    """``(B, N, ph*pw*C) -> (B,C,H,W)``, exact inverse of ``patchify``."""
    ph, pw = patch_size
    b, c, h, w = shape
    nh, nw = h // ph, w // pw
    x = patches.reshape(b, nh, nw, ph, pw, c)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, nh, ph, nw, pw)
    return x.reshape(b, c, h, w)


def _sample_random_ids(
    batch_size: int,
    num_patches: int,
    len_keep: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """He-style random token removal: exactly ``len_keep`` visible per sample.

    Returns ``(ids_keep, ids_restore)``: ``ids_keep`` is ``(B, len_keep)``
    int64 indices into ``[0, N)``; ``ids_restore`` is ``(B, N)`` int64 such
    that gathering ``cat([x_vis, mask_tokens], dim=1)`` with ``ids_restore``
    recovers the original ``k=0..N-1`` patch order.
    """
    noise = torch.rand(batch_size, num_patches, device=device, generator=generator)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    return ids_keep, ids_restore


def _partition_ids(batch_size: int, nh: int, nw: int, group: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic inference partition: patch row ``group`` is "visible".

    Same ``(ids_keep, ids_restore)`` contract as ``_sample_random_ids``, used
    by ``ViTMAE.forward``'s ``nh``-pass partitioned reconstruction.
    """
    num_patches = nh * nw
    visible = torch.arange(group * nw, (group + 1) * nw, device=device)
    masked = torch.cat([torch.arange(0, group * nw, device=device),
                         torch.arange((group + 1) * nw, num_patches, device=device)])
    ids_shuffle = torch.cat([visible, masked])
    ids_restore = ids_shuffle.argsort().unsqueeze(0).expand(batch_size, -1)
    ids_keep = visible.unsqueeze(0).expand(batch_size, -1)
    return ids_keep, ids_restore


class ViTMAE(nn.Module):
    """ViT Masked Autoencoder, scored by masked-patch reconstruction error.

    Training (``compute_loss``): a fresh random subset of
    ``N * (1 - mask_ratio)`` patch tokens is encoded; the decoder reconstructs
    all ``N`` patches from those tokens plus a shared mask token; loss is the
    pixel-space masked MSE over the dropped patches only.

    Inference (``forward``): deterministic ``nh``-pass partitioned
    reconstruction — each grid row is, in turn, the "visible" group — so every
    patch is reconstructed from context that never includes itself.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int] = (4, 64),
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mlp_ratio: int = 4,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
    ):
        super().__init__()
        h, w, c = input_shape
        ph, pw = patch_size
        if h % ph or w % pw:
            raise ValueError(
                f"Input spatial dims {(h, w)} must be divisible by patch_size "
                f"{patch_size} for ViT-MAE patch tokenisation."
            )

        self.input_shape = (c, h, w)
        self.patch_size = patch_size
        nh, nw = h // ph, w // pw
        self.grid_size = (nh, nw)
        self.num_patches = nh * nw
        self.patch_dim = ph * pw * c
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.patch_embed = PatchEmbed(patch_size, in_chans=c, embed_dim=embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim))

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth, norm=nn.LayerNorm(decoder_embed_dim))
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _decode_from_keep(self, x: torch.Tensor, ids_keep: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Shared encode-decode path: ``(B,C,H,W)`` -> ``(B,N,patch_dim)``."""
        embed_dim = self.encoder_pos_embed.shape[-1]
        dec_dim = self.decoder_pos_embed.shape[-1]
        b = x.shape[0]

        tokens = self.patch_embed(x) + self.encoder_pos_embed          # (B,N,De)
        x_vis = tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim))
        x_vis = self.encoder(x_vis)                                     # (B,K,De)

        x_vis = self.decoder_embed(x_vis)                               # (B,K,Dd)
        mask_tokens = self.mask_token.expand(b, self.num_patches - x_vis.shape[1], -1)
        x_full = torch.cat([x_vis, mask_tokens], dim=1)                 # shuffled order
        x_full = x_full.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, dec_dim))
        x_full = x_full + self.decoder_pos_embed                        # original order
        x_full = self.decoder(x_full)
        return self.decoder_pred(x_full)                                # (B,N,patch_dim)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar training loss: masked-patch reconstruction error."""
        if self.norm_pix_loss:
            raise NotImplementedError("norm_pix_loss=True is not yet implemented")
        b, n = x.shape[0], self.num_patches
        len_keep = n - int(self.mask_ratio * n)
        ids_keep, ids_restore = _sample_random_ids(b, n, len_keep, device=x.device)
        pred_patches = self._decode_from_keep(x, ids_keep, ids_restore)

        mask = torch.ones(b, n, device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = mask.gather(1, ids_restore)  # (B,N), 1=masked, original patch order

        nh, nw = self.grid_size
        ph, pw = self.patch_size
        pred = unpatchify(pred_patches, self.patch_size, (b, *self.input_shape))
        pixel_mask = mask.view(b, 1, nh, nw).repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)
        return _masked_mse(x, pred, pixel_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic ``nh``-pass partitioned reconstruction."""
        b, n = x.shape[0], self.num_patches
        nh, nw = self.grid_size
        accum = torch.zeros(b, n, self.patch_dim, device=x.device, dtype=x.dtype)
        counts = torch.zeros(n, device=x.device, dtype=x.dtype)
        for g in range(nh):
            ids_keep, ids_restore = _partition_ids(b, nh, nw, g, x.device)
            pred_patches = self._decode_from_keep(x, ids_keep, ids_restore)
            masked = torch.ones(n, device=x.device, dtype=x.dtype)
            masked[g * nw:(g + 1) * nw] = 0.0
            accum += pred_patches * masked.view(1, n, 1)
            counts += masked
        pred_patches = accum / counts.view(1, n, 1)  # counts == nh-1 everywhere
        return unpatchify(pred_patches, self.patch_size, (b, *self.input_shape))


def build_vit_mae(
    input_shape: Tuple[int, int, int],
    model_config: Dict,
    loss: str = "mse",
    learning_rate: float = 1.0e-3,
) -> ViTMAE:
    """Build a ``ViTMAE`` from a merged ``configs/model/vit_mae.yaml``.

    ``loss``/``learning_rate`` are accepted only for call-site parity with the
    other ``build_autoencoder`` branches — ``ViTMAE.compute_loss`` always uses
    ``_masked_mse`` (hardcoded MSE), same as the CNN ``MAE``.
    ``model.learning_rate`` is set by ``build_autoencoder``, not here.
    """
    return ViTMAE(
        input_shape=input_shape,
        patch_size=tuple(model_config.get("patch_size", (4, 64))),
        embed_dim=int(model_config.get("embed_dim", 128)),
        depth=int(model_config.get("depth", 6)),
        num_heads=int(model_config.get("num_heads", 4)),
        decoder_embed_dim=int(model_config.get("decoder_embed_dim", 64)),
        decoder_depth=int(model_config.get("decoder_depth", 2)),
        decoder_num_heads=int(model_config.get("decoder_num_heads", 4)),
        mlp_ratio=int(model_config.get("mlp_ratio", 4)),
        mask_ratio=float(model_config.get("mask_ratio", 0.75)),
        norm_pix_loss=bool(model_config.get("norm_pix_loss", False)),
    )
