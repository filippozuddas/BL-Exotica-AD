"""
Transposed-CNN decoder blocks.

Mirrors the encoder: ``len(filters)`` transposed-conv blocks, each upsampling
2x, restoring the original snippet shape. The pre-upsample spatial shape is
passed in (``spatial_shape``) rather than hardcoded — this is the fix for the
brittle ``Reshape((1, 32, 256))`` in the reference projects, which only worked
for a single fixed input size.

- ``variational=False``: input is the latent feature map ``(latent_dim, H', W')``;
  a 1x1 conv expands it back to ``filters[-1]`` channels before upsampling.
- ``variational=True``: input is the latent vector ``(latent_dim,)``; a linear
  layer projects it to ``filters[-1] * H' * W'`` and reshapes.

Shape arithmetic (NCHW, square odd kernel ``k``, ``pad = k // 2``):
- stride-2 deconv with ``output_padding=1`` → ``out = 2 * in`` (upsample);
- stride-1 refinement deconv with ``output_padding=0`` → dims unchanged.
``output_padding`` must be ``< stride``, so ``op=1`` is valid *only* on the
stride-2 deconv; using it on a stride-1 layer fails at construction.
"""

import torch
from torch import nn
from typing import List, Tuple

from .encoder import _activation

__all__ = ["Decoder", "build_decoder"]


def _deconv_block(
    in_channels: int,
    filters: int,
    kernel_size: Tuple[int, int],
    activation: str,
    use_batchnorm: bool,
    n_conv: int = 2,
) -> nn.Sequential:
    """Stride-2 upsample + (n_conv-1) stride-1 refinement convs.

    Mirrors the encoder block structure: upsample first (stride 2,
    ``output_padding=1`` → ``out = 2 * in``), then refine at the new resolution
    (stride 1, ``output_padding=0``).
    """
    kh, kw = kernel_size
    padding = (kh // 2, kw // 2)
    layers: List[nn.Module] = []
    prev = in_channels
    for i in range(n_conv):
        stride, output_padding = (2, 1) if i == 0 else (1, 0)
        layers.append(
            nn.ConvTranspose2d(
                prev,
                filters,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=not use_batchnorm,
            )
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        layers.append(_activation(activation))
        prev = filters
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    """Transposed-CNN decoder producing the reconstruction."""

    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        filters: List[int],
        latent_dim: int,
        spatial_shape: Tuple[int, int],
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_batchnorm: bool = True,
        output_activation: str = "sigmoid",
        convs_per_block: int = 2,
        variational: bool = False,
    ):
        super().__init__()
        self.variational = variational
        h, w = spatial_shape
        bottleneck_channels = filters[-1]
        self.h, self.w, self.bottleneck_channels = h, w, bottleneck_channels

        if variational:
            # Vector latent → project and reshape to the feature-map grid.
            self.project = nn.Sequential(
                nn.Linear(latent_dim, h * w * bottleneck_channels),
                _activation(activation),
            )
        else:
            # Feature-map latent → 1x1 expand back to bottleneck channels.
            expand: List[nn.Module] = [nn.Conv2d(latent_dim, bottleneck_channels, 1)]
            if use_batchnorm:
                expand.append(nn.BatchNorm2d(bottleneck_channels))
            expand.append(_activation(activation))
            self.expand = nn.Sequential(*expand)

        blocks: List[nn.Module] = []
        prev = bottleneck_channels
        for f in reversed(filters):
            blocks.append(
                _deconv_block(prev, f, kernel_size, activation, use_batchnorm, convs_per_block)
            )
            prev = f
        self.deconv = nn.Sequential(*blocks)

        kh, kw = kernel_size
        self.out = nn.Conv2d(prev, output_shape[-1], kernel_size, padding=(kh // 2, kw // 2))
        self.out_activation = _activation(output_activation) if output_activation else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.variational:
            x = self.project(z).view(-1, self.bottleneck_channels, self.h, self.w)
        else:
            x = self.expand(z)
        x = self.deconv(x)
        return self.out_activation(self.out(x))


def build_decoder(
    output_shape: Tuple[int, int, int],
    filters: List[int],
    latent_dim: int,
    spatial_shape: Tuple[int, int],
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    output_activation: str = "sigmoid",
    convs_per_block: int = 2,
    variational: bool = False,
) -> Decoder:
    """Build the transposed-CNN decoder.

    Args:
        output_shape: ``(tchans, fchans, 1)`` target reconstruction shape.
        filters: same encoder-order list; reversed internally for upsampling.
        latent_dim: bottleneck channels (AE) or latent vector size (VAE).
        spatial_shape: ``(H', W')`` feature-map size after the encoder stack.
        variational: if True, the latent input is a vector, else a feature map.

    Returns:
        A ``Decoder`` module producing the reconstruction.
    """
    return Decoder(
        output_shape=output_shape,
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
