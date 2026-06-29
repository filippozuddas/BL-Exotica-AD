"""
CNN encoder blocks.

The convolutional stack is identical for the deterministic autoencoder and the
variational (VAE) variant — only the bottleneck differs:

- ``variational=False`` (default): a 1x1 conv compresses the final feature map
  to ``latent_dim`` channels, keeping the spatial layout. Output: a single
  latent feature map ``z`` of shape ``(latent_dim, H', W')``.
- ``variational=True``: the feature map is flattened and projected to vector
  ``z_mean`` / ``z_log_var`` heads, then reparameterised. Output:
  ``(z_mean, z_log_var, z)`` with ``z`` of shape ``(latent_dim,)``.

Block structure follows Ma et al.: each downsampling step consists of a
stride-2 conv followed by (convs_per_block - 1) stride-1 refinement convs at
the new resolution. This gives depth without additional memory on the feature
maps. Ma et al. use 2-3 convs per block across 4 downsampling steps (9 total).

Tensors are NCHW; the channels-last (NHWC) shape ``(tchans, fchans, 1)`` used
in the configs is interpreted here as ``(C=1, H=tchans, W=fchans)``.
"""

import torch
from torch import nn
from typing import List, Tuple

__all__ = ["Encoder", "build_encoder"]


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _activation(name: str) -> nn.Module:
    """Resolve an activation name (Keras took the string directly)."""
    key = name.lower().strip()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Expected one of {sorted(_ACTIVATIONS)}."
        )
    return _ACTIVATIONS[key]()


def _conv_block(
    in_channels: int,
    filters: int,
    kernel_size: Tuple[int, int],
    activation: str,
    use_batchnorm: bool,
    n_conv: int = 2,
) -> nn.Sequential:
    """
    Stride-2 downsample + (n_conv-1) stride-1 refinement convs.

    The first conv halves H and W (stride 2, symmetric padding ``k // 2`` gives
    ``out = in / 2`` for even, odd-kernel inputs); subsequent convs refine
    features at the new resolution without further spatial compression.
    """
    kh, kw = kernel_size
    padding = (kh // 2, kw // 2)
    layers: List[nn.Module] = []
    prev = in_channels
    for i in range(n_conv):
        stride = 2 if i == 0 else 1
        layers.append(
            nn.Conv2d(
                prev,
                filters,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        layers.append(_activation(activation))
        prev = filters
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    CNN encoder. Single latent feature map (AE) or ``(z_mean, z_log_var, z)``
    vector heads (VAE).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters: List[int],
        latent_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_batchnorm: bool = True,
        convs_per_block: int = 2,
        variational: bool = False,
    ):
        super().__init__()
        self.variational = variational
        th, fw, in_channels = input_shape

        blocks: List[nn.Module] = []
        prev = in_channels
        for f in filters:
            blocks.append(
                _conv_block(prev, f, kernel_size, activation, use_batchnorm, convs_per_block)
            )
            prev = f
        self.conv = nn.Sequential(*blocks)

        if not variational:
            # 1x1 spatial bottleneck: keep H', W', compress channels to latent_dim.
            self.z = nn.Conv2d(filters[-1], latent_dim, 1)
        else:
            # Vector bottleneck: the flatten size must be computed explicitly
            # (Keras inferred it from the graph; PyTorch cannot).
            factor = 2 ** len(filters)
            hp, wp = th // factor, fw // factor
            flat = filters[-1] * hp * wp
            self.z_mean = nn.Linear(flat, latent_dim)
            self.z_log_var = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if not self.variational:
            return self.z(x)  # (B, latent_dim, H', W')
        x = torch.flatten(x, 1)
        z_mean = self.z_mean(x)
        # Clamp prevents z_log_var from diverging when beta=0 (KL annealing).
        z_log_var = self.z_log_var(x).clamp(-4.0, 4.0)
        # Reparameterisation trick: z = z_mean + exp(0.5 * z_log_var) * eps.
        z = z_mean + torch.exp(0.5 * z_log_var) * torch.randn_like(z_mean)
        return z_mean, z_log_var, z


def build_encoder(
    input_shape: Tuple[int, int, int],
    filters: List[int],
    latent_dim: int,
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    convs_per_block: int = 2,
    variational: bool = False,
) -> Encoder:
    """
    Build the CNN encoder.

    Args:
        input_shape: ``(tchans, fchans, 1)`` snippet shape (interpreted NCHW).
        filters: one entry per downsampling block (each halves H and W).
        latent_dim: bottleneck channels (AE) or latent vector size (VAE).
        convs_per_block: conv layers per block — first is stride-2, the rest
            are stride-1 refinement. Ma et al. use 2-3 per block.
        variational: if True, emit ``(z_mean, z_log_var, z)`` vector heads.

    Returns:
        An ``Encoder`` module. Single output ``z`` (AE) or a 3-tuple (VAE).
    """
    return Encoder(
        input_shape=input_shape,
        filters=filters,
        latent_dim=latent_dim,
        kernel_size=kernel_size,
        activation=activation,
        use_batchnorm=use_batchnorm,
        convs_per_block=convs_per_block,
        variational=variational,
    )
