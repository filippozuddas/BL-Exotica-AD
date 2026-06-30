"""
Memory module for the memory-augmented autoencoder (MemAE).

Implements the content-addressable memory of Gong et al., "Memorizing Normality
to Detect Anomaly" (ICCV 2019, https://donggong1.github.io/anomdec-memae).

The memory holds ``N`` prototype vectors of dimension ``C``. The encoder output
is used as a *query*: each query position retrieves a sparse, attention-weighted
combination of memory items, and that combination — not the raw encoding — is
passed to the decoder. Trained on normal data only, the memory records
prototypical *normal* patterns; at test time an anomalous encoding is replaced
by the nearest normal prototypes, so the decoder cannot reproduce the anomaly
and the reconstruction error is amplified.

This is the **per-pixel** addressing variant (Gong §3.3 / §4.2, used there for
conv feature maps): the memory addresses each spatial position of the encoder's
latent feature map ``(B, C, H', W')`` independently, with ``C = latent_dim``.
This matches the spatial bottleneck of the deterministic CNN autoencoder in this
repo (``encoder.py``: a 1x1 conv keeps the spatial layout) — which is exactly
the bottleneck that lets a plain AE copy a narrowband line through to the
decoder. Restricting the decoder to normal prototypes breaks that passthrough.

Verified against the authors' official implementation
(``models/memory_module.py`` of donggong1/memae-anomaly-detection): the addressing
is **dot-product** (not the cosine of the paper text), the memory init is
``uniform(-1/sqrt(C), 1/sqrt(C))``, and the shrinkage / L1-renorm / entropy match.
The one deliberate deviation is ``eps`` (1e-6 vs the reference 1e-12) to stay
numerically safe under bf16-mixed training.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["MemoryUnit"]


class MemoryUnit(nn.Module):
    """Attention-based sparse memory addressing (Gong et al. 2019).

    Args:
        mem_slots: number of memory items ``N`` (Gong: insensitive to ``N`` once
            large enough; 500 is a sound default at this scale).
        feature_dim: item dimension ``C`` — must equal the encoder's
            ``latent_dim`` (the channel count of the spatial bottleneck).
        shrink_threshold: hard-shrinkage threshold ``lambda`` (Eq. 7). Paper
            range ``[1/N, 3/N]``; default ``1/N``.
        eps: numerical floor for the shrinkage division and the entropy log.
    """

    def __init__(
        self,
        mem_slots: int,
        feature_dim: int,
        shrink_threshold: float | None = None,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.mem_slots = mem_slots
        self.feature_dim = feature_dim
        # Default lambda = 1/N (lower end of the paper's [1/N, 3/N] range).
        self.shrink_threshold = (
            float(shrink_threshold) if shrink_threshold is not None else 1.0 / mem_slots
        )
        self.eps = eps

        # Memory bank M in R^{N x C}; learned jointly with encoder/decoder.
        self.memory = nn.Parameter(torch.empty(mem_slots, feature_dim))
        # Match the official MemAE init: uniform(-1/sqrt(C), 1/sqrt(C)).
        stdv = 1.0 / math.sqrt(feature_dim)
        nn.init.uniform_(self.memory, -stdv, stdv)

    def _hard_shrink(self, w: torch.Tensor) -> torch.Tensor:
        """Continuous ReLU form of the hard-shrinkage operator (Gong Eq. 7).

        ``w_hat_i = max(w_i - lambda, 0) * w_i / (|w_i - lambda| + eps)``.
        """
        lam = self.shrink_threshold
        return F.relu(w - lam) * w / (torch.abs(w - lam) + self.eps)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Address the memory per spatial position.

        Args:
            z: encoder latent feature map ``(B, C, H, W)`` with ``C = feature_dim``.

        Returns:
            ``(z_hat, w_hat)`` where ``z_hat`` is the memory-reconstructed feature
            map ``(B, C, H, W)`` for the decoder, and ``w_hat`` is the sparse
            addressing weight matrix ``(B*H*W, N)`` (for the entropy loss).
        """
        b, c, h, w = z.shape
        # Flatten spatial positions into independent queries: (B*H*W, C).
        query = z.permute(0, 2, 3, 1).reshape(-1, c)

        # Dot-product attention + softmax over memory items. NOTE: Gong's paper
        # (Eq. 4-5) describes cosine similarity, but the authors' official code
        # uses a raw dot product (`F.linear(query, memory)`, no normalisation);
        # we follow the released implementation, not the paper text.
        sim = F.linear(query, self.memory)  # (B*H*W, N)
        att = F.softmax(sim, dim=1)

        # Hard shrinkage + L1 renormalisation to induce a sparse combination (Eq. 7).
        att = self._hard_shrink(att)
        att = att / (att.sum(dim=1, keepdim=True) + self.eps)

        # Reconstruct the latent from the addressed memory items (Eq. 3).
        z_hat = att @ self.memory  # (B*H*W, C)
        z_hat = z_hat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return z_hat, att

    def entropy(self, att: torch.Tensor) -> torch.Tensor:
        """Sparsity-promoting entropy of the addressing weights (Gong Eq. 9).

        ``E(w_hat) = mean_positions( sum_i -w_hat_i * log(w_hat_i) )``.
        """
        return torch.mean(torch.sum(-att * torch.log(att + self.eps), dim=1))
