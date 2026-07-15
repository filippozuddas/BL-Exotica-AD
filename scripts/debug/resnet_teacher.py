"""ResNet-18 (ImageNet, frozen) — "P" in docs/2026-07-14_paper_alignment_plan.md, Fase 2
(D6/D7): the out-of-domain generic backbone the UDMA paper (Qi et al. 2024, Eq. 2) distills
its teacher CNN from. Read at ``layer3`` (256 ch, stride /16 -> native (6,64) grid on a
(96,1024) input) so its token grid matches the ViT-MAE teacher's without any downstream
architecture change.

Exposes the minimal interface ``scripts/debug/teacher_sensitivity_test.py``'s
``encode_tokens_layer()`` expects from a ViT-MAE (``patch_embed``, ``pos_embed``,
``encoder.layers``, ``encoder.norm``) so the SAME gate (G1-G3 thresholds) runs unmodified on
this backbone. P has no transformer blocks of its own — the entire conv stack through
``layer3`` is the ``patch_embed``, and ``encoder.layers`` is empty (nothing further to
iterate).
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

__all__ = ["ResNetTeacher"]


class _EmptyEncoder(nn.Module):
    """Stand-in for ViTMAE's transformer encoder: zero blocks, no final norm —
    P's entire feature extraction happens in ``patch_embed``."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = None


class ResNetTeacher(nn.Module):
    """Frozen ResNet-18 (ImageNet-pretrained), read at ``layer3``.

    Input: ``(B, 1, 96, 1024)`` standardized snippets (this repo's median/MAD
    ``core_transform``) — replicated to 3 channels, with NO ImageNet mean/std
    renormalization. That is a deliberate simplification the gate itself is
    meant to validate (docs/2026-07-14_paper_alignment_plan.md, §5 assumptions),
    not an oversight.
    """

    grid_size = (6, 64)
    channels = 256

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        for p in self.stem.parameters():
            p.requires_grad_(False)
        # No positional embedding: conv receptive fields already encode
        # position, unlike a ViT patch sequence. Plain float broadcasts fine
        # against the token tensor in encode_tokens_layer's `+ pos_embed`.
        self.pos_embed = 0.0
        self.encoder = _EmptyEncoder()
        self.eval()

    def train(self, mode: bool = True) -> "ResNetTeacher":
        return super().train(False)

    @torch.no_grad()
    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 96, 1024) -> (B, nh*nw, channels)`` token sequence, row-major
        over ``(nh, nw)`` — matches ``patch_pixels()``'s patch ordering."""
        x3 = x.repeat(1, 3, 1, 1)
        feat = self.stem(x3)  # (B, channels, nh, nw)
        b, c, nh, nw = feat.shape
        assert (nh, nw) == self.grid_size, (
            f"expected grid {self.grid_size} from a (96,1024) input, got {(nh, nw)}"
        )
        return feat.permute(0, 2, 3, 1).reshape(b, nh * nw, c)
