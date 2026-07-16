"""
UDMA — Unsupervised Distillation and Memory-enhanced Autoencoder.

5th ``build_autoencoder`` backbone (``architecture: udma``). Adapts Qi et al.
2024 ("Unsupervised Spectrum Anomaly Detection With Distillation and Memory
Enhanced Autoencoders", IEEE IoT Journal 11(24):39361) to this repo: two
lightweight CNN "students" are trained to regress the frozen, self-supervised
ViT-MAE encoder's token features (not pixels). Disagreement between what the
students predict and what the teacher actually produced is the anomaly signal
— on in-distribution RFI/noise the students learn to agree with the teacher
(and, via the ``map_ss`` term, with each other); on never-seen morphology they
cannot, and the prediction gap opens up. This is a feature-space generalisation
of the validated pixel-space probe ``‖AE(x)−MemAE(x)‖²``
(``scripts/debug/encode_separation_test.py``, ``DisagreementPair``), attacking
its diagnosed failure mode (recon-MSE measures predictability, not
anomalousness — the RFI tail swamps the operating point below SNR 10) by
moving the score off the pixel manifold where that tail lives.

Full design rationale, gate results, and pre-registered acceptance bars:
``docs/2026-07-05_udma_design_spec.md`` (Q1-Q10). Teacher fitness (G1-G3b) was
verified before this module was written (``scripts/debug/teacher_sensitivity_test.py``);
do not skip that gate for a new teacher checkpoint/layer.

Components:
- ``TeacherViT``: frozen ``ViTMAE.encode_tokens_at`` wrapper. Reshapes the
  ``(B, 384, 128)`` token sequence (patch grid 6x64 over a (96,1024) input) to
  ``(B, 128, 6, 64)`` and applies an offline per-channel Norm (``fit_normalization``).
- ``FeatureStudent``: the same CNN trunk as ``Autoencoder``/``MemAE``
  (``build_encoder``, 4 blocks -> 16x spatial reduction) produces
  ``(B, 64, 6, 64)`` — the SAME (6,64) grid as the teacher tokens, by
  construction (patch_size (16,16) on (96,1024) == encoder's 16x downsampling).
  A small projection head maps this to ``(B, 128, 6, 64)`` to match the
  teacher's channel count. ``memory=True`` inserts the validated
  ``MemoryUnit`` between trunk and head (the "MemAE" student).
- ``UDMA``: teacher + two students (AE-student, MemAE-student); joint loss
  (Q4), three disagreement maps + fused score (Q5).

No pixel decoder anywhere — the "decoder" for each student is the projection
head onto the shared token grid; this is what eliminates the pixel-space
dilution (384 positions vs ~98k pixels) that limited every prior recon scorer.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .encoder import build_encoder
from .memory import MemoryUnit
from .vit_mae import ViTMAE, build_vit_mae

__all__ = ["TeacherViT", "TeacherCNN", "FeatureStudent", "UDMA", "build_udma"]

_ROOT = Path(__file__).resolve().parents[2]


class _FrozenTokenTeacher(nn.Module):
    """Shared base for a frozen, Norm-ed token-grid teacher: buffers
    (``mu``/``sigma``), eval-mode pinning, ``forward``, and
    ``fit_normalization`` are identical regardless of what produces the raw
    ``(B, channels, nh, nw)`` grid (a ViT-MAE block or a distilled CNN trunk)
    — subclasses set ``self.channels``/``self.grid_size`` and implement
    ``_raw_tokens`` in ``__init__``, and MUST call ``self.eval()`` at the end
    of their own ``__init__`` (registering the ``mu``/``sigma`` buffers here
    doesn't pin eval mode by itself — a subclass with BatchNorm layers left
    in the default ``training=True`` state would normalize with per-batch
    stats instead of the frozen running stats until something up the module
    hierarchy calls ``.train()``/``.eval()``, which does not happen for
    standalone use like the teacher-sensitivity gate scripts).
    """

    def train(self, mode: bool = True) -> "_FrozenTokenTeacher":
        return super().train(False)

    def _raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B, channels, nh, nw)``, before Norm. Override in subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._raw_tokens(x)
        return (raw - self.mu.view(1, -1, 1, 1)) / (self.sigma.view(1, -1, 1, 1) + 1e-6)

    @torch.no_grad()
    def fit_normalization(self, loader, device: torch.device, max_batches: Optional[int] = None) -> None:
        """Offline per-channel mean/std of the raw (pre-Norm) token/feature grid
        over ``loader``.

        One-shot (Q2/Q7): call once on the training set, then the resulting
        ``mu``/``sigma`` buffers are saved with the UDMA checkpoint.
        """
        self.eval()
        total = torch.zeros(self.channels, dtype=torch.float64, device=device)
        total_sq = torch.zeros(self.channels, dtype=torch.float64, device=device)
        count = 0
        for i, x in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            raw = self._raw_tokens(x).double()  # (B, D, nh, nw)
            flat = raw.permute(1, 0, 2, 3).reshape(self.channels, -1)
            total += flat.sum(dim=1)
            total_sq += (flat ** 2).sum(dim=1)
            count += flat.shape[1]
        mean = total / count
        var = (total_sq / count - mean ** 2).clamp_min(0.0)
        self.mu.copy_(mean.to(self.mu.dtype))
        self.sigma.copy_(var.sqrt().to(self.sigma.dtype))


class TeacherViT(_FrozenTokenTeacher):
    """Frozen ViT-MAE encoder, read at an intermediate transformer block.

    Wraps a checkpointed :class:`ViTMAE` (never trained further — parameters
    have ``requires_grad=False`` and the module is pinned in ``eval()`` mode
    regardless of the parent Lightning module's train/eval state, since it has
    no BatchNorm to need train-mode statistics). ``forward`` returns the
    Norm-ed token grid ``(B, 128, 6, 64)`` that is the UDMA students' regression
    target.
    """

    def __init__(self, vit: ViTMAE, teacher_layer: int = 3):
        super().__init__()
        self.vit = vit
        self.teacher_layer = teacher_layer
        for p in self.vit.parameters():
            p.requires_grad_(False)
        nh, nw = self.vit.grid_size
        self.grid_size = (nh, nw)
        self.channels = self.vit.patch_embed.proj.out_channels
        self.register_buffer("mu", torch.zeros(self.channels))
        self.register_buffer("sigma", torch.ones(self.channels))
        self.eval()

    def _raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B, channels, nh, nw)``, before Norm."""
        nh, nw = self.grid_size
        tokens = self.vit.encode_tokens_at(x, layer=self.teacher_layer)  # (B, nh*nw, D)
        b, n, d = tokens.shape
        return tokens.reshape(b, nh, nw, d).permute(0, 3, 1, 2)  # (B, D, nh, nw)


class TeacherCNN(_FrozenTokenTeacher):
    """Frozen CNN teacher distilled from a generic pretrained backbone P
    (paper-faithful route, Qi et al. 2024 Eq. 2 / Bergmann "Uninformed
    Students" — teacher feature space anchored out-of-domain by construction;
    spectrum data enters only as distillation input, never as a learning
    target for P itself). Trunk = :func:`build_encoder` with the same
    parametrisation as the UDMA students (``docs/2026-07-14_paper_alignment_plan.md``,
    D8) — its own ``latent_dim``-channel 1x1 projection already lands on the
    target channel count, no separate head needed.

    Trained once by ``scripts/distill_teacher.py`` (frozen P, MSE on a
    (6,64) grid via a distillation-only 1x1 projection D, discarded after
    training) then loaded here as a fixed teacher — same
    ``forward``/``grid_size``/``channels``/``mu``/``sigma``/``fit_normalization``
    interface as :class:`TeacherViT` (both via :class:`_FrozenTokenTeacher`),
    so :func:`build_udma` only needs to branch at construction time
    (``teacher.type: cnn_distilled``).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters,
        latent_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_batchnorm: bool = True,
        convs_per_block: int = 2,
    ):
        super().__init__()
        self.trunk = build_encoder(
            input_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            variational=False,
        )
        for p in self.trunk.parameters():
            p.requires_grad_(False)

        n_blocks = len(filters)
        factor = 2 ** n_blocks
        th, fw, _ = input_shape
        self.grid_size = (th // factor, fw // factor)
        self.channels = latent_dim
        self.register_buffer("mu", torch.zeros(self.channels))
        self.register_buffer("sigma", torch.ones(self.channels))
        # Required: trunk has BatchNorm (use_batchnorm=True by default) and
        # this class is also used standalone (e.g. teacher_sensitivity_test.py
        # --architecture cnn_distilled), which never calls .eval() itself —
        # without this, BatchNorm normalizes with per-batch stats instead of
        # the running stats learned during distillation. See class docstring.
        self.eval()

    def _raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B, channels, nh, nw)``, before Norm."""
        return self.trunk(x)


class FeatureStudent(nn.Module):
    """CNN trunk (shared with ``Autoencoder``/``MemAE``) + projection head onto
    the teacher's token grid, with an optional :class:`MemoryUnit`.

    The trunk's spatial output ``(B, latent_dim, H', W')`` already sits on the
    same ``(H', W')`` grid as the teacher tokens (both come from a 16x
    downsampling of a (96,1024) input) — no upsampling/pooling is needed to
    align them.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters,
        latent_dim: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_batchnorm: bool = True,
        convs_per_block: int = 2,
        memory: bool = False,
        mem_slots: int = 500,
        shrink_threshold: Optional[float] = None,
        head_hidden: int = 128,
        head_context_convs: int = 2,
    ):
        super().__init__()
        self.trunk = build_encoder(
            input_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            variational=False,
        )
        self.memory = MemoryUnit(mem_slots, latent_dim, shrink_threshold=shrink_threshold) if memory else None

        head_layers = []
        prev = latent_dim
        for _ in range(head_context_convs):
            head_layers += [nn.Conv2d(prev, head_hidden, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)), nn.ReLU(inplace=True)]
            prev = head_hidden
        head_layers.append(nn.Conv2d(prev, out_channels, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        att = None
        if self.memory is not None:
            z, att = self.memory(z)
        out = self.head(z)
        return (out, att) if self.memory is not None else out


class UDMA(nn.Module):
    """Teacher-student distillation + memory ensemble, scored by feature-space
    disagreement (Q4/Q5).

    ``compute_loss`` trains the two students to regress ``Norm(T(x))``
    (``map_st1``/``map_st2``) while also minimising their mutual disagreement
    on normal data (``map_ss``); at inference, the SAME three squared-error
    maps become the anomaly signal — the AE-student can only copy through what
    it was trained on, the MemAE-student can only redraw learned normal
    prototypes, and neither can track token features it has never regressed
    correctly, so unseen morphology reopens a gap that training closed for
    normal data.
    """

    def __init__(
        self,
        teacher: TeacherViT,
        student_ae: FeatureStudent,
        student_mem: FeatureStudent,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        entropy_weight: float = 2.0e-4,
        score_weights: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        topk_frac: float = 0.02,
    ):
        super().__init__()
        self.teacher = teacher
        self.student_ae = student_ae
        self.student_mem = student_mem
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.entropy_weight = entropy_weight
        self.score_weights = score_weights
        self.topk_frac = topk_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "UDMA has no pixel decoder/reconstruction — use anomaly_score(x, method=...) "
            "or anomaly_map(x), not model(x)."
        )

    def _forward_students(self, x: torch.Tensor):
        s_ae = self.student_ae(x)  # (B, 128, nh, nw), no memory
        s_mem, att = self.student_mem(x)  # memory=True -> (out, att)
        return s_ae, s_mem, att

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target = self.teacher(x)  # (B, 128, nh, nw), frozen, no grad by construction
        s_ae, s_mem, att = self._forward_students(x)

        st1 = F.mse_loss(s_ae, target)
        st2 = F.mse_loss(s_mem, target)
        ss = F.mse_loss(s_ae, s_mem)
        entropy = self.student_mem.memory.entropy(att)

        total = self.lambda1 * st1 + self.lambda2 * st2 + self.lambda3 * ss + self.entropy_weight * entropy
        return total, {
            "st1": st1.detach(),
            "st2": st2.detach(),
            "ss": ss.detach(),
            "entropy": entropy.detach(),
            # Early-stopping target (Q7): S-T regression quality only, excluding
            # ss/entropy so a degenerate student pair (both collapsed, agreeing
            # everywhere) can't look like convergence.
            "st_sum": (st1 + st2).detach(),
        }

    @torch.no_grad()
    def anomaly_map_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The three disagreement maps (Q5) plus their fusion, each ``(B, nh, nw)``.

        Exposed separately from ``anomaly_map`` for qualitative inspection
        (``scripts/debug/udma_anomaly_maps.py``) — e.g. checking whether the
        teacher's global attention smears ``st1``/``st2`` into diffuse maps
        (the Q1 risk) even where the fused ``cob`` map still looks localised.
        """
        target = self.teacher(x)
        s_ae, s_mem, _ = self._forward_students(x)
        w1, w2, w3 = self.score_weights
        map_st1 = (target - s_ae).pow(2).mean(dim=1)
        map_st2 = (target - s_mem).pow(2).mean(dim=1)
        map_ss = (s_ae - s_mem).pow(2).mean(dim=1)
        map_cob = w1 * map_st1 + w2 * map_st2 + w3 * map_ss
        return {"st1": map_st1, "st2": map_st2, "ss": map_ss, "cob": map_cob}

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Fused disagreement map ``(B, nh, nw)`` (``map_cob``, Q5)."""
        return self.anomaly_map_components(x)["cob"]

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, method: str = "topk", topk_frac: Optional[float] = None, **kwargs) -> torch.Tensor:
        """Per-sample anomaly score ``(B,)`` from ``map_cob`` (Q5).

        ``method``: ``recon`` (mean over the (nh,nw) grid — duck-type name for
        harness compatibility, ``encode_separation_test.py --scoring recon``),
        ``topk`` (top-``topk_frac`` grid positions, default), or ``max``.
        """
        if method not in ("recon", "topk", "max"):
            raise ValueError(f"UDMA supports method='recon'/'topk'/'max', got '{method}'.")
        m = self.anomaly_map(x).flatten(1)  # (B, nh*nw)
        if method == "recon":
            return m.mean(dim=1)
        if method == "max":
            return m.max(dim=1).values
        frac = topk_frac if topk_frac is not None else self.topk_frac
        k = max(1, int(round(frac * m.shape[1])))
        return m.topk(k, dim=1).values.mean(dim=1)


def _load_teacher_vit(vit_config_path: Path, checkpoint_path: Path, input_shape: Tuple[int, int, int]) -> ViTMAE:
    with open(vit_config_path) as f:
        vit_config = yaml.safe_load(f)
    vit = build_vit_mae(input_shape, vit_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    vit.load_state_dict(state)
    return vit


def _load_teacher_cnn(teacher_cfg: Dict, input_shape: Tuple[int, int, int]) -> TeacherCNN:
    """Build a :class:`TeacherCNN` and load a trunk-only checkpoint saved by
    ``scripts/distill_teacher.py`` (``{"trunk_state_dict", "filters",
    "latent_dim", "convs_per_block", ...}`` — no distillation projection D).

    Architecture comes from the checkpoint itself (self-describing — the
    single source of truth for what was actually distilled), never silently
    from ``teacher_cfg``: a stale/mismatched config value could otherwise
    load a checkpoint's weights into the wrong architecture without any
    shape error (e.g. ``activation`` differs but produces identical tensor
    shapes). ``kernel_size``/``activation``/``use_batchnorm`` fall back to
    :class:`TeacherCNN`'s own defaults only for checkpoints saved before
    2026-07-16 (when those keys were added) — their true value was always
    that default.
    """
    checkpoint_path = _ROOT / teacher_cfg["checkpoint"]
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    missing = [k for k in ("filters", "latent_dim", "convs_per_block") if k not in ckpt]
    if missing:
        raise KeyError(
            f"Distilled-teacher checkpoint {checkpoint_path} is missing required "
            f"architecture key(s) {missing} — was it saved by scripts/distill_teacher.py? "
            f"Re-run distillation."
        )
    teacher = TeacherCNN(
        input_shape,
        filters=list(ckpt["filters"]),
        latent_dim=int(ckpt["latent_dim"]),
        kernel_size=tuple(ckpt.get("kernel_size", (3, 3))),
        activation=ckpt.get("activation", "relu"),
        use_batchnorm=bool(ckpt.get("use_batchnorm", True)),
        convs_per_block=int(ckpt["convs_per_block"]),
    )
    teacher.trunk.load_state_dict(ckpt["trunk_state_dict"])
    return teacher


def build_udma(
    input_shape: Tuple[int, int, int],
    model_config: Dict,
) -> UDMA:
    """Build a :class:`UDMA` from a merged ``configs/model/udma.yaml``.

    Args:
        input_shape: ``(tchans, fchans, 1)``, must match the teacher's
            ``vit_config``'s ``patch_size`` divisibility (Q6: v1 = (96,1024)).
        model_config: parsed ``udma.yaml`` — ``teacher`` (checkpoint + vit
            config path + ``teacher_layer``), ``student`` (encoder section,
            reusing the ``convae.yaml`` schema), memory hyperparameters, loss
            lambdas, and scoring weights/``topk_frac``.
    """
    teacher_cfg = model_config["teacher"]
    teacher_type = teacher_cfg.get("type", "vit_mae")
    if teacher_type == "cnn_distilled":
        # Paper-faithful route (D6-D8): teacher distilled from an out-of-domain
        # generic backbone (scripts/distill_teacher.py), not self-supervised
        # in-domain. See docs/2026-07-14_paper_alignment_plan.md, Fase 2.
        teacher = _load_teacher_cnn(teacher_cfg, input_shape)
    elif teacher_type == "vit_mae":
        vit_config_path = _ROOT / teacher_cfg["vit_config"]
        checkpoint_path = _ROOT / teacher_cfg["checkpoint"]
        teacher_layer = int(teacher_cfg.get("teacher_layer", 3))
        vit = _load_teacher_vit(vit_config_path, checkpoint_path, input_shape)
        teacher = TeacherViT(vit, teacher_layer=teacher_layer)
    else:
        raise ValueError(f"teacher.type must be 'vit_mae' or 'cnn_distilled', got '{teacher_type}'.")

    norm_stats_path = teacher_cfg.get("norm_stats")
    if norm_stats_path is not None:
        # Precomputed offline (Q2/Q7 one-shot): scripts/fit_udma_teacher_norm.py.
        # Without this, mu/sigma stay at their identity default (0/1) and
        # training regresses raw, unnormalized teacher features.
        stats = torch.load(str(_ROOT / norm_stats_path), map_location="cpu")
        teacher.mu.copy_(stats["mu"])
        teacher.sigma.copy_(stats["sigma"])

    student_cfg = model_config["student"]
    enc_cfg = student_cfg["encoder"]
    filters = list(enc_cfg["filters"])
    kernel_size = tuple(enc_cfg.get("kernel_size", (3, 3)))
    activation = enc_cfg.get("activation", "relu")
    use_batchnorm = enc_cfg.get("use_batchnorm", True)
    convs_per_block = int(enc_cfg.get("convs_per_block", 2))
    latent_dim = int(student_cfg["bottleneck"]["latent_dim"])
    head_hidden = int(student_cfg.get("head_hidden", 128))
    head_context_convs = int(student_cfg.get("head_context_convs", 2))
    mem_slots = int(student_cfg.get("mem_slots", 500))
    shrink_threshold = student_cfg.get("shrink_threshold", None)

    n_blocks = len(filters)
    factor = 2 ** n_blocks
    th, fw, _ = input_shape
    if th % factor or fw % factor:
        raise ValueError(
            f"Input spatial dims {(th, fw)} must be divisible by {factor} "
            f"(2 ** {n_blocks} student downsampling blocks)."
        )
    student_grid = (th // factor, fw // factor)
    if student_grid != teacher.grid_size:
        raise ValueError(
            f"Student trunk grid {student_grid} (input {(th, fw)} / {factor}) must "
            f"match the teacher's patch grid {teacher.grid_size} (Q3) — the two feed "
            f"the same disagreement maps. Adjust student.encoder.filters or the ViT "
            f"patch_size so both downsample to the same (H', W')."
        )
    out_channels = teacher.channels

    def _make_student(memory: bool) -> FeatureStudent:
        return FeatureStudent(
            input_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            memory=memory,
            mem_slots=mem_slots,
            shrink_threshold=shrink_threshold,
            head_hidden=head_hidden,
            head_context_convs=head_context_convs,
        )

    student_ae = _make_student(memory=False)
    student_mem = _make_student(memory=True)

    loss_cfg = model_config.get("loss", {})
    score_cfg = model_config.get("scoring", {})

    return UDMA(
        teacher=teacher,
        student_ae=student_ae,
        student_mem=student_mem,
        lambda1=float(loss_cfg.get("lambda1", 1.0)),
        lambda2=float(loss_cfg.get("lambda2", 1.0)),
        lambda3=float(loss_cfg.get("lambda3", 1.0)),
        entropy_weight=float(loss_cfg.get("entropy_weight", 2.0e-4)),
        score_weights=tuple(score_cfg.get("weights", (0.5, 0.5, 0.5))),
        topk_frac=float(score_cfg.get("topk_frac", 0.02)),
    )
