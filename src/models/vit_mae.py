"""
ViT-MAE (SSAST-style): Vision-Transformer Masked Autoencoder for anomaly detection.

4th ``build_autoencoder`` backbone (``architecture: vit_mae``), alongside the CNN
``Autoencoder`` / ``MAE`` / ``VAE`` in ``autoencoder.py``. The design follows
SSAST (Gong et al. 2022, arXiv:2110.09784) rather than He et al.'s MAE:

- **In-place (BERT-style) masking, single transformer.** Every patch token is
  fed to one ``nn.TransformerEncoder``; masked positions have their patch
  embedding replaced by a shared learnable ``mask_token`` (then positional
  embedding is added to all). There is no separate *encoder*-side decoder (the
  reconstruction head reads the encoder output directly). This keeps the full
  encoder representation available at masked positions — required by the
  discriminative head — and means ``encode(x)`` (no masking) is the *same* token
  regime as training, so the encoder embeddings are meaningful (the embedding
  scoring path depends on this).

- **Optional cross-attention decoder** (``decoder_depth > 0``). The default
  reconstruction head is a per-token MLP that reconstructs each patch
  independently; setting ``decoder_depth`` swaps it for a lightweight transformer
  decoder over *all* tokens, adding cross-token attention so adjacent patches
  coordinate their reconstructions. It refines the (already full) encoder output
  ``O`` — no decoder-side mask token is needed. ``loss_weighting: variance``
  independently upweights high-variance (RFI) patches in the generative loss.

- **Joint discriminative + generative objective** (SSAST Algorithm 1). Two
  2-layer MLP heads read the encoder output ``O_i`` at masked positions:
  ``reconstruction_head`` -> ``r_i`` (generative, MSE) and ``classification_head``
  -> ``c_i`` (discriminative, InfoNCE: identify the correct raw patch among all
  masked patches of the *same* spectrogram). Total loss ``L = L_d + lambda*L_g``.

- **MSPM cluster masking.** ``mask_mode: cluster`` masks ``C x C`` patch blocks
  (``C ~ unif{c_min, c_max}`` per step) instead of scattered random patches,
  forcing both local (small C) and global (large C) structure. ``mask_mode:
  random`` keeps He-style scattered masking.

Three inference anomaly scores (``anomaly_score``), all from one trained model:
``recon`` (partitioned reconstruction MSE — anti-copy, for morphology-rich
products), ``infonce`` (per-patch self-recognition difficulty), and
``embedding`` (encoder features + an external one-class classifier — the
default, independent of reconstruction magnitude; see
``src/search/scorer.py``).

``input_shape = (H, W, C)`` matches the ``build_autoencoder`` convention (NOT
``(C, H, W)``); tensors passed to ``forward``/``compute_loss``/``encode`` are NCHW.

Patch index ``k = i*nw + j`` (``i`` = time/tchans row, ``j`` = freq/fchans col)
is the single source of truth: it underlies ``PatchEmbed``'s Conv2d-flatten,
``patchify``/``unpatchify``, the positional embedding, and all mask bookkeeping.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple

__all__ = ["PatchEmbed", "ViTMAE", "build_vit_mae", "patchify", "unpatchify"]


class PatchEmbed(nn.Module):
    """Conv2d patch tokeniser: ``(B,C,H,W) -> (B, nh*nw, embed_dim)``."""

    def __init__(self, patch_size: Tuple[int, int] = (16, 16), in_chans: int = 1, embed_dim: int = 128):
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


def _sample_random_masked_ids(
    batch_size: int,
    num_patches: int,
    n_masked: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """He-style scattered masking: ``(B, n_masked)`` int64 masked-patch indices."""
    noise = torch.rand(batch_size, num_patches, device=device, generator=generator)
    return noise.argsort(dim=1)[:, :n_masked]


def _sample_cluster_masked_ids(
    batch_size: int,
    nh: int,
    nw: int,
    n_masked: int,
    c_min: int,
    c_max: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """SSAST MSPM cluster masking: ``(B, n_masked)`` int64 masked-patch indices.

    Per sample, stamp ``C x C`` blocks (``C ~ unif{c_min, c_max}``, fresh per
    block, clipped to the grid) at random seed positions until at least
    ``n_masked`` patches are covered, then trim to exactly ``n_masked`` (row-major
    order; the surplus is only the final block's overshoot). The fixed count per
    sample keeps the batch rectangular for the InfoNCE objective.

    Sampling/stamping is done in NumPy on the CPU and transferred to ``device``
    once — the per-block Python loop must not touch the GPU (a per-block
    ``.item()`` sync would dominate the step time and throttle training). A
    single int is drawn from ``generator`` to seed NumPy so the result stays
    reproducible under Lightning's ``seed_everything``.
    """
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item())
    rng = np.random.default_rng(seed)
    out = np.empty((batch_size, n_masked), dtype=np.int64)
    for b in range(batch_size):
        grid = np.zeros((nh, nw), dtype=bool)
        while grid.sum() < n_masked:
            c = int(rng.integers(c_min, c_max + 1))
            si = int(rng.integers(0, nh))
            sj = int(rng.integers(0, nw))
            grid[si:si + c, sj:sj + c] = True  # NumPy slicing clips at the grid edge
        out[b] = np.flatnonzero(grid.ravel())[:n_masked]
    return torch.from_numpy(out).to(device)


def _partition_groups(num_patches: int, n_groups: int, device: torch.device) -> list:
    """Disjoint round-robin partition of ``[0, N)`` into ``n_groups`` groups.

    Used by partitioned inference (``forward`` / ``infonce`` scoring): each pass
    keeps one group *visible* and masks the rest, so every patch is predicted
    from context that never includes itself (anti-copy), at a mask ratio of
    ``(n_groups-1)/n_groups`` matching training.
    """
    ids = torch.arange(num_patches, device=device)
    return [ids[g::n_groups] for g in range(n_groups)]


class ViTMAE(nn.Module):
    """SSAST-style ViT masked autoencoder with a joint discriminative+generative loss."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        mask_ratio: float = 0.75,
        loss_mode: str = "joint",
        mask_mode: str = "cluster",
        cluster_factor: Tuple[int, int] = (3, 5),
        infonce_lambda: float = 10.0,
        infonce_temperature: float = 0.07,
        scoring: str = "embedding",
        norm_pix_loss: bool = False,
        cadence_n_obs: int = 6,
        cadence_on_obs: Tuple[int, ...] = (0, 2, 4),
        decoder_depth: int = 0,
        decoder_dim: int = 64,
        decoder_num_heads: int = 4,
        loss_weighting: str = "none",
        noise_sigma: float = 0.1,
    ):
        super().__init__()
        h, w, c = input_shape
        ph, pw = patch_size
        if h % ph or w % pw:
            raise ValueError(
                f"Input spatial dims {(h, w)} must be divisible by patch_size "
                f"{patch_size} for ViT-MAE patch tokenisation."
            )
        if loss_mode not in ("generative", "discriminative", "joint", "denoising"):
            raise ValueError(f"Unknown loss_mode '{loss_mode}'.")
        if mask_mode not in ("random", "cluster"):
            raise ValueError(f"Unknown mask_mode '{mask_mode}'.")
        if loss_weighting not in ("none", "variance"):
            raise ValueError(f"Unknown loss_weighting '{loss_weighting}'.")

        self.input_shape = (c, h, w)
        self.patch_size = patch_size
        nh, nw = h // ph, w // pw
        self.grid_size = (nh, nw)
        self.num_patches = nh * nw
        self.patch_dim = ph * pw * c
        self.mask_ratio = mask_ratio
        self.loss_mode = loss_mode
        self.mask_mode = mask_mode
        self.cluster_factor = (int(cluster_factor[0]), int(cluster_factor[1]))
        self.infonce_lambda = infonce_lambda
        self.infonce_temperature = infonce_temperature
        self.scoring = scoring
        self.norm_pix_loss = norm_pix_loss
        self._discriminative = loss_mode in ("discriminative", "joint")
        self.cadence_n_obs = cadence_n_obs
        self.cadence_on_obs = tuple(cadence_on_obs)
        self.decoder_depth = int(decoder_depth)
        self.loss_weighting = loss_weighting
        self.noise_sigma = float(noise_sigma)

        self.patch_embed = PatchEmbed(patch_size, in_chans=c, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False)

        # Reconstruction path: per-token MLP head (decoder_depth=0, default) or a
        # lightweight transformer decoder (decoder_depth>0) that adds cross-token
        # attention so adjacent patches coordinate their reconstructions. The two
        # are mutually exclusive; _reconstruct() dispatches on decoder_depth.
        if self.decoder_depth == 0:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, self.patch_dim)
            )
        else:
            self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_dim * mlp_ratio,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder_blocks = nn.TransformerEncoder(
                decoder_layer, num_layers=self.decoder_depth,
                norm=nn.LayerNorm(decoder_dim), enable_nested_tensor=False)
            self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)
        if self._discriminative:
            self.classification_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, self.patch_dim)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if self.decoder_depth > 0:
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    @property
    def n_masked(self) -> int:
        return int(round(self.mask_ratio * self.num_patches))

    # ---- core encode path (in-place mask substitution) ----

    def _encode(self, x: torch.Tensor, mask_bool: Optional[torch.Tensor]) -> torch.Tensor:
        """``(B,C,H,W)`` + optional ``(B,N)`` bool mask -> encoder output ``(B,N,De)``.

        Masked positions (``mask_bool`` True) get their patch embedding replaced
        by the shared ``mask_token``; positional embedding is then added to all.
        ``mask_bool=None`` means no masking (used by ``encode``).
        """
        tokens = self.patch_embed(x)  # (B,N,De)
        if mask_bool is not None:
            m = mask_bool.unsqueeze(-1).to(tokens.dtype)
            tokens = tokens * (1.0 - m) + self.mask_token * m
        tokens = tokens + self.pos_embed
        return self.encoder(tokens)

    def _mask_from_ids(self, ids_masked: torch.Tensor) -> torch.Tensor:
        """``(B, n_masked)`` indices -> ``(B, N)`` bool mask (True = masked)."""
        b = ids_masked.shape[0]
        mask = torch.zeros(b, self.num_patches, dtype=torch.bool, device=ids_masked.device)
        mask.scatter_(1, ids_masked, True)
        return mask

    def _sample_masked_ids(self, b: int, device: torch.device,
                           generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.mask_mode == "cluster":
            nh, nw = self.grid_size
            return _sample_cluster_masked_ids(
                b, nh, nw, self.n_masked, self.cluster_factor[0], self.cluster_factor[1],
                device, generator)
        return _sample_random_masked_ids(b, self.num_patches, self.n_masked, device, generator)

    # ---- training ----

    def _generative_loss(self, pred_patches: torch.Tensor, target_patches: torch.Tensor,
                         mask_bool: torch.Tensor) -> torch.Tensor:
        """Masked per-patch MSE (mean over masked patches). ``mask_bool`` is ``(B,N)``.

        With ``loss_weighting='variance'`` each masked patch's error is weighted by
        its *target* variance (no gradient through the weight — it only redistributes
        gradient across patches), upweighting high-variance structured RFI patches so
        the model learns to reconstruct them instead of predicting the noise mean.
        Normalised by the weight sum (weighted mean) to keep the loss scale stable.
        """
        per_patch = ((pred_patches - target_patches) ** 2).mean(dim=-1)  # (B,N)
        m = mask_bool.to(per_patch.dtype)
        if self.loss_weighting == "variance":
            m = m * target_patches.var(dim=-1)  # (B,N) — upweight high-variance (RFI) patches
        return (per_patch * m).sum() / (m.sum() + 1e-8)

    def _reconstruct(self, O: torch.Tensor) -> torch.Tensor:
        """``(B,N,De) -> (B,N,patch_dim)``: per-token MLP head or transformer decoder.

        ``O`` is the encoder output at *every* position (SSAST in-place masking
        already substituted ``mask_token`` pre-encoding), so the decoder needs no
        mask-token of its own — it just refines all tokens with cross-token attention.
        """
        if self.decoder_depth == 0:
            return self.reconstruction_head(O)
        z = self.decoder_embed(O) + self.decoder_pos_embed
        z = self.decoder_blocks(z)
        return self.decoder_pred(z)

    def _infonce_loss(self, O: torch.Tensor, target_patches: torch.Tensor,
                      ids_masked: torch.Tensor) -> torch.Tensor:
        """SSAST discriminative loss: identify the correct raw patch among masked.

        For each masked position the classification head predicts ``c_i``; the
        InfoNCE logits are ``c_i . x_j`` over all masked ``j`` in the *same*
        sample (negatives), with the diagonal as the positive. Both are
        L2-normalised (cosine logits in [-1, 1]); ``infonce_temperature`` must be
        small (~0.07) so the softmax can concentrate over the ~M candidates —
        otherwise the loss is floored near ``log(M)`` and the objective is inert.
        """
        d = O.shape[-1]
        c = self.classification_head(
            O.gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, d))
        )  # (B, M, patch_dim)
        x = target_patches.gather(
            1, ids_masked.unsqueeze(-1).expand(-1, -1, self.patch_dim)
        )  # (B, M, patch_dim)
        c = F.normalize(c, dim=-1)
        x = F.normalize(x, dim=-1)
        logits = torch.bmm(c, x.transpose(1, 2)) / self.infonce_temperature  # (B, M, M)
        b, m, _ = logits.shape
        labels = torch.arange(m, device=logits.device).expand(b, m).reshape(-1)
        return F.cross_entropy(logits.reshape(b * m, m), labels)

    def compute_loss(self, x: torch.Tensor):
        """Training loss.

        Returns a scalar for ``generative``/``discriminative`` ``loss_mode``, and a
        ``(total, {"recon_loss":…, "infonce_loss":…})`` tuple for ``joint`` — the
        tuple path is handled by the Lightning trainer exactly like the VAE.
        """
        if self.norm_pix_loss:
            raise NotImplementedError("norm_pix_loss=True is not yet implemented")

        if self.loss_mode == "denoising":
            x_noisy = x + torch.randn_like(x) * self.noise_sigma
            # All-False mask keeps mask_token in the computation graph (× 0) so
            # DDP never sees it as an unused parameter across any loss_mode.
            mask_bool = torch.zeros(x.shape[0], self.num_patches, dtype=torch.bool, device=x.device)
            O = self._encode(x_noisy, mask_bool=mask_bool)
            pred_patches = self._reconstruct(O)
            target_patches = patchify(x, self.patch_size)
            return F.mse_loss(pred_patches, target_patches)

        b = x.shape[0]
        ids_masked = self._sample_masked_ids(b, x.device)
        mask_bool = self._mask_from_ids(ids_masked)
        O = self._encode(x, mask_bool)
        target_patches = patchify(x, self.patch_size)

        if self.loss_mode == "discriminative":
            return self._infonce_loss(O, target_patches, ids_masked)

        pred_patches = self._reconstruct(O)
        recon = self._generative_loss(pred_patches, target_patches, mask_bool)
        if self.loss_mode == "generative":
            return recon

        infonce = self._infonce_loss(O, target_patches, ids_masked)
        total = infonce + self.infonce_lambda * recon
        return total, {"recon_loss": recon, "infonce_loss": infonce}

    # ---- inference ----

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled encoder embedding ``(B, embed_dim)`` (no masking).

        Feature extractor for the ``embedding`` anomaly score (one-class
        classifier). Because training uses in-place masking, the unmasked
        forward here is the same token regime minus the substitution.
        """
        O = self._encode(x, mask_bool=None)
        return O.mean(dim=1)

    def _n_groups(self) -> int:
        """Partition group count whose mask ratio ``(G-1)/G`` matches ``mask_ratio``."""
        return max(2, int(round(1.0 / (1.0 - self.mask_ratio))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic partitioned reconstruction ``(B,C,H,W)``.

        Each of ``G`` passes keeps one round-robin group visible and masks the
        rest; a patch's reconstruction is averaged over the passes where it was
        masked. Used by the ``recon`` anomaly score.
        """
        b = x.shape[0]
        groups = _partition_groups(self.num_patches, self._n_groups(), x.device)
        accum = torch.zeros(b, self.num_patches, self.patch_dim, device=x.device, dtype=x.dtype)
        counts = torch.zeros(self.num_patches, device=x.device, dtype=x.dtype)
        for visible in groups:
            mask_bool = torch.ones(b, self.num_patches, dtype=torch.bool, device=x.device)
            mask_bool[:, visible] = False
            O = self._encode(x, mask_bool)
            pred_patches = self._reconstruct(O)
            masked_f = mask_bool[0].to(x.dtype)  # same group structure across batch
            accum += pred_patches * masked_f.view(1, -1, 1)
            counts += masked_f
        pred_patches = accum / counts.view(1, -1, 1).clamp_min(1.0)
        return unpatchify(pred_patches, self.patch_size, (b, *self.input_shape))

    def _cadence_on_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """``(B, N)`` bool mask: True for all ON-observation patches."""
        nh, nw = self.grid_size
        tchans_per_obs = (nh * self.patch_size[0]) // self.cadence_n_obs
        rows_per_obs = tchans_per_obs // self.patch_size[0]
        if rows_per_obs < 1 or nh < self.cadence_n_obs:
            raise ValueError(
                f"Cadence scoring requires grid height nh={nh} >= cadence_n_obs="
                f"{self.cadence_n_obs} and patch_size[0]={self.patch_size[0]} "
                f"to divide evenly into observation boundaries."
            )
        mask = torch.zeros(self.num_patches, dtype=torch.bool, device=device)
        for obs_idx in self.cadence_on_obs:
            start_row = obs_idx * rows_per_obs
            for r in range(start_row, start_row + rows_per_obs):
                mask[r * nw : (r + 1) * nw] = True
        return mask.unsqueeze(0).expand(batch_size, -1)

    def _cadence_score(self, x: torch.Tensor) -> torch.Tensor:
        """Cadence-aware anomaly score: mask all ON, reconstruct from OFF context.

        Single forward pass. Score = MSE on ON (masked) patches only.
        """
        b = x.shape[0]
        mask_bool = self._cadence_on_mask(b, x.device)
        O = self._encode(x, mask_bool)
        pred_patches = self._reconstruct(O)
        target_patches = patchify(x, self.patch_size)
        on_mask = mask_bool[0].float()
        diff_sq = (pred_patches - target_patches).pow(2).mean(dim=-1)
        return (diff_sq * on_mask).sum(dim=1) / on_mask.sum()

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, method: Optional[str] = None, occ=None) -> torch.Tensor:
        """Per-sample anomaly score ``(B,)``.

        ``method``: ``recon`` (partitioned reconstruction MSE), ``cadence``
        (mask-ON/reconstruct-from-OFF), ``infonce`` (per-patch self-recognition
        difficulty), or ``embedding`` (one-class classifier ``occ`` on
        ``encode(x)``). Defaults to ``self.scoring``.
        """
        method = method or self.scoring
        if method == "recon":
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2, 3))
        if method == "cadence":
            return self._cadence_score(x)
        if method == "infonce":
            return self._infonce_score(x)
        if method == "embedding":
            if occ is None:
                raise ValueError("method='embedding' requires a fitted one-class classifier `occ`.")
            return torch.as_tensor(occ.score(self.encode(x).cpu().numpy()), device=x.device)
        raise ValueError(f"Unknown scoring method '{method}'.")

    def _infonce_score(self, x: torch.Tensor) -> torch.Tensor:
        """Partitioned per-patch self-recognition difficulty, averaged per sample.

        For each masked patch the discriminative head must recognise its own raw
        patch among the masked set; the negative log-probability of the correct
        match is the per-patch anomaly score. Requires the classification head.
        """
        if not self._discriminative:
            raise ValueError("infonce scoring requires loss_mode 'discriminative' or 'joint'.")
        b = x.shape[0]
        target_patches = patchify(x, self.patch_size)
        groups = _partition_groups(self.num_patches, self._n_groups(), x.device)
        score = torch.zeros(b, device=x.device)
        counts = 0
        for visible in groups:
            mask_bool = torch.ones(b, self.num_patches, dtype=torch.bool, device=x.device)
            mask_bool[:, visible] = False
            ids_masked = mask_bool[0].nonzero(as_tuple=False).squeeze(1).unsqueeze(0).expand(b, -1)
            O = self._encode(x, mask_bool)
            d = O.shape[-1]
            c = F.normalize(self.classification_head(
                O.gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, d))), dim=-1)
            xt = F.normalize(target_patches.gather(
                1, ids_masked.unsqueeze(-1).expand(-1, -1, self.patch_dim)), dim=-1)
            logits = torch.bmm(c, xt.transpose(1, 2)) / self.infonce_temperature  # (B,M,M)
            m = logits.shape[1]
            logp = F.log_softmax(logits, dim=-1)
            diag = logp.diagonal(dim1=1, dim2=2)  # (B, M) log p(correct)
            score += (-diag).mean(dim=1)
            counts += 1
        return score / max(counts, 1)


def build_vit_mae(
    input_shape: Tuple[int, int, int],
    model_config: Dict,
    loss: str = "mse",
    learning_rate: float = 1.0e-3,
) -> ViTMAE:
    """Build a ``ViTMAE`` from a merged ``configs/model/vit_mae.yaml``.

    ``loss``/``learning_rate`` are accepted only for call-site parity with the
    other ``build_autoencoder`` branches; ``ViTMAE`` hardcodes its objective
    (masked-patch MSE + InfoNCE). ``model.learning_rate`` is set by
    ``build_autoencoder``, not here.
    """
    return ViTMAE(
        input_shape=input_shape,
        patch_size=tuple(model_config.get("patch_size", (16, 16))),
        embed_dim=int(model_config.get("embed_dim", 128)),
        depth=int(model_config.get("depth", 6)),
        num_heads=int(model_config.get("num_heads", 4)),
        mlp_ratio=int(model_config.get("mlp_ratio", 4)),
        mask_ratio=float(model_config.get("mask_ratio", 0.75)),
        loss_mode=str(model_config.get("loss_mode", "joint")),
        mask_mode=str(model_config.get("mask_mode", "cluster")),
        cluster_factor=tuple(model_config.get("cluster_factor", (3, 5))),
        infonce_lambda=float(model_config.get("infonce_lambda", 10.0)),
        infonce_temperature=float(model_config.get("infonce_temperature", 0.07)),
        scoring=str(model_config.get("scoring", "embedding")),
        norm_pix_loss=bool(model_config.get("norm_pix_loss", False)),
        cadence_n_obs=int(model_config.get("cadence_n_obs", 6)),
        cadence_on_obs=tuple(model_config.get("cadence_on_obs", (0, 2, 4))),
        decoder_depth=int(model_config.get("decoder_depth", 0)),
        decoder_dim=int(model_config.get("decoder_dim", 64)),
        decoder_num_heads=int(model_config.get("decoder_num_heads", 4)),
        loss_weighting=str(model_config.get("loss_weighting", "none")),
        noise_sigma=float(model_config.get("noise_sigma", 0.1)),
    )
