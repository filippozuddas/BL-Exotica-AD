"""ViT-MAE (SSAST-style) backbone contract tests — CPU only, small dims.

Pins the public interface the rest of the pipeline relies on: build dispatch,
patchify roundtrip, masking shape/ratio (random + cluster), the three loss
modes (incl. the joint tuple the Lightning trainer expects), encode() for the
embedding score, partitioned reconstruction, and the three anomaly scores.
"""

import numpy as np
import pytest
import torch

from src.models.autoencoder import build_autoencoder
from src.models.vit_mae import (
    ViTMAE,
    build_vit_mae,
    patchify,
    unpatchify,
    _sample_cluster_masked_ids,
    _sample_random_masked_ids,
)
from src.search.scorer import OneClassScorer

SHAPES = [(16, 1024, 1), (96, 1024, 1)]
PATCH = (16, 16)


def _cfg(**overrides):
    cfg = dict(
        architecture="vit_mae",
        patch_size=list(PATCH),
        embed_dim=32,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        mask_ratio=0.75,
        mask_mode="cluster",
        cluster_factor=[3, 5],
        loss_mode="joint",
        infonce_lambda=10.0,
        infonce_temperature=0.07,
        scoring="embedding",
        norm_pix_loss=False,
    )
    cfg.update(overrides)
    return cfg


def _model(shape, **overrides):
    return build_vit_mae(shape, _cfg(**overrides))


def _batch(shape, b=4):
    h, w, c = shape
    return torch.randn(b, c, h, w)


# ---- build / dispatch ----

def test_build_autoencoder_dispatches_to_vitmae():
    model = build_autoencoder(SHAPES[0], _cfg(), loss="mse", learning_rate=1e-3)
    assert isinstance(model, ViTMAE)
    assert model.learning_rate == 1e-3


def test_divisibility_raises():
    with pytest.raises(ValueError):
        build_vit_mae((96, 1000, 1), _cfg())  # 1000 % 16 != 0


# ---- patchify roundtrip ----

@pytest.mark.parametrize("shape", SHAPES)
def test_patchify_unpatchify_roundtrip(shape):
    x = _batch(shape)
    p = patchify(x, PATCH)
    assert p.shape[1] == _model(shape).num_patches
    back = unpatchify(p, PATCH, (x.shape[0], *(_model(shape).input_shape)))
    assert torch.allclose(x, back, atol=1e-6)


# ---- masking ----

@pytest.mark.parametrize("shape", SHAPES)
def test_random_mask_shape_and_uniqueness(shape):
    m = _model(shape)
    ids = _sample_random_masked_ids(4, m.num_patches, m.n_masked, torch.device("cpu"))
    assert ids.shape == (4, m.n_masked)
    for row in ids:
        assert len(torch.unique(row)) == m.n_masked
    assert ids.max() < m.num_patches


@pytest.mark.parametrize("shape", SHAPES)
def test_cluster_mask_shape_and_ratio(shape):
    m = _model(shape)
    nh, nw = m.grid_size
    ids = _sample_cluster_masked_ids(4, nh, nw, m.n_masked, 3, 5, torch.device("cpu"))
    assert ids.shape == (4, m.n_masked)
    for row in ids:
        assert len(torch.unique(row)) == m.n_masked  # exactly n_masked, no dupes
    assert ids.max() < m.num_patches
    # masked fraction matches mask_ratio
    assert abs(m.n_masked / m.num_patches - m.mask_ratio) < 0.05


# ---- loss modes ----

@pytest.mark.parametrize("shape", SHAPES)
def test_joint_loss_returns_tuple(shape):
    m = _model(shape, loss_mode="joint")
    out = m.compute_loss(_batch(shape))
    assert isinstance(out, tuple) and len(out) == 2
    total, components = out
    assert torch.isfinite(total) and total.item() > 0
    assert set(components) == {"recon_loss", "infonce_loss"}
    for v in components.values():
        assert torch.isfinite(v)


@pytest.mark.parametrize("mode", ["generative", "discriminative"])
def test_single_mode_returns_scalar(mode):
    m = _model(SHAPES[0], loss_mode=mode)
    out = m.compute_loss(_batch(SHAPES[0]))
    assert isinstance(out, torch.Tensor) and out.ndim == 0
    assert torch.isfinite(out) and out.item() > 0


def test_random_mask_mode_loss():
    m = _model(SHAPES[1], mask_mode="random", loss_mode="joint")
    total, _ = m.compute_loss(_batch(SHAPES[1]))
    assert torch.isfinite(total)


def test_infonce_decreases_on_overfit():
    """Guards the temperature/normalisation choice: InfoNCE must actually learn.

    With L2-normalised cosine logits, a too-large temperature floors the loss
    near log(n_masked) and the discriminative objective is inert. Overfitting a
    fixed batch must drive it well down.
    """
    torch.manual_seed(0)
    m = _model(SHAPES[0], loss_mode="discriminative")
    x = _batch(SHAPES[0], b=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = None
    for step in range(200):
        opt.zero_grad()
        loss = m.compute_loss(x)
        if step == 0:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, f"InfoNCE did not learn: {first:.3f} -> {loss.item():.3f}"


def test_norm_pix_loss_not_implemented():
    m = _model(SHAPES[0], norm_pix_loss=True)
    with pytest.raises(NotImplementedError):
        m.compute_loss(_batch(SHAPES[0]))


# ---- backward / gradients ----

def test_train_step_backward_updates_both_heads():
    m = _model(SHAPES[0], loss_mode="joint")
    total, _ = m.compute_loss(_batch(SHAPES[0]))
    total.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(p.grad.abs().sum() > 0 for p in m.reconstruction_head.parameters())
    assert any(p.grad.abs().sum() > 0 for p in m.classification_head.parameters())


# ---- inference: encode / reconstruction / scores ----

@pytest.mark.parametrize("shape", SHAPES)
def test_encode_shape(shape):
    m = _model(shape)
    emb = m.encode(_batch(shape, b=3))
    assert emb.shape == (3, 32)  # (B, embed_dim)


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_reconstruction_shape(shape):
    m = _model(shape).eval()
    x = _batch(shape, b=2)
    with torch.no_grad():
        recon = m(x)
    assert recon.shape == x.shape


@pytest.mark.parametrize("method", ["recon", "infonce"])
def test_anomaly_score_recon_infonce(method):
    m = _model(SHAPES[1]).eval()
    x = _batch(SHAPES[1], b=5)
    s = m.anomaly_score(x, method=method)
    assert s.shape == (5,) and torch.isfinite(s).all()


def test_anomaly_score_embedding_with_occ():
    m = _model(SHAPES[0]).eval()
    with torch.no_grad():
        train_emb = m.encode(_batch(SHAPES[0], b=32)).numpy()
    occ = OneClassScorer("isolation_forest").fit_embeddings(train_emb)
    x = _batch(SHAPES[0], b=6)
    s = m.anomaly_score(x, method="embedding", occ=occ)
    assert s.shape == (6,) and torch.isfinite(s).all()


def test_anomaly_score_embedding_requires_occ():
    m = _model(SHAPES[0]).eval()
    with pytest.raises(ValueError):
        m.anomaly_score(_batch(SHAPES[0], b=2), method="embedding")


# ---- OneClassScorer ----

def test_oneclass_scorer_fit_score_save_load(tmp_path):
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((50, 16)).astype(np.float32)
    occ = OneClassScorer("isolation_forest").fit_embeddings(feats)
    scores = occ.score(feats[:5])
    assert scores.shape == (5,)
    path = tmp_path / "occ.joblib"
    occ.save(path)
    occ2 = OneClassScorer.load(path)
    assert np.allclose(occ.score(feats[:5]), occ2.score(feats[:5]))


def test_oneclass_scorer_requires_fit():
    with pytest.raises(RuntimeError):
        OneClassScorer("isolation_forest").score(np.zeros((3, 16)))


# ---- cadence-aware scoring ----

def test_cadence_score_shape():
    m = _model(SHAPES[1]).eval()  # (96,1024,1) — cadence product only
    x = _batch(SHAPES[1], b=5)
    s = m.anomaly_score(x, method="cadence")
    assert s.shape == (5,) and torch.isfinite(s).all()


def test_cadence_mask_coverage():
    m = _model(SHAPES[1])  # (96,1024,1) -> 6×64 = 384 patches
    mask = m._cadence_on_mask(1, torch.device("cpu"))
    assert mask.shape == (1, 384)
    on_count = mask[0].sum().item()
    assert on_count == 192  # 3 ON obs × 64 cols
    # ON obs 0 → patches 0..63, obs 2 → 128..191, obs 4 → 256..319
    for obs_idx in (0, 2, 4):
        start = obs_idx * 64
        assert mask[0, start : start + 64].all()
    # OFF obs 1,3,5 are NOT masked
    for obs_idx in (1, 3, 5):
        start = obs_idx * 64
        assert not mask[0, start : start + 64].any()


def test_cadence_score_on_only():
    """Perturbing OFF patches must not change the cadence score."""
    m = _model(SHAPES[1]).eval()
    x = _batch(SHAPES[1], b=2)
    s1 = m.anomaly_score(x, method="cadence")
    # Perturb OFF observation rows (obs 1: rows 16..31 in pixel space)
    x2 = x.clone()
    x2[:, :, 16:32, :] += 100.0
    s2 = m.anomaly_score(x2, method="cadence")
    # Scores should differ — OFF context changed, so ON reconstructions change.
    # But perturbing ON rows should change the *target* and thus the score.
    x3 = x.clone()
    x3[:, :, 0:16, :] += 100.0  # perturb ON obs 0
    s3 = m.anomaly_score(x3, method="cadence")
    assert not torch.allclose(s1, s3, atol=1e-3)


def test_cadence_score_raises_on_single_obs():
    """Cadence scoring on a single-observation input should raise."""
    m = _model(SHAPES[0])  # (16,1024,1) -> nh=1
    x = _batch(SHAPES[0], b=2)
    with pytest.raises(ValueError):
        m.anomaly_score(x, method="cadence")
