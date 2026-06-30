"""
GMM scorer on full-background embedding — B5 operational test.

Fits a Gaussian Mixture Model on the FULL training-split embeddings
(noise + RFI, no labels), using BIC to select the number of components K.
Score = -log p(z) under the GMM (negative log-likelihood; higher = more anomalous).

Compares directly against the reference Mahalanobis-on-embedding (B5 AUC=0.927).

The key principle: fit on the FULL background so that RFI is "normal" —
same as encode_separation_test.py block 5 that produced AUC=0.927.  Fitting on
quiet-only (as dist384_scorer.py did) collapses AUC to ~0.5 because RFI and
signals score equally high against a quiet-only baseline.

K=1 with full covariance is approximately equivalent to Ledoit-Wolf Mahalanobis
(GaussianMixture uses EM, not LW shrinkage, so results differ slightly).
K>1 captures multimodal structure (noise cluster, narrowband-RFI cluster,
broadband-RFI cluster, …) that a single Gaussian cannot represent.

Usage (server):
    python scripts/debug/gmm_scorer.py \\
        --checkpoint outputs/training/20260624_084754_057f87c/checkpoints/epoch=006-val_loss=2.1715.ckpt \\
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \\
        --n_samples 1000 --device cuda

Optional PCA pre-reduction (recommended for high-d embeddings, stabilises GMM covariance):
    ... --pca_dim 64

Reference numbers:
    mean_pool Mahalanobis (bg LW)   B5 AUC@SNR10 = 0.927  (encode_separation_test B5)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import build_autoencoder
from scripts.debug.injection_vs_rfi_test import preprocess_raw, inject_narrowband_on_only

INPUT_SHAPE = (96, 1024, 1)
REF_AUC_SNR10 = 0.927


# ---------------------------------------------------------------------------
# Model loading + embedding
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, model_cfg: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_cfg, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    if not hasattr(model, "encode"):
        raise SystemExit("Model has no encode() — needs ViT-MAE backbone.")
    return model


@torch.no_grad()
def embed(model, snippets: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(N, H, W) preprocessed → (N, D) mean-pooled encoder embeddings."""
    out = []
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# GMM scorer
# ---------------------------------------------------------------------------

class GMMScorer:
    """Negative log-likelihood under a full-covariance GMM.

    Fitted on full-background embeddings (noise + RFI, no labels).
    Score = -gmm.score_samples(z); higher = more anomalous.

    If pca_dim is set, applies whitened PCA before fitting; the transform is
    stored and applied at score time.  This stabilises covariance estimation
    when n_samples / embed_dim is low.
    """

    def __init__(self, n_components: int, pca_dim: int | None = None,
                 reg_covar: float = 1e-4, n_init: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.pca_dim = pca_dim
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._gmm: GaussianMixture | None = None

    def fit(self, embeddings: np.ndarray) -> "GMMScorer":
        z = embeddings
        if self.pca_dim is not None and self.pca_dim < z.shape[1]:
            self._scaler = StandardScaler()
            z = self._scaler.fit_transform(z)
            n_comp = min(self.pca_dim, z.shape[0] - 1, z.shape[1])
            self._pca = PCA(n_components=n_comp, whiten=True)
            z = self._pca.fit_transform(z)
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=200,
        ).fit(z)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        z = embeddings
        if self._scaler is not None:
            z = self._scaler.transform(z)
        if self._pca is not None:
            z = self._pca.transform(z)
        return -self._gmm.score_samples(z)

    def bic(self, embeddings: np.ndarray) -> float:
        z = embeddings
        if self._scaler is not None:
            z = self._scaler.transform(z)
        if self._pca is not None:
            z = self._pca.transform(z)
        return float(self._gmm.bic(z))


# ---------------------------------------------------------------------------
# Reference Mahalanobis (identical to encode_separation_test block 5)
# ---------------------------------------------------------------------------

class MahalanobisScorer:
    def __init__(self):
        self._lw: LedoitWolf | None = None

    def fit(self, embeddings: np.ndarray) -> "MahalanobisScorer":
        self._lw = LedoitWolf().fit(embeddings)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(self._lw.mahalanobis(embeddings), 0, None))


# ---------------------------------------------------------------------------
# B5 operational evaluation
# ---------------------------------------------------------------------------

def b5_block(name: str, scorer, emb_fit: np.ndarray,
             emb_ev_rfi: np.ndarray, emb_ev_inj: dict[float, np.ndarray],
             snr_list: list[float]) -> list[tuple]:
    """Fit scorer on emb_fit, evaluate signal-vs-RFI on held-out sets.

    Returns list of (snr, auc, tpr%) rows.
    """
    scorer.fit(emb_fit)
    neg = scorer.score(emb_ev_rfi)
    thr = np.percentile(neg, 90)  # 10% RFI-FP operating point
    print(f"\n  [{name}]  RFI score range [{neg.min():.3f}, {neg.max():.3f}]"
          f"  threshold@10%FP={thr:.3f}")
    print(f"  {'SNR':>5}  {'AUC':>8}  {'TPR@10%FP':>10}")
    rows = []
    floor = None
    for snr in snr_list:
        pos = scorer.score(emb_ev_inj[snr])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        auc = float(roc_auc_score(y, np.concatenate([pos, neg])))
        tpr = float((pos > thr).mean() * 100)
        if floor is None and tpr >= 50:
            floor = snr
        rows.append((snr, auc, tpr))
        print(f"  {snr:5.0f}  {auc:8.3f}  {tpr:9.1f}%")
    tag = f"SNR≈{floor:.0f}" if floor else "never ≥50%"
    print(f"  -> floor: {tag}")
    return rows


# ---------------------------------------------------------------------------
# BIC selection
# ---------------------------------------------------------------------------

def select_k_bic(emb_fit: np.ndarray, k_list: list[int],
                 pca_dim: int | None, reg_covar: float) -> int:
    print(f"\n  BIC sweep (pca_dim={pca_dim}):")
    best_k, best_bic = k_list[0], float("inf")
    for k in k_list:
        gmm = GMMScorer(n_components=k, pca_dim=pca_dim, reg_covar=reg_covar).fit(emb_fit)
        bic_val = gmm.bic(emb_fit)
        marker = ""
        if bic_val < best_bic:
            best_bic = bic_val
            best_k = k
            marker = " ←"
        print(f"    K={k:3d}  BIC={bic_val:.1f}{marker}")
    print(f"  -> BIC selects K={best_k}")
    return best_k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[3, 5, 7, 10, 12, 15, 20])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 5, 10, 20],
                   help="GMM component counts to sweep via BIC")
    p.add_argument("--pca_dim", type=int, default=None,
                   help="PCA whitening before GMM (None = use raw embeddings)")
    p.add_argument("--reg_covar", type=float, default=1e-4,
                   help="Covariance regularisation for GaussianMixture")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/gmm_scorer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    # ── data ──
    npy_path = args.cache / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    idx = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[idx])
    del arr
    print(f"  Raw snippets: {raw_snippets.shape}")

    preprocessed, hot_fracs = [], []
    for s in raw_snippets:
        pp = preprocess_raw(s, preproc)
        preprocessed.append(pp)
        hot_fracs.append(float((pp > 5.0).sum()) / pp.size)
    preprocessed = np.array(preprocessed)
    hot_fracs = np.array(hot_fracs)

    quiet_idx = np.where(hot_fracs <= np.percentile(hot_fracs, 25))[0]
    rfi_idx = np.where(hot_fracs >= np.percentile(hot_fracs, 75))[0]
    print(f"  Quiet: {len(quiet_idx)}  RFI: {len(rfi_idx)}")

    # ── embed ALL snippets ──
    print("Encoding all snippets...")
    emb_all = embed(model, preprocessed, args.device)
    D = emb_all.shape[1]
    print(f"  Embedding shape: {emb_all.shape}  (embed_dim={D})")

    # ── 70/30 split (mirrors encode_separation_test block 5) ──
    perm = rng.permutation(len(preprocessed))
    fit_mask = np.zeros(len(preprocessed), bool)
    fit_mask[perm[: int(0.7 * len(perm))]] = True

    emb_fit = emb_all[fit_mask]                               # fit set (full background)
    ev_rfi = [i for i in rfi_idx if not fit_mask[i]]          # held-out RFI (negatives)
    ev_quiet = [i for i in quiet_idx if not fit_mask[i]]      # held-out quiet (for injection)

    print(f"  B5 split: fit={fit_mask.sum()}  eval-RFI={len(ev_rfi)}  eval-quiet={len(ev_quiet)}")
    if len(ev_rfi) < 10 or len(ev_quiet) < 10:
        raise SystemExit("Held-out sets too small — raise --n_samples")

    emb_ev_rfi = emb_all[ev_rfi]

    # ── inject signals into held-out quiet snippets ──
    print(f"Injecting {len(args.snr_list)} SNR levels into {len(ev_quiet)} held-out quiet frames...")
    emb_ev_inj: dict[float, np.ndarray] = {}
    for snr in args.snr_list:
        inj = np.array([
            preprocess_raw(
                inject_narrowband_on_only(raw_snippets[i], snr=snr,
                                          drift_rate=args.drift_rate,
                                          seed=args.seed + j), preproc)
            for j, i in enumerate(ev_quiet)
        ])
        emb_ev_inj[snr] = embed(model, inj, args.device)
    print("  Done.")

    # ── BIC sweep to select K ──
    print(f"\n{'='*65}")
    print("BIC SWEEP — selecting best K for GMM")
    print(f"{'='*65}")
    best_k = select_k_bic(emb_fit, args.k_list, args.pca_dim, args.reg_covar)

    # ── B5 evaluation ──
    print(f"\n{'='*65}")
    print("B5. OPERATIONAL — signal vs RFI (full-background fit, 70/30 split)")
    print(f"{'='*65}")
    print(f"  held-out: RFI(neg)={len(ev_rfi)}  quiet→inject(pos)={len(ev_quiet)}")

    all_rows: dict[str, list] = {}

    # Reference: Mahalanobis (LW) — same as encode_separation_test block 5
    ref_rows = b5_block("maha_lw_ref", MahalanobisScorer(),
                        emb_fit, emb_ev_rfi, emb_ev_inj, args.snr_list)
    all_rows["maha_lw"] = ref_rows

    # GMM with BIC-selected K
    gmm_bic_rows = b5_block(f"gmm_K{best_k}_bic", GMMScorer(best_k, args.pca_dim, args.reg_covar),
                             emb_fit, emb_ev_rfi, emb_ev_inj, args.snr_list)
    all_rows[f"gmm_K{best_k}"] = gmm_bic_rows

    # GMM sweep over all K values (diagnostic — which K actually performs best?)
    for k in args.k_list:
        if k == best_k:
            continue
        rows = b5_block(f"gmm_K{k}", GMMScorer(k, args.pca_dim, args.reg_covar),
                        emb_fit, emb_ev_rfi, emb_ev_inj, args.snr_list)
        all_rows[f"gmm_K{k}"] = rows

    # ── Summary ──
    print(f"\n{'='*65}")
    print("SUMMARY — B5 AUC@SNR10  (ref = 0.927)")
    print(f"{'='*65}")
    snr10 = 10.0
    for name, rows in all_rows.items():
        row = next((r for r in rows if r[0] == snr10), None)
        if row:
            delta = row[1] - REF_AUC_SNR10
            flag = " [BEATS ref!]" if delta > 0.005 else f" ({delta:+.3f} vs ref)"
            print(f"  {name:<20}: AUC@SNR10={row[1]:.3f}  TPR@10%FP={row[2]:.1f}%{flag}")

    # ── Save ──
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {name: np.array(rows) for name, rows in all_rows.items()}
    out["best_k"] = np.array([best_k])
    out["k_list"] = np.array(args.k_list)
    np.savez(str(args.out_dir / "gmm_b5_results.npz"), **out)
    print(f"\nSaved → {args.out_dir}/gmm_b5_results.npz")


if __name__ == "__main__":
    main()
