"""Encoder-feature diagnostic: did the ViT-MAE *encoder* collapse, or just the recon head?

Decides the fork left open by the AE-vs-ViT-MAE recon sweep (recon-MSE scoring is a
dead end: AE copies -> 0% det, ViT-MAE collapses to the noise mean -> energy detector).
The open question: does ``encode(x)`` still carry usable structure even though the
*reconstruction* is ~zero?

Two things are measured on the ViT-MAE's mean-pooled encoder embedding ``encode(x)``:

  1. COLLAPSE CHECK. Per-dimension std of the embedding across quiet snippets. If the
     embedding barely varies between inputs, the *encoder* collapsed (representation
     collapse) -> recon-head fixes (lower mask_ratio, variance-weight, decoder) cannot
     help; the objective itself (InfoNCE/encoder) must change. If it varies, the info
     is there and the right pivot is feature/one-class scoring.

  2. ONE-CLASS SEPARATION vs SNR. Score = Euclidean distance from the quiet centroid in
     per-dim-whitened embedding space (the simplest one-class detector). Produces the
     SAME ``det@3σ / det@5σ vs SNR`` table as ``cadence_snr_sweep.py`` recon, so the two
     scorers are directly comparable. The payoff question: does feature-distance detect
     at *lower* SNR than recon's SNR>=20 energy-detector floor? Also reports the quiet-vs-RFI
     separation (Cohen's d) — high d there would mean RFI also lights up (FP risk).

Mirrors the sweep's harness exactly (same cache, q25 hot-frac quiet split, ON-only
injection into the quiet snippets) so the comparison is apples-to-apples.

Usage (server, not dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/encode_separation_test.py \
        --checkpoint outputs/training/20260624_084754_057f87c/checkpoints/epoch=006-val_loss=2.1715.ckpt \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \
        --out_dir outputs/sweeps/encode_separation
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import build_autoencoder
from scripts.debug.injection_vs_rfi_test import preprocess_raw, inject_narrowband_on_only

INPUT_SHAPE = (96, 1024, 1)


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    if not hasattr(model, "encode"):
        raise SystemExit(f"Model {type(model).__name__} has no encode() — this test needs "
                         f"the ViT-MAE backbone (architecture: vit_mae).")
    return model


@torch.no_grad()
def embed(model, snippets: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(N, H, W) preprocessed -> (N, D) encoder embeddings."""
    out = []
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size between two 1-D score arrays (pooled std)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Per-snippet input energy = mean(x^2) over the preprocessed frame. (N,)"""
    return (frames.astype(np.float64) ** 2).mean(axis=(1, 2))


def morphology_beyond_energy(emb_inj, en_inj, emb_rfi, en_rfi, seed=0):
    """Upper-bound test: can a LINEAR readout tell injected-narrowband (anomaly)
    from RFI (normal-but-energetic) using the embedding *beyond* input energy?

    Both classes are energy-confounded (a signal IS energy), so we compare three
    5-fold-CV AUCs of a logistic readout, on balanced subsamples:
      - energy-only   : AUC reachable from the scalar energy alone (the confound).
      - embedding     : AUC from the raw embedding (energy + morphology mixed).
      - emb | energy  : AUC from the embedding with energy regressed out per-dim
                        (OLS residuals) — morphology sensitivity with energy removed.

    If 'emb | energy' >> 0.5 there is a morphology axis -> feature scoring is
    viable. If it collapses to ~0.5 (and only energy-only/embedding separate), the
    embedding encodes ~only energy -> recon and feature scoring are the same thing.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  (scikit-learn not available — skipping morphology-vs-energy test)")
        return None

    rng = np.random.default_rng(seed)
    n = min(len(emb_inj), len(emb_rfi))
    ia = rng.choice(len(emb_inj), n, replace=False)
    ib = rng.choice(len(emb_rfi), n, replace=False)
    X = np.concatenate([emb_inj[ia], emb_rfi[ib]], 0)
    e = np.concatenate([en_inj[ia], en_rfi[ib]], 0)[:, None]
    y = np.concatenate([np.ones(n), np.zeros(n)])

    def auc(feats):
        clf = LogisticRegression(max_iter=2000)
        return float(cross_val_score(clf, StandardScaler().fit_transform(feats),
                                     y, cv=5, scoring="roc_auc").mean())

    A = np.concatenate([e, np.ones_like(e)], 1)          # (N, 2): [energy, 1]
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)         # per-dim OLS on energy
    X_resid = X - A @ coef                               # embedding with energy removed
    return {"n_per_class": n, "energy_only": auc(e),
            "embedding": auc(X), "emb_given_energy": auc(X_resid)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/encode_separation")
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

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    idx = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[idx])
    del arr
    print(f"  Raw snippets: {raw_snippets.shape}")

    # Same quiet/RFI split as the sweep: hot-fraction quartiles.
    preprocessed, hot_fracs = [], []
    for i in range(len(raw_snippets)):
        snip = preprocess_raw(raw_snippets[i], preproc)
        preprocessed.append(snip)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)
    preprocessed = np.array(preprocessed)
    hot_fracs = np.array(hot_fracs)
    quiet_idx = np.where(hot_fracs <= np.percentile(hot_fracs, 25))[0]
    rfi_idx = np.where(hot_fracs >= np.percentile(hot_fracs, 75))[0]
    print(f"  Quiet: {len(quiet_idx)}  RFI: {len(rfi_idx)}")

    emb_quiet = embed(model, preprocessed[quiet_idx], args.device)
    emb_rfi = embed(model, preprocessed[rfi_idx], args.device)
    en_rfi = frame_energy(preprocessed[rfi_idx])
    D = emb_quiet.shape[1]

    # ---- 1. COLLAPSE CHECK ----
    mu = emb_quiet.mean(axis=0)
    sd = emb_quiet.std(axis=0)
    emb_norm = float(np.linalg.norm(mu))
    rel_std = float(sd.mean()) / (emb_norm / np.sqrt(D) + 1e-12)
    dead = int((sd < 1e-4).sum())
    print(f"\n{'='*64}\n1. ENCODER COLLAPSE CHECK (embed_dim={D})\n{'='*64}")
    print(f"  mean per-dim std across quiet : {sd.mean():.5f}  (min {sd.min():.5f}, max {sd.max():.5f})")
    print(f"  ||mean embedding||            : {emb_norm:.4f}")
    print(f"  rel. variation (std / per-dim mag): {rel_std:.4f}")
    print(f"  near-dead dims (std<1e-4)     : {dead}/{D}")
    if rel_std < 0.05 or dead > 0.9 * D:
        print("  VERDICT: encoder appears COLLAPSED — features barely move with input.")
        print("           -> recon-head fixes won't help; change the objective.")
    else:
        print("  VERDICT: encoder features VARY with input — info is present.")
        print("           -> feature/one-class scoring is the right pivot (see table below).")

    # ---- 2. ONE-CLASS SEPARATION vs SNR ----
    # Whitened distance from the quiet centroid. Calibrate on quiet (matches the
    # sweep: baseline = clean quiet, injected = same quiet snippets + signal).
    sd_safe = np.where(sd < 1e-6, 1.0, sd)

    def score(emb: np.ndarray) -> np.ndarray:
        return np.linalg.norm((emb - mu) / sd_safe, axis=1)

    base = score(emb_quiet)
    rfi = score(emb_rfi)
    b_mean, b_std = base.mean(), base.std()
    d_rfi = cohens_d(rfi, base)
    print(f"\n{'='*64}\n2. ONE-CLASS FEATURE SCORE (whitened dist. from quiet centroid)\n{'='*64}")
    print(f"  Quiet baseline : {b_mean:.4f} ± {b_std:.4f}")
    print(f"  RFI            : {rfi.mean():.4f} ± {rfi.std():.4f}   (Cohen's d vs quiet = {d_rfi:+.2f})")
    print(f"\n  {'SNR':>5s}  {'mean':>8s}  {'sigma':>8s}  {'cohen_d':>8s}  {'det@3σ':>8s}  {'det@5σ':>8s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    t3, t5 = b_mean + 3 * b_std, b_mean + 5 * b_std
    inj_scores, inj_emb, inj_en, results = {}, {}, {}, []
    for snr in args.snr_list:
        inj = np.array([preprocess_raw(
            inject_narrowband_on_only(raw_snippets[i], snr=snr, drift_rate=args.drift_rate,
                                      seed=args.seed + j), preproc)
            for j, i in enumerate(quiet_idx)])
        e_inj = embed(model, inj, args.device)
        inj_emb[snr] = e_inj
        inj_en[snr] = frame_energy(inj)
        s = score(e_inj)
        inj_scores[snr] = s
        sigma = (s.mean() - b_mean) / b_std if b_std > 0 else 0.0
        det3, det5 = (s > t3).mean() * 100, (s > t5).mean() * 100
        results.append((snr, s.mean(), sigma, cohens_d(s, base), det3, det5))
        print(f"  {snr:5.0f}  {s.mean():8.4f}  {sigma:8.2f}σ  {cohens_d(s, base):8.2f}  "
              f"{det3:7.1f}%  {det5:7.1f}%")

    # ---- 3. MORPHOLOGY BEYOND ENERGY (the deciding test) ----
    # Pool injected snippets whose energy overlaps the RFI range (SNR>=15), so the
    # separation can't be a trivial low-vs-high energy split.
    pool_snr = [s for s in args.snr_list if s >= 15]
    emb_pool = np.concatenate([inj_emb[s] for s in pool_snr], 0)
    en_pool = np.concatenate([inj_en[s] for s in pool_snr], 0)
    print(f"\n{'='*64}\n3. MORPHOLOGY BEYOND ENERGY  (injected SNR>={int(min(pool_snr))} vs RFI)\n{'='*64}")
    print(f"  energy overlap: injected {en_pool.min():.2f}–{en_pool.max():.2f} | "
          f"RFI {en_rfi.min():.2f}–{en_rfi.max():.2f}")
    m = morphology_beyond_energy(emb_pool, en_pool, emb_rfi, en_rfi, seed=args.seed)
    if m is not None:
        print(f"  balanced n/class : {m['n_per_class']}")
        print(f"  AUC energy-only        : {m['energy_only']:.3f}")
        print(f"  AUC embedding (raw)    : {m['embedding']:.3f}")
        print(f"  AUC embedding | energy : {m['emb_given_energy']:.3f}   "
              f"<-- morphology axis with energy regressed out")
        if m["emb_given_energy"] > 0.65:
            print("  VERDICT: a MORPHOLOGY axis survives energy removal -> feature scoring "
                  "is viable; build a better one-class than naive distance.")
        else:
            print("  VERDICT: embedding separates injected/RFI ~only via energy -> recon and "
                  "feature scoring are the same energy detector; faint-narrowband AD is a "
                  "dead end here (decision needed: change objective, or change target/product).")

    # ---- save + plot ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "encode_separation_results.npz",
             snr_list=np.array(args.snr_list), embed_quiet=emb_quiet, embed_rfi=emb_rfi,
             en_rfi=en_rfi, quiet_score=base, rfi_score=rfi, per_dim_std=sd,
             **{f"inject_score_snr_{int(s)}": v for s, v in inj_scores.items()},
             **{f"inject_emb_snr_{int(s)}": v for s, v in inj_emb.items()},
             **{f"inject_en_snr_{int(s)}": v for s, v in inj_en.items()})

    snrs = sorted(args.snr_list)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].errorbar(snrs, [inj_scores[s].mean() for s in snrs],
                   yerr=[inj_scores[s].std() for s in snrs], marker="o", capsize=3)
    ax[0].axhline(b_mean, ls="--", color="gray", label=f"quiet ({b_mean:.2f})")
    ax[0].axhline(t3, ls=":", color="red", label="3σ")
    ax[0].set(xlabel="Injection SNR", ylabel="feature distance", title="Feature score vs SNR")
    ax[0].legend(fontsize=8)
    for ns, ls in [(3, "-"), (5, "--")]:
        thr = b_mean + ns * b_std
        ax[1].plot(snrs, [(inj_scores[s] > thr).mean() * 100 for s in snrs], ls,
                   marker="o", label=f"@ {ns}σ")
    ax[1].set(xlabel="Injection SNR", ylabel="Detection rate (%)",
              title="Feature-distance detection vs SNR", ylim=(-5, 105))
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_dir / "feature_detection_vs_snr.png", dpi=150)
    print(f"\nSaved → {args.out_dir / 'feature_detection_vs_snr.png'}")
    print(f"Saved → {args.out_dir / 'encode_separation_results.npz'}")
    print("\nCompare det@3σ above with the recon sweep: if feature-distance detects at "
          "lower SNR than recon's ~SNR20 floor, pivot to one-class feature scoring.")


if __name__ == "__main__":
    main()
