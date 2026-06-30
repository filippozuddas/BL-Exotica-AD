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


def load_model(checkpoint_path: Path, model_config: dict, device: str, require_encode: bool = True):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    if require_encode and not hasattr(model, "encode"):
        raise SystemExit(f"Model {type(model).__name__} has no encode() — the embedding "
                         f"blocks need an encoder embedding (use --scoring recon for "
                         f"reconstruction-error models such as the plain AE / MemAE).")
    return model


@torch.no_grad()
def embed(model, snippets: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(N, H, W) preprocessed -> (N, D) encoder embeddings."""
    out = []
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def recon_score(model, snippets: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(N, H, W) preprocessed -> (N,) per-sample reconstruction MSE.

    The reconstruction-error analogue of ``embed`` — works for any backbone that
    exposes ``anomaly_score(x, method='recon')`` (plain AE, MAE, ViT-MAE, MemAE).
    """
    out = []
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model.anomaly_score(x, method="recon").cpu().numpy())
    return np.concatenate(out, axis=0)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size between two 1-D score arrays (pooled std)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Per-snippet input energy = mean(x^2) over the preprocessed frame. (N,)"""
    return (frames.astype(np.float64) ** 2).mean(axis=(1, 2))


def frame_stats(frames: np.ndarray) -> np.ndarray:
    """Hand-crafted shape scalars per snippet, (N, 3): peakiness (max/std),
    excess kurtosis, and fraction of energy in the brightest 0.1% pixels.

    These are the trivial 'is it a thin bright line vs. diffuse RFI' statistics a
    2-line numpy detector would use. The control: if the embedding's matched-energy
    AUC is no better than an AUC from THESE, the encoder adds nothing over a scalar
    threshold and the feature-scoring pivot is pointless.
    """
    f = frames.reshape(len(frames), -1).astype(np.float64)
    mu = f.mean(1, keepdims=True)
    sd = f.std(1) + 1e-9
    peak = f.max(1) / sd
    kurt = (((f - mu) ** 4).mean(1)) / (sd ** 4) - 3.0
    p2 = np.sort(f ** 2, axis=1)
    k = max(1, int(0.001 * f.shape[1]))
    topfrac = p2[:, -k:].sum(1) / (p2.sum(1) + 1e-9)
    return np.stack([peak, kurt, topfrac], axis=1)


def _caliper_match(en_small, en_big, caliper, rng):
    """Greedy 1:1 nearest-energy matching, without replacement. Iterates over the
    smaller class; for each, takes the closest unused sample of the big class if
    within ``caliper`` energy units. Returns (small_idx, big_idx) paired arrays."""
    order = rng.permutation(len(en_small))
    used = np.zeros(len(en_big), bool)
    ps, pb = [], []
    for i in order:
        d = np.abs(en_big - en_small[i])
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= caliper:
            used[j] = True
            ps.append(int(i))
            pb.append(j)
    return np.array(ps, int), np.array(pb, int)


def morphology_matched_energy(emb_inj, en_inj, st_inj, emb_rfi, en_rfi, st_rfi, seed=0):
    """Deciding test: at MATCHED input energy, does the embedding separate
    injected-narrowband (anomaly) from RFI (normal-but-energetic) — and does it do
    so beyond what a TRIVIAL hand-crafted shape statistic already achieves?

    Energy is the confound (a faint narrowband barely raises energy while RFI is
    genuinely energetic), so we pair each RFI to an injected snippet of nearly
    identical mean(x^2) (1:1 nearest-neighbour, adaptive caliper tightened until the
    energy-only AUC drops to ~0.5). On the matched pairs we then compare three
    5-fold-CV logistic AUCs:
      - energy_only : sanity — ~0.5 means matching worked.
      - trivial     : from frame_stats (peakiness/kurtosis/top-pixel fraction). The
                      embedding must BEAT this — else it only re-encodes a higher-order
                      energy statistic a 2-line detector captures (no AE needed).
      - embedding   : if >> trivial and >> 0.5, the encoder carries genuine
                      morphology a scalar can't -> feature scoring is justified.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  (scikit-learn not available — skipping matched-energy test)")
        return None

    def auc(feats, y):
        return float(cross_val_score(LogisticRegression(max_iter=2000),
                                     StandardScaler().fit_transform(feats),
                                     y, cv=5, scoring="roc_auc").mean())

    rng = np.random.default_rng(seed)
    best = None
    for caliper in [0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
        ps, pb = _caliper_match(en_rfi, en_inj, caliper, rng)
        if len(ps) < 20:
            continue
        e = np.concatenate([en_inj[pb], en_rfi[ps]], 0)[:, None]
        y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
        ea = auc(e, y)
        cand = {"caliper": caliper, "n_per_class": len(ps), "energy_only": ea, "ps": ps, "pb": pb}
        if ea <= 0.58:  # loosest caliper that already neutralises energy (keeps n high)
            best = cand
            break
        if best is None or ea < best["energy_only"]:
            best = cand
    if best is None:
        return {"error": "too few energy-matched pairs (RFI scarce in the injected band)"}

    ps, pb = best["ps"], best["pb"]
    y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
    X_emb = np.concatenate([emb_inj[pb], emb_rfi[ps]], 0)
    X_triv = np.concatenate([st_inj[pb], st_rfi[ps]], 0)
    return {"n_per_class": int(best["n_per_class"]), "caliper": best["caliper"],
            "energy_only": best["energy_only"], "trivial": auc(X_triv, y),
            "embedding": auc(X_emb, y)}


def morphology_matched_energy_recon(rec_inj, en_inj, st_inj, rec_rfi, en_rfi, st_rfi, seed=0):
    """Recon analogue of ``morphology_matched_energy``: at MATCHED input energy, does
    the model's RECONSTRUCTION ERROR separate injected-narrowband from RFI beyond a
    trivial shape statistic — i.e. is the recon 'win' morphology or just energy?

    ``rec_*`` are per-sample reconstruction MSE scalars. We caliper-match RFI to
    injected snippets of near-identical mean(x^2), then on the matched pairs report:
      - energy_only : roc_auc of mean(x^2) — sanity, ~0.5 means matching worked.
      - trivial     : 5-fold-CV logistic AUC of frame_stats (peakiness/kurtosis/top).
      - recon       : roc_auc of the recon-MSE scalar.
    recon ~ energy_only (~0.5) => the recon-B5 'win' is pure energy/contrast. recon
    >> trivial and >> 0.5 => recon error carries genuine morphology.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  (scikit-learn not available — skipping matched-energy recon test)")
        return None

    def cv_auc(feats, y):
        return float(cross_val_score(LogisticRegression(max_iter=2000),
                                     StandardScaler().fit_transform(feats),
                                     y, cv=5, scoring="roc_auc").mean())

    rng = np.random.default_rng(seed)
    best = None
    for caliper in [0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
        ps, pb = _caliper_match(en_rfi, en_inj, caliper, rng)
        if len(ps) < 20:
            continue
        e = np.concatenate([en_inj[pb], en_rfi[ps]], 0)
        y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
        ea = float(roc_auc_score(y, e))
        ea = max(ea, 1 - ea)  # energy AUC is orientation-free here (sanity only)
        cand = {"caliper": caliper, "n_per_class": len(ps), "energy_only": ea, "ps": ps, "pb": pb}
        if ea <= 0.58:
            best = cand
            break
        if best is None or ea < best["energy_only"]:
            best = cand
    if best is None:
        return {"error": "too few energy-matched pairs (injected/RFI energy ranges barely overlap)"}

    ps, pb = best["ps"], best["pb"]
    y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
    rec = np.concatenate([rec_inj[pb], rec_rfi[ps]], 0)
    X_triv = np.concatenate([st_inj[pb], st_rfi[ps]], 0)
    return {"n_per_class": int(best["n_per_class"]), "caliper": best["caliper"],
            "energy_only": best["energy_only"], "trivial": cv_auc(X_triv, y),
            "recon": float(roc_auc_score(y, rec))}


def unsupervised_occ(emb_quiet, emb_rfi, inj_emb, snr_list, seed=0):
    """Fully-unsupervised one-class detectors on the frozen embedding (NO labels).

    Fits three OCCs on the normal (quiet) embeddings only, then scores the held-out
    quiet baseline, RFI, and each injected SNR. Threshold = quiet mean+3σ/5σ of each
    OCC's own score (same calibration convention as the recon sweep), so the det@SNR
    floor is directly comparable to recon's SNR>=20.

      - maha : Mahalanobis distance, Ledoit-Wolf-shrunk covariance (proper full-cov
               Gaussian one-class; the naive block-2 distance was diagonal-only).
      - iforest / ocsvm : sklearn IsolationForest / OneClassSVM on standardised emb.

    The question the supervised AUC could NOT answer: is the injected narrowband an
    OUTLIER of the normal density (flaggable without labels), or does it sit inside it?
    """
    try:
        from sklearn.covariance import LedoitWolf
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  (scikit-learn not available — skipping unsupervised OCC test)")
        return None

    scaler = StandardScaler().fit(emb_quiet)
    qz = scaler.transform(emb_quiet)
    lw = LedoitWolf().fit(emb_quiet)
    iso = IsolationForest(n_estimators=300, random_state=seed).fit(qz)
    svm = OneClassSVM(nu=0.1, gamma="scale").fit(qz)

    scorers = {
        "maha": lambda E: np.sqrt(np.clip(lw.mahalanobis(E), 0, None)),
        "iforest": lambda E: -iso.score_samples(scaler.transform(E)),
        "ocsvm": lambda E: -svm.decision_function(scaler.transform(E)),
    }
    out = {}
    for name, fn in scorers.items():
        base = fn(emb_quiet)
        bm, bs = base.mean(), base.std() + 1e-12
        out[name] = {
            "b_mean": bm, "b_std": bs,
            "rfi_d": cohens_d(fn(emb_rfi), base),
            "det3": {s: float((fn(inj_emb[s]) > bm + 3 * bs).mean() * 100) for s in snr_list},
            "det5": {s: float((fn(inj_emb[s]) > bm + 5 * bs).mean() * 100) for s in snr_list},
        }
    return out


def operational_signal_vs_rfi(model, preprocessed, raw_snippets, quiet_idx, rfi_idx,
                              preproc, snr_list, drift_rate, device, scoring="embedding", seed=0):
    """THE operational test: in a real search (no labels), can the scorer rank an
    injected signal ABOVE real RFI? Energy works AGAINST us here (injected-into-quiet
    is LOW energy, RFI is HIGH) — so a high AUC means the scorer uses morphology, not
    energy. Reports, per SNR, AUC(signal vs RFI) and TPR at 10% RFI-false-positive
    rate. AUC~0.5 (or <0.5) => signal looks as normal as / more normal than RFI => no win.

    Two scorings share the SAME held-out partition (so they are directly comparable):
    - ``embedding`` (default): Mahalanobis (Ledoit-Wolf) fit on the fit-half embeddings
      (RFI folded into 'normal'); score = Mahalanobis distance of ``encode(x)``.
    - ``recon``: per-sample reconstruction MSE ``anomaly_score(x, 'recon')`` — no fit
      needed. This is the score MemAE / the plain AE / ViT-MAE are actually deployed
      with, and the metric that exposes the noise-floor failure mode.

    Held-out: RFI = negatives, injected-into-held-out-quiet = positives.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  (scikit-learn not available — skipping operational test)")
        return None

    if scoring == "recon":
        # Recon scoring fits NOTHING, so no held-out partition is needed: use ALL
        # quiet (inject) and ALL RFI (negatives) with zero leakage. This also
        # avoids the ~75/class held-out regime that produced the retracted 0.927
        # artifact and clears the >=500/class sampling-guardrail floor.
        ev_rfi = list(rfi_idx)
        ev_quiet = list(quiet_idx)
        if len(ev_rfi) < 10 or len(ev_quiet) < 10:
            return {"error": f"too few quiet/RFI (RFI {len(ev_rfi)}, quiet {len(ev_quiet)}); "
                             f"raise --n_samples"}

        def score(frames):
            return recon_score(model, frames, device)
        neg = score(preprocessed[ev_rfi])
        en_neg = frame_energy(preprocessed[ev_rfi])  # energy-only control baseline
    else:
        # Embedding scoring fits a Ledoit-Wolf covariance, so it DOES need a held-out
        # split to avoid leaking the eval samples into the fit.
        from sklearn.covariance import LedoitWolf
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(len(preprocessed))
        fit = np.zeros(len(preprocessed), bool)
        fit[perm[: int(0.7 * len(perm))]] = True

        ev_rfi = [i for i in rfi_idx if not fit[i]]
        ev_quiet = [i for i in quiet_idx if not fit[i]]
        if len(ev_rfi) < 10 or len(ev_quiet) < 10:
            return {"error": f"held-out too small (RFI {len(ev_rfi)}, quiet {len(ev_quiet)}); "
                             f"raise --n_samples"}

        emb_all = embed(model, preprocessed, device)
        lw = LedoitWolf().fit(emb_all[fit])

        def maha(E):
            return np.sqrt(np.clip(lw.mahalanobis(E), 0, None))

        def score(frames):
            return maha(embed(model, frames, device))
        neg = maha(emb_all[ev_rfi])
        en_neg = frame_energy(preprocessed[ev_rfi])  # energy-only control baseline

    thr = np.percentile(neg, 90)  # 10% RFI false-positive operating point
    rows = []
    for snr in snr_list:
        inj = np.array([preprocess_raw(
            inject_narrowband_on_only(raw_snippets[i], snr=snr, drift_rate=drift_rate,
                                      seed=seed + j), preproc)
            for j, i in enumerate(ev_quiet)])
        pos = score(inj)
        en_pos = frame_energy(inj)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        auc = float(roc_auc_score(y, np.concatenate([pos, neg])))
        # Energy-only control on the SAME partition: if the model's AUC is no better
        # than this, the "win" is just energy/contrast, not morphology.
        auc_en = float(roc_auc_score(y, np.concatenate([en_pos, en_neg])))
        rows.append((snr, auc, float((pos > thr).mean() * 100), auc_en))
    return {"n_rfi": len(ev_rfi), "n_quiet": len(ev_quiet), "rows": rows}


def print_operational(op, scoring="embedding"):
    """Print the operational signal-vs-RFI table + verdict (shared by both scorings)."""
    if op is None:
        return
    if "error" in op:
        print(f"  SKIPPED: {op['error']}")
        return
    print(f"  held-out: RFI(neg)={op['n_rfi']}  quiet→inject(pos)={op['n_quiet']}")
    print(f"\n  {'SNR':>5s}  {'AUC(model)':>11s}  {'AUC(energy)':>12s}  {'Δ vs energy':>12s}  {'TPR@10%FP':>10s}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*12}  {'-'*12}  {'-'*10}")
    floor = None
    beats_energy = []
    for snr, auc, tpr, auc_en in op["rows"]:
        if floor is None and tpr >= 50:
            floor = snr
        beats_energy.append(auc - auc_en)
        print(f"  {snr:5.0f}  {auc:11.3f}  {auc_en:12.3f}  {auc - auc_en:+12.3f}  {tpr:9.1f}%")
    # The honest verdict is NOT "floor <= 15" (that just measures detection); it is
    # whether the model's recon AUC clears the energy-only control. A model whose AUC
    # only tracks (or trails) energy is a contrast detector, not a morphology detector.
    max_delta = max(beats_energy)
    print(f"\n  VERDICT: best margin over the energy-only control = {max_delta:+.3f}.")
    if max_delta <= 0.03:
        print("    -> the model does NOT beat a pure energy/contrast baseline on this "
              "(un-matched) test -> the 'win' is energy, NOT morphology. Read the "
              "matched-energy recon block below; the operational floor here is "
              "energy-confounded and must not be trusted as a morphology result.")
    else:
        print("    -> the model's recon AUC exceeds the energy-only control -> there is signal "
              "beyond contrast here; confirm it on the matched-energy recon block below "
              "before trusting it as morphology.")


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
    p.add_argument("--scoring", choices=["embedding", "recon"], default="embedding",
                   help="embedding (default): run all blocks on encode(x) embeddings. "
                        "recon: run ONLY the operational signal-vs-RFI block on "
                        "reconstruction MSE (for AE / MAE / ViT-MAE / MemAE).")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device,
                       require_encode=(args.scoring == "embedding"))

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

    # ---- RECON scoring: operational signal-vs-RFI + the matched-energy control ----
    # The embedding blocks (1-4) need encode(x); recon scoring is for the AE/MAE/
    # ViT-MAE/MemAE deployment metric (reconstruction MSE). Two blocks:
    #   (R1) operational signal-vs-RFI, WITH an energy-only column (is the win energy?)
    #   (R2) matched-energy recon — the discriminating morphology test.
    if args.scoring == "recon":
        print(f"\n{'='*64}\nR1. OPERATIONAL (RECON) — signal vs RFI, with energy-only control\n{'='*64}")
        op = operational_signal_vs_rfi(model, preprocessed, raw_snippets, quiet_idx, rfi_idx,
                                       preproc, args.snr_list, args.drift_rate, args.device,
                                       scoring="recon", seed=args.seed)
        print_operational(op, scoring="recon")

        # R2: matched-energy recon. Pool injected snippets across all SNRs so their
        # energies span/overlap the RFI range, then caliper-match and compare recon
        # AUC vs trivial-stats vs energy-only at matched energy.
        print(f"\n{'='*64}\nR2. MATCHED-ENERGY RECON — is the recon win morphology or energy?\n{'='*64}")
        rec_rfi = recon_score(model, preprocessed[rfi_idx], args.device)
        en_rfi = frame_energy(preprocessed[rfi_idx])
        st_rfi = frame_stats(preprocessed[rfi_idx])
        rec_pool, en_pool, st_pool = [], [], []
        for snr in args.snr_list:
            inj = np.array([preprocess_raw(
                inject_narrowband_on_only(raw_snippets[i], snr=snr, drift_rate=args.drift_rate,
                                          seed=args.seed + j), preproc)
                for j, i in enumerate(quiet_idx)])
            rec_pool.append(recon_score(model, inj, args.device))
            en_pool.append(frame_energy(inj))
            st_pool.append(frame_stats(inj))
        rec_pool = np.concatenate(rec_pool)
        en_pool = np.concatenate(en_pool)
        st_pool = np.concatenate(st_pool)
        mm = morphology_matched_energy_recon(rec_pool, en_pool, st_pool,
                                             rec_rfi, en_rfi, st_rfi, seed=args.seed)
        if mm is None:
            pass
        elif "error" in mm:
            print(f"  SKIPPED: {mm['error']}")
        else:
            print(f"  matched pairs: {mm['n_per_class']}/class  (caliper {mm['caliper']})")
            print(f"  energy_only AUC : {mm['energy_only']:.3f}   (~0.5 => energy neutralised)")
            print(f"  trivial   AUC   : {mm['trivial']:.3f}   (peakiness/kurtosis/top-pixel)")
            print(f"  recon     AUC   : {mm['recon']:.3f}")
            if mm["recon"] <= max(mm["trivial"], 0.58):
                print("    -> recon at matched energy is NO better than a trivial contrast stat "
                      "-> NOT morphology. The recon-B5 win is energy/contrast.")
            else:
                print("    -> recon at matched energy BEATS the trivial stat -> the model's "
                      "reconstruction error carries genuine morphology (real win).")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{args.model_config.stem}_{args.checkpoint.stem}".replace("=", "").replace(".", "p")
        out_npz = args.out_dir / f"recon_b5_{tag}.npz"
        if op is not None and "error" not in op:
            np.savez(out_npz, snr_list=np.array(args.snr_list),
                     rows=np.array(op["rows"]), n_rfi=op["n_rfi"], n_quiet=op["n_quiet"],
                     matched=np.array([mm.get("energy_only", np.nan), mm.get("trivial", np.nan),
                                       mm.get("recon", np.nan)]) if mm and "error" not in mm
                                     else np.array([np.nan, np.nan, np.nan]))
            print(f"\nSaved → {out_npz}")
        return

    emb_quiet = embed(model, preprocessed[quiet_idx], args.device)
    emb_rfi = embed(model, preprocessed[rfi_idx], args.device)
    en_rfi = frame_energy(preprocessed[rfi_idx])
    st_rfi = frame_stats(preprocessed[rfi_idx])
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
    inj_scores, inj_emb, inj_en, inj_st, results = {}, {}, {}, {}, []
    for snr in args.snr_list:
        inj = np.array([preprocess_raw(
            inject_narrowband_on_only(raw_snippets[i], snr=snr, drift_rate=args.drift_rate,
                                      seed=args.seed + j), preproc)
            for j, i in enumerate(quiet_idx)])
        e_inj = embed(model, inj, args.device)
        inj_emb[snr] = e_inj
        inj_en[snr] = frame_energy(inj)
        inj_st[snr] = frame_stats(inj)
        s = score(e_inj)
        inj_scores[snr] = s
        sigma = (s.mean() - b_mean) / b_std if b_std > 0 else 0.0
        det3, det5 = (s > t3).mean() * 100, (s > t5).mean() * 100
        results.append((snr, s.mean(), sigma, cohens_d(s, base), det3, det5))
        print(f"  {snr:5.0f}  {s.mean():8.4f}  {sigma:8.2f}σ  {cohens_d(s, base):8.2f}  "
              f"{det3:7.1f}%  {det5:7.1f}%")

    # ---- 3. MORPHOLOGY AT MATCHED ENERGY (the deciding test) ----
    # Pool ALL injected SNRs (narrowband adds little energy, so even SNR=50 sits low)
    # to maximise low-energy samples that overlap the RFI energy band.
    emb_pool = np.concatenate([inj_emb[s] for s in args.snr_list], 0)
    en_pool = np.concatenate([inj_en[s] for s in args.snr_list], 0)
    st_pool = np.concatenate([inj_st[s] for s in args.snr_list], 0)
    print(f"\n{'='*64}\n3. MORPHOLOGY AT MATCHED ENERGY  (injected vs RFI, energy-matched)\n{'='*64}")
    print(f"  raw energy range: injected {en_pool.min():.2f}–{en_pool.max():.2f} | "
          f"RFI {en_rfi.min():.2f}–{en_rfi.max():.2f}")
    m = morphology_matched_energy(emb_pool, en_pool, st_pool, emb_rfi, en_rfi, st_rfi, seed=args.seed)
    if m is not None and "error" in m:
        print(f"  SKIPPED: {m['error']}")
    elif m is not None:
        print(f"  matched pairs    : n/class = {m['n_per_class']}  (caliper = {m['caliper']:.3f})")
        print(f"  AUC energy-only  : {m['energy_only']:.3f}   (sanity: ~0.5 means matching worked)")
        print(f"  AUC trivial-stats: {m['trivial']:.3f}   (peakiness/kurtosis/top-pixel — the baseline to beat)")
        print(f"  AUC embedding    : {m['embedding']:.3f}   <-- must beat trivial-stats to justify the AE")
        if m["n_per_class"] < 25:
            print("  WARNING: few matched samples — AUCs underpowered; treat as indicative only.")
        if m["energy_only"] > 0.60:
            print("  WARNING: energy not neutralised even at the tightest caliper — embedding AUC suspect.")
        elif m["embedding"] > m["trivial"] + 0.07:
            print("  VERDICT: embedding beats trivial shape stats at matched energy -> the encoder "
                  "carries genuine morphology -> feature scoring justified. NOTE: this is a "
                  "SUPERVISED upper bound -> pursue a probe trained on synthetic injections, "
                  "not (only) an unsupervised one-class.")
        else:
            print("  VERDICT: embedding ~= trivial shape stats -> the AE re-encodes a peakiness "
                  "statistic a 2-line detector captures; the feature-scoring pivot adds little. "
                  "Reconsider objective/target.")

    # ---- 4. UNSUPERVISED ONE-CLASS DETECTORS (no labels) ----
    print(f"\n{'='*64}\n4. UNSUPERVISED ONE-CLASS on frozen embedding (vs recon's SNR>=20 floor)\n{'='*64}")
    occ = unsupervised_occ(emb_quiet, emb_rfi, inj_emb, args.snr_list, seed=args.seed)
    if occ is not None:
        for name, r in occ.items():
            print(f"\n  [{name}]  RFI Cohen's d vs quiet = {r['rfi_d']:+.2f}")
            print(f"    {'SNR':>5s}  {'det@3σ':>8s}  {'det@5σ':>8s}")
            floor = None
            for s in sorted(args.snr_list):
                d3, d5 = r["det3"][s], r["det5"][s]
                if floor is None and d3 >= 50:
                    floor = s
                print(f"    {s:5.0f}  {d3:7.1f}%  {d5:7.1f}%")
            tag = (f"detects from SNR≈{floor:.0f}" if floor is not None else "never reaches 50% det@3σ")
            print(f"    -> {name}: {tag}")
        floors = [min((s for s in sorted(args.snr_list) if r['det3'][s] >= 50), default=None)
                  for r in occ.values()]
        best = min([f for f in floors if f is not None], default=None)
        print(f"\n  VERDICT: best unsupervised OCC floor = "
              f"{('SNR≈%.0f' % best) if best else 'none <50%'}  vs recon SNR≈20.")
        if best is not None and best < 20:
            print("    -> unsupervised feature scoring BEATS recon — pursue it (no labels needed).")
        else:
            print("    -> no unsupervised OCC beats recon: the morphology is real but sits inside "
                  "the normal density. Fully-unsupervised gains need a new OBJECTIVE "
                  "(denoising / intermediate-mask / ON->OFF prediction), not just a new scorer.")

    # ---- 5. OPERATIONAL: signal vs RFI, OCC fit on FULL background ----
    print(f"\n{'='*64}\n5. OPERATIONAL — signal vs RFI (OCC fit on FULL background, no labels)\n{'='*64}")
    op = operational_signal_vs_rfi(model, preprocessed, raw_snippets, quiet_idx, rfi_idx,
                                   preproc, args.snr_list, args.drift_rate, args.device,
                                   scoring="embedding", seed=args.seed)
    print_operational(op, scoring="embedding")

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
