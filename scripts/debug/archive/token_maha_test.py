"""Per-token max Mahalanobis anomaly scoring — no retraining required.

Fixes the 9/384 dilution that kills mean-pooled Mahalanobis on localised signals:

  CURRENT (mean_pool):  encode(x) -> mean over 384 tokens -> 1×128 vector -> maha
  NEW     (token_max):  encode_tokens(x) -> 384×128 matrix -> maha per token -> MAX

A narrowband signal occupies ~9 patches; after mean-pool their contribution is
diluted to 2.3% of the vector. Per-token max catches the single most anomalous
patch directly.

Three scorers compared (all from the SAME frozen checkpoint, no retraining):

  token_max   : max Mahalanobis over all 384 tokens   (LW fit on N_quiet×384 tokens)
  token_top9  : mean of top-9 token Mahalanobis       (signal occupies ~9 patches)
  mean_pool   : Mahalanobis of encode(x) mean-pooled  (current approach — reference)

Reference numbers to beat:
  mean_pool   B3 AUC ≈ 0.746   B5 AUC@SNR10 ≈ 0.927
  peak_snr    B3 AUC ≈ 0.665   B5 AUC@SNR10 ≈ 0.813  (statistical baseline)

Usage (server, not dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/acabras/data/filippo/BL-Exotica-AD \\
    python scripts/debug/token_maha_test.py \\
        --checkpoint outputs/training/20260624_084754_057f87c/checkpoints/epoch=006-val_loss=2.1715.ckpt \\
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \\
        --out_dir outputs/sweeps/token_maha
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


# ---------------------------------------------------------------------------
# Model loading + token extraction
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    if not hasattr(model, "encode_tokens"):
        raise SystemExit("Model has no encode_tokens() — needs ViT-MAE backbone.")
    return model


@torch.no_grad()
def get_tokens(model, snippets: np.ndarray, device: str, batch: int = 32) -> np.ndarray:
    """(N, H, W) preprocessed -> (N, num_patches, embed_dim) token matrix."""
    out = []
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model.encode_tokens(x).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def get_mean_emb(model, snippets: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(N, H, W) preprocessed -> (N, embed_dim) mean-pooled embedding."""
    out = []
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Mahalanobis fit + scorers
# ---------------------------------------------------------------------------

def fit_lw(matrix: np.ndarray):
    """Fit Ledoit-Wolf covariance on a (N, D) matrix. Returns fitted LedoitWolf."""
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(matrix)


def maha_scores(lw, matrix: np.ndarray) -> np.ndarray:
    """Mahalanobis distances: sqrt of lw.mahalanobis((N, D)) -> (N,)."""
    return np.sqrt(np.clip(lw.mahalanobis(matrix), 0, None))


def scorer_token_max(tokens: np.ndarray, lw_tok) -> np.ndarray:
    """(N, P, D) -> (N,): max Mahalanobis over all P tokens."""
    N, P, D = tokens.shape
    dists = maha_scores(lw_tok, tokens.reshape(N * P, D)).reshape(N, P)
    return dists.max(axis=1)


def scorer_token_topk(tokens: np.ndarray, lw_tok, k: int = 9) -> np.ndarray:
    """(N, P, D) -> (N,): mean of top-k Mahalanobis distances."""
    N, P, D = tokens.shape
    k = min(k, P)
    dists = maha_scores(lw_tok, tokens.reshape(N * P, D)).reshape(N, P)
    return np.sort(dists, axis=1)[:, -k:].mean(axis=1)


def scorer_mean_pool(mean_emb: np.ndarray, lw_mean) -> np.ndarray:
    """(N, D) -> (N,): Mahalanobis of the mean-pooled embedding (current approach)."""
    return maha_scores(lw_mean, mean_emb)


# ---------------------------------------------------------------------------
# Shared utilities (mirrors encode_separation_test.py)
# ---------------------------------------------------------------------------

def frame_energy(frames: np.ndarray) -> np.ndarray:
    return (frames.astype(np.float64) ** 2).mean(axis=(1, 2))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def _caliper_match(en_small, en_big, caliper, rng):
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


# ---------------------------------------------------------------------------
# Block B2: detection rate vs SNR (calibrated on quiet baseline)
# ---------------------------------------------------------------------------

def block_b2(scores_quiet, scores_rfi, scores_inj_by_snr, snr_list, scorer_name):
    """Print detection table for one scorer. Returns list of (snr, det3, det5)."""
    bm, bs = scores_quiet.mean(), scores_quiet.std()
    t3, t5 = bm + 3 * bs, bm + 5 * bs
    print(f"\n  [{scorer_name}]")
    print(f"  quiet={bm:.4f}±{bs:.4f}  RFI={scores_rfi.mean():.4f}±{scores_rfi.std():.4f}"
          f"  (Cohen's d={cohens_d(scores_rfi, scores_quiet):+.2f})")
    print(f"  {'SNR':>5s}  {'mean':>8s}  {'sigma':>8s}  {'cohen_d':>8s}"
          f"  {'det@3σ':>8s}  {'det@5σ':>8s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    rows = []
    for snr in snr_list:
        s = scores_inj_by_snr[snr]
        sigma = (s.mean() - bm) / (bs + 1e-12)
        det3 = float((s > t3).mean() * 100)
        det5 = float((s > t5).mean() * 100)
        rows.append((snr, s.mean(), sigma, cohens_d(s, scores_quiet), det3, det5))
        print(f"  {snr:5.0f}  {s.mean():8.4f}  {sigma:8.2f}σ  {cohens_d(s, scores_quiet):8.2f}"
              f"  {det3:7.1f}%  {det5:7.1f}%")
    floor3 = next((r[0] for r in rows if r[4] >= 50), None)
    print(f"  -> floor @50% det@3σ: {f'SNR≈{floor3:.0f}' if floor3 else 'never'}")
    return rows, (bm, bs)


# ---------------------------------------------------------------------------
# Block B3: morphology at matched energy
# ---------------------------------------------------------------------------

def block_b3(score_fn_dict, inj_frames_all, rfi_frames, seed=0):
    """AUC at matched energy for each scorer.
    score_fn_dict: {name: callable(frames) -> (N,) scores}
    """
    from sklearn.metrics import roc_auc_score
    en_inj = frame_energy(inj_frames_all)
    en_rfi = frame_energy(rfi_frames)
    rng = np.random.default_rng(seed)

    print(f"\n{'='*70}")
    print("B3. MORPHOLOGY AT MATCHED ENERGY  (injected vs RFI, energy-matched)")
    print(f"{'='*70}")
    print(f"  energy range: injected {en_inj.min():.3f}–{en_inj.max():.3f} | "
          f"RFI {en_rfi.min():.3f}–{en_rfi.max():.3f}")

    best = None
    for caliper in [0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
        ps, pb = _caliper_match(en_rfi, en_inj, caliper, rng)
        if len(ps) < 20:
            continue
        e = np.concatenate([en_inj[pb], en_rfi[ps]], 0)[:, None]
        y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
        ea = float(roc_auc_score(y, e))
        cand = {"caliper": caliper, "n": len(ps), "energy_auc": ea, "ps": ps, "pb": pb}
        if ea <= 0.58:
            best = cand
            break
        if best is None or ea < best["energy_auc"]:
            best = cand

    if best is None:
        print("  SKIPPED: too few energy-matched pairs.")
        return {}

    ps, pb = best["ps"], best["pb"]
    y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
    n = best["n"]
    print(f"\n  matched pairs: n/class={n}  caliper={best['caliper']:.3f}"
          f"  energy AUC={best['energy_auc']:.3f}")
    if n < 25:
        print("  WARNING: few matched samples — treat as indicative only.")

    print(f"\n  {'Scorer':<22s}  {'AUC (matched-energy)':>20s}  Notes")
    print(f"  {'-'*22}  {'-'*20}  {'-'*38}")

    b3_results = {}
    for name, fn in score_fn_dict.items():
        s_inj = fn(inj_frames_all[pb])
        s_rfi = fn(rfi_frames[ps])
        auc = float(roc_auc_score(y, np.concatenate([s_inj, s_rfi])))
        b3_results[name] = auc
        beat = "  <- beats mean_pool (0.746)" if auc > 0.746 else (
               "  <- beats stat baseline (0.665)" if auc > 0.665 else "")
        print(f"  {name:<22s}  {auc:20.3f}  {beat}")

    print(f"  {'  [mean_pool ref]':<22s}  {'0.746':>20s}  (encode_separation_test B3, run 20260624)")
    print(f"  {'  [peak_snr ref]':<22s}  {'0.665':>20s}  (statistical_baseline B3)")
    return b3_results


# ---------------------------------------------------------------------------
# Block B5: operational signal vs RFI
# ---------------------------------------------------------------------------

def block_b5(score_fn_dict, preprocessed, inj_by_snr, quiet_idx, rfi_idx,
             snr_list, ev_quiet_pos, fit_mask):
    """AUC(signal vs RFI) + TPR@10% RFI-FP for each scorer."""
    from sklearn.metrics import roc_auc_score

    ev_rfi = [i for i in rfi_idx if not fit_mask[i]]
    ev_quiet = [i for i in quiet_idx if not fit_mask[i]]

    print(f"\n{'='*70}")
    print("B5. OPERATIONAL — signal vs RFI (no labels, 70/30 split)")
    print(f"{'='*70}")

    if len(ev_rfi) < 10 or len(ev_quiet) < 10:
        print(f"  SKIPPED: held-out too small (RFI={len(ev_rfi)}, quiet={len(ev_quiet)}); "
              "raise --n_samples")
        return {}

    print(f"  held-out: RFI(neg)={len(ev_rfi)}  quiet→inject(pos)={len(ev_quiet)}")

    b5_results = {}
    for name, fn in score_fn_dict.items():
        neg = fn(preprocessed[ev_rfi])
        thr = np.percentile(neg, 90)

        print(f"\n  [{name}]  RFI range [{neg.min():.3f}, {neg.max():.3f}]"
              f"  threshold@10%FP={thr:.3f}")
        print(f"  {'SNR':>5s}  {'AUC(sig vs RFI)':>16s}  {'TPR@10%RFI-FP':>14s}")
        print(f"  {'-'*5}  {'-'*16}  {'-'*14}")

        rows = []
        floor = None
        for snr in snr_list:
            pos = fn(inj_by_snr[snr][ev_quiet_pos])
            auc = float(roc_auc_score(
                np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
                np.concatenate([pos, neg])
            ))
            tpr = float((pos > thr).mean() * 100)
            if floor is None and tpr >= 50:
                floor = snr
            rows.append((snr, auc, tpr))
            print(f"  {snr:5.0f}  {auc:16.3f}  {tpr:13.1f}%")

        tag = f"SNR≈{floor:.0f}" if floor else "never ≥50%"
        print(f"  -> signal beats 90% of RFI from {tag} (@10% RFI-FP)")
        b5_results[name] = {"rows": rows, "floor": floor}

    return b5_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    p.add_argument("--top_k", type=int, default=9,
                   help="k for token_topk scorer (default 9, ~signal patch footprint)")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/token_maha")
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
    num_patches = model.num_patches
    embed_dim = model.patch_embed.proj.out_channels
    print(f"  num_patches={num_patches}  embed_dim={embed_dim}")

    # ---- Load + preprocess ----
    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    idx = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[idx])
    del arr
    print(f"  Raw snippets: {raw_snippets.shape}")

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

    # ---- B5 split (before injection) ----
    rng_b5 = np.random.default_rng(args.seed + 1)
    perm = rng_b5.permutation(len(preprocessed))
    fit_mask = np.zeros(len(preprocessed), bool)
    fit_mask[perm[: int(0.7 * len(perm))]] = True
    quiet_pos = {int(v): j for j, v in enumerate(quiet_idx)}
    ev_quiet_pos = np.array([quiet_pos[i] for i in quiet_idx if not fit_mask[i]])

    # ---- Encode tokens ----
    print(f"Encoding tokens for {len(quiet_idx)} quiet + {len(rfi_idx)} RFI snippets...")
    tok_quiet = get_tokens(model, preprocessed[quiet_idx], args.device)
    tok_rfi = get_tokens(model, preprocessed[rfi_idx], args.device)
    emb_quiet = get_mean_emb(model, preprocessed[quiet_idx], args.device)
    emb_rfi = get_mean_emb(model, preprocessed[rfi_idx], args.device)
    print(f"  tok_quiet shape: {tok_quiet.shape}  "
          f"({tok_quiet.shape[0] * tok_quiet.shape[1]} total tokens for LW fit)")

    # ---- Fit Mahalanobis ----
    print("Fitting Ledoit-Wolf covariances...")
    N, P, D = tok_quiet.shape
    lw_tok = fit_lw(tok_quiet.reshape(N * P, D))   # on all tokens (N*384 samples)
    lw_mean = fit_lw(emb_quiet)                     # on mean embeddings (N samples)
    print(f"  lw_tok: {N * P} samples × {D}-dim")
    print(f"  lw_mean: {N} samples × {D}-dim  (reference — same as encode_separation_test)")

    # ---- Inject all SNRs into quiet snippets ----
    print(f"Injecting {len(args.snr_list)} SNR levels × {len(quiet_idx)} quiet snippets...")
    inj_by_snr = {}
    for snr in args.snr_list:
        inj_by_snr[snr] = np.array([
            preprocess_raw(
                inject_narrowband_on_only(raw_snippets[i], snr=snr,
                                          drift_rate=args.drift_rate,
                                          seed=args.seed + j),
                preproc)
            for j, i in enumerate(quiet_idx)
        ])
    inj_all = np.concatenate(list(inj_by_snr.values()), axis=0)
    print("  Done.")

    # ---- Define scorer functions (each maps frames -> scores) ----
    # Token-based scorers need to go through the model; wrap them to accept frames.
    def score_tok_max(frames):
        tok = get_tokens(model, frames, args.device)
        return scorer_token_max(tok, lw_tok)

    def score_tok_topk(frames):
        tok = get_tokens(model, frames, args.device)
        return scorer_token_topk(tok, lw_tok, k=args.top_k)

    def score_mean(frames):
        emb = get_mean_emb(model, frames, args.device)
        return scorer_mean_pool(emb, lw_mean)

    score_fns = {
        "token_max": score_tok_max,
        f"token_top{args.top_k}": score_tok_topk,
        "mean_pool": score_mean,
    }

    # ---- Background LW for B5: fit on ALL fit-set tokens (quiet + RFI) ----
    # This mirrors encode_separation_test.py block 5 exactly: LW is trained on the
    # combined background so RFI is "normal" and the narrowband signal stands out
    # from BOTH quiet and RFI — the operationally correct setup.
    print(f"\nFitting background LW on {fit_mask.sum()} fit-set snippets (quiet + RFI)...")
    tok_bg_fit = get_tokens(model, preprocessed[fit_mask], args.device)
    N_bg, P_bg, D_bg = tok_bg_fit.shape
    lw_bg = fit_lw(tok_bg_fit.reshape(N_bg * P_bg, D_bg))
    print(f"  lw_bg: {N_bg * P_bg} samples × {D_bg}-dim  (quiet + RFI, comparable to encode_separation_test B5)")
    del tok_bg_fit

    def score_tok_max_bg(frames):
        tok = get_tokens(model, frames, args.device)
        return scorer_token_max(tok, lw_bg)

    def score_tok_topk_bg(frames):
        tok = get_tokens(model, frames, args.device)
        return scorer_token_topk(tok, lw_bg, k=args.top_k)

    # B5 scorers: bg-fit versions first, then quiet-only for comparison
    score_fns_b5 = {
        "token_max_bg": score_tok_max_bg,
        f"token_top{args.top_k}_bg": score_tok_topk_bg,
        "token_max": score_tok_max,
        "mean_pool": score_mean,
    }

    # ---- Pre-compute scores on quiet, RFI, injected (for B2 + B3) ----
    print("\nComputing scores on quiet / RFI / injected frames...")
    scores_quiet = {n: fn(preprocessed[quiet_idx]) for n, fn in score_fns.items()}
    scores_rfi = {n: fn(preprocessed[rfi_idx]) for n, fn in score_fns.items()}
    scores_inj_by_snr = {
        n: {snr: fn(inj_by_snr[snr]) for snr in args.snr_list}
        for n, fn in score_fns.items()
    }
    print("  Done.")

    # ---- B2 ----
    print(f"\n{'='*70}")
    print("B2. DETECTION RATE vs SNR (calibrated on quiet baseline)")
    print(f"{'='*70}")
    b2_results = {}
    for name in score_fns:
        rows, base = block_b2(scores_quiet[name], scores_rfi[name],
                              scores_inj_by_snr[name], args.snr_list, name)
        b2_results[name] = {"rows": rows, "base": base}

    # ---- B3 ----
    # score_fn_dict must map names to functions that accept (N, H, W) frames
    b3_results = block_b3(score_fns, inj_all, preprocessed[rfi_idx], seed=args.seed)

    # ---- B5 ----
    b5_results = block_b5(score_fns_b5, preprocessed, inj_by_snr, quiet_idx, rfi_idx,
                          args.snr_list, ev_quiet_pos, fit_mask)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\n  Block B3 — matched-energy AUC:")
    refs_b3 = {"mean_pool (ref)": 0.746, "peak_snr stat (ref)": 0.665}
    for name, auc in b3_results.items():
        vs_mean = auc - 0.746
        print(f"    {name:<22s}: AUC={auc:.3f}  (vs mean_pool {vs_mean:+.3f})")
    print(f"    {'mean_pool (ref)':<22s}: AUC=0.746  (encode_separation_test B3)")
    print(f"    {'peak_snr stat (ref)':<22s}: AUC=0.665  (statistical_baseline B3)")

    print("\n  Block B5 — operational AUC@SNR10 and detection floor:")
    print("  (NOTE: *_bg scorers use full-background LW — directly comparable to encode_separation_test B5)")
    snr10 = 10.0
    for name, r in b5_results.items():
        floor = r.get("floor")
        tag = f"SNR≈{floor:.0f}" if floor else "never"
        row10 = next((row for row in r["rows"] if row[0] == snr10), None)
        auc10 = f"{row10[1]:.3f}" if row10 else "N/A"
        if row10:
            delta = row10[1] - 0.927
            flag = " [BEATS ref!]" if delta > 0 else f" ({delta:+.3f} vs ref 0.927)"
        else:
            flag = ""
        print(f"    {name:<24s}: AUC@SNR10={auc10}  floor={tag}{flag}")
    print(f"    {'mean_pool (ref)':<24s}: AUC@SNR10=0.927  floor=? (encode_separation_test B5, bg-fit)")
    print(f"    {'peak_snr stat (ref)':<24s}: AUC@SNR10=0.813  floor=SNR≈15  (statistical_baseline)")

    # ---- Plots ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snrs = sorted(args.snr_list)

    # B2: det@3σ vs SNR
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in score_fns:
        rows = b2_results[name]["rows"]
        axes[0].plot(snrs, [r[4] for r in rows], marker="o", label=name)
        axes[1].plot(snrs, [r[5] for r in rows], marker="o", label=name)
    for ax, sig in zip(axes, [3, 5]):
        ax.set(xlabel="Injection SNR", ylabel="Detection rate (%)",
               title=f"B2: det@{sig}σ vs SNR (quiet-calibrated)", ylim=(-5, 105))
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_dir / "b2_detection_vs_snr.png", dpi=150)
    plt.close()
    print(f"\nSaved → {args.out_dir / 'b2_detection_vs_snr.png'}")

    # B5: AUC vs SNR — split into bg-fit and quiet-only panels
    if b5_results:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        styles = {"_bg": "-", "token_max": "--", "mean_pool": ":"}
        for name, r in b5_results.items():
            ls = "-" if name.endswith("_bg") else ("--" if "max" in name else ":")
            axes[0].plot(snrs, [row[1] for row in r["rows"]], ls, marker="o",
                         label=name, lw=2 if name.endswith("_bg") else 1)
            axes[1].plot(snrs, [row[2] for row in r["rows"]], ls, marker="o",
                         label=name, lw=2 if name.endswith("_bg") else 1)
        axes[0].axhline(0.927, ls="--", color="red", alpha=0.7,
                        label="encode_sep ref 0.927 (bg-fit)")
        axes[0].axhline(0.813, ls=":", color="gray", alpha=0.6,
                        label="peak_snr stat 0.813")
        axes[0].set(xlabel="Injection SNR", ylabel="AUC (signal vs RFI)",
                    title="B5: AUC(signal vs RFI) vs SNR\n(bold = bg-fit LW)", ylim=(0.3, 1.05))
        axes[0].legend(fontsize=7)
        axes[1].axhline(50, ls="--", color="gray", alpha=0.5, label="50% TPR")
        axes[1].set(xlabel="Injection SNR", ylabel="TPR@10% RFI-FP (%)",
                    title="B5: operational TPR vs SNR\n(bold = bg-fit LW)", ylim=(-5, 105))
        axes[1].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(args.out_dir / "b5_vs_snr.png", dpi=150)
        plt.close()
        print(f"Saved → {args.out_dir / 'b5_vs_snr.png'}")

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
