"""Model-free statistical baseline for anomaly detection.

Replaces the ViT-MAE embedding in encode_separation_test.py with four model-free
scorers, running through equivalent blocks B2, B3 and B5 (same harness, same
cache, same injection function). No model checkpoint needed.

Purpose
-------
- Establish a concrete bar any deep-learning architecture must beat.
- Diagnose whether the injection pipeline is working correctly before attributing
  failures to architecture.

Expected outcome guide
----------------------
  B3 (morphology at matched energy):
    max_pixel AUC > 0.60  → injection works; mean-pooling is the architecture
                             bottleneck (9/384 patches dilute localised signals).
    max_pixel AUC ≈ 0.50  → energy-matching discards detectable pairs, OR
                             injection/preprocessing bug — audit first.

  B5 (operational, signal vs RFI):
    any scorer AUC > 0.93 → simple threshold detector beats ViT-MAE on this
                             product; deep learning adds negative value. Consider
                             a matched-filter / peak-SNR threshold pipeline.

Scorers tested
--------------
  max_pixel        : preprocessed.max()
  peak_snr         : max / std  (energy-invariant peakiness)
  top1pct_energy   : sum of squared brightest 1% of pixels
  frame_stats      : [peakiness, excess kurtosis, top-0.1% fraction]
                     → supervised logistic AUC in B3
                     → OneClassSVM in B5

Reference
---------
  ViT-MAE (20260624_084754):  B3 trivial AUC ≈ 0.529, B3 embedding AUC ≈ 0.746
                               B5 AUC @ SNR 10 ≈ 0.927

Usage (server, not dev machine):
    PYTHONPATH=/content/filippo/BL-Exotica-AD \\
    python scripts/debug/statistical_baseline.py \\
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \\
        --out_dir outputs/sweeps/statistical_baseline
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.debug.injection_vs_rfi_test import preprocess_raw, inject_narrowband_on_only


# ---------------------------------------------------------------------------
# Scorer functions:  (N, H, W) preprocessed frames  →  (N,) anomaly scores
# ---------------------------------------------------------------------------

def _flat(frames: np.ndarray) -> np.ndarray:
    return frames.reshape(len(frames), -1).astype(np.float64)


def sc_max(frames: np.ndarray) -> np.ndarray:
    """Max pixel value — sensitive to a single bright pixel."""
    return _flat(frames).max(axis=1)


def sc_peak_snr(frames: np.ndarray) -> np.ndarray:
    """Max / std — energy-invariant peakiness."""
    g = _flat(frames)
    return g.max(axis=1) / (g.std(axis=1) + 1e-9)


def sc_top1pct(frames: np.ndarray) -> np.ndarray:
    """Sum of squared values of the top-1% brightest pixels."""
    g = _flat(frames)
    k = max(1, int(0.01 * g.shape[1]))
    return np.sort(g ** 2, axis=1)[:, -k:].sum(axis=1)


def sc_frame_stats_vec(frames: np.ndarray) -> np.ndarray:
    """3-D feature vector per frame: [peakiness, excess kurtosis, top-0.1%-fraction].
    Same as frame_stats() in encode_separation_test.py — the 'trivial' baseline."""
    g = _flat(frames)
    mu = g.mean(1, keepdims=True)
    sd = g.std(1) + 1e-9
    peak = g.max(1) / sd
    kurt = ((g - mu) ** 4).mean(1) / (sd ** 4) - 3.0
    k = max(1, int(0.001 * g.shape[1]))
    sq = np.sort(g ** 2, axis=1)
    topfrac = sq[:, -k:].sum(1) / (sq.sum(1) + 1e-9)
    return np.stack([peak, kurt, topfrac], axis=1)


SCALAR_SCORERS = {
    "max_pixel": sc_max,
    "peak_snr": sc_peak_snr,
    "top1pct_energy": sc_top1pct,
}


# ---------------------------------------------------------------------------
# Utilities (mirrors encode_separation_test.py)
# ---------------------------------------------------------------------------

def frame_energy(frames: np.ndarray) -> np.ndarray:
    return (frames.astype(np.float64) ** 2).mean(axis=(1, 2))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def _caliper_match(en_small, en_big, caliper, rng):
    """Greedy 1:1 nearest-energy matching without replacement.
    Mirrors encode_separation_test.py._caliper_match exactly."""
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
# Block B2: detection rate vs SNR (scalar analogue of block 2 in encode_separation_test)
# ---------------------------------------------------------------------------

def block_b2(quiet_frames, rfi_frames, inj_by_snr, snr_list):
    """For each scalar scorer: mean, sigma, Cohen's d, det@3σ/5σ vs SNR.

    Calibration: threshold = quiet mean + N * quiet std.
    """
    print(f"\n{'='*70}")
    print("B2. DETECTION RATE vs SNR (calibrated on quiet baseline)")
    print(f"{'='*70}")

    b2_results = {}
    for name, fn in SCALAR_SCORERS.items():
        base = fn(quiet_frames)
        rfi_s = fn(rfi_frames)
        bm, bs = base.mean(), base.std()
        t3, t5 = bm + 3 * bs, bm + 5 * bs

        print(f"\n  [{name}]")
        print(f"  quiet={bm:.4f}±{bs:.4f}  RFI={rfi_s.mean():.4f}±{rfi_s.std():.4f}"
              f"  (Cohen's d vs quiet = {cohens_d(rfi_s, base):+.2f})")
        print(f"  {'SNR':>5s}  {'mean':>8s}  {'sigma':>8s}  {'cohen_d':>8s}"
              f"  {'det@3σ':>8s}  {'det@5σ':>8s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        snr_rows = []
        for snr in snr_list:
            s = fn(inj_by_snr[snr])
            sigma = (s.mean() - bm) / (bs + 1e-12)
            det3 = float((s > t3).mean() * 100)
            det5 = float((s > t5).mean() * 100)
            d = cohens_d(s, base)
            snr_rows.append((snr, s.mean(), sigma, d, det3, det5))
            print(f"  {snr:5.0f}  {s.mean():8.4f}  {sigma:8.2f}σ  {d:8.2f}"
                  f"  {det3:7.1f}%  {det5:7.1f}%")

        floor3 = next((r[0] for r in snr_rows if r[4] >= 50), None)
        tag = f"SNR≈{floor3:.0f}" if floor3 else "never ≥50% @3σ"
        print(f"  -> detection floor (50% @3σ): {tag}")
        b2_results[name] = {"rows": snr_rows, "base": (bm, bs)}

    return b2_results


# ---------------------------------------------------------------------------
# Block B3: morphology at matched energy (mirrors block 3 of encode_separation_test)
# ---------------------------------------------------------------------------

def block_b3(quiet_frames, rfi_frames, inj_frames_all, seed=0):
    """At matched input energy, can simple stats distinguish injected narrowband
    from RFI?  Mirrors morphology_matched_energy() in encode_separation_test.py.

    For scalar scorers: roc_auc_score on matched pairs.
    For frame_stats: 5-fold logistic AUC (same as 'trivial' in block 3).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  (scikit-learn not available — skipping block B3)")
        return None

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
        print("  SKIPPED: too few energy-matched pairs — RFI scarce in the injected energy band.")
        print("           (narrowband raises energy very little; tightest caliper yields < 20 pairs)")
        return None

    ps, pb = best["ps"], best["pb"]
    y = np.concatenate([np.ones(len(pb)), np.zeros(len(ps))])
    n = best["n"]
    print(f"\n  matched pairs: n/class={n}  caliper={best['caliper']:.3f}"
          f"  energy AUC={best['energy_auc']:.3f} (sanity: ~0.5 = matching worked)")
    if n < 25:
        print("  WARNING: few matched samples — AUCs underpowered; treat as indicative only.")
    if best["energy_auc"] > 0.60:
        print("  WARNING: energy not fully neutralised — AUCs may partially reflect energy bias.")

    def lr_auc(feats, y_):
        return float(cross_val_score(
            LogisticRegression(max_iter=2000),
            StandardScaler().fit_transform(feats), y_,
            cv=5, scoring="roc_auc").mean())

    print(f"\n  {'Scorer':<22s}  {'AUC (matched-energy)':>20s}  Notes")
    print(f"  {'-'*22}  {'-'*20}  {'-'*38}")

    b3_results = {}
    for name, fn in SCALAR_SCORERS.items():
        s_inj = fn(inj_frames_all[pb])
        s_rfi = fn(rfi_frames[ps])
        auc = float(roc_auc_score(y, np.concatenate([s_inj, s_rfi])))
        b3_results[name] = auc
        flag = "  <- beats chance" if auc > 0.60 else ""
        print(f"  {name:<22s}  {auc:20.3f}  {flag}")

    fs_inj = sc_frame_stats_vec(inj_frames_all[pb])
    fs_rfi = sc_frame_stats_vec(rfi_frames[ps])
    fs_auc = lr_auc(np.concatenate([fs_inj, fs_rfi], 0), y)
    b3_results["frame_stats"] = fs_auc
    print(f"  {'frame_stats':<22s}  {fs_auc:20.3f}  (5-fold logistic on 3D features)")
    print(f"  {'  [ViT-MAE embed ref]':<22s}  {'0.746':>20s}  (block 3, run 20260624)")
    print(f"  {'  [ViT-MAE trivial ref]':<22s}  {'0.529':>20s}  (block 3 trivial, same run)")

    best_auc = max(b3_results.values())
    best_name = max(b3_results, key=b3_results.get)
    print(f"\n  Best baseline scorer: [{best_name}]  AUC={best_auc:.3f}")
    if best_auc > 0.60:
        print("  VERDICT: injection pipeline is healthy. A simple scalar threshold")
        print("           distinguishes narrowband from energy-matched RFI.")
        print("           -> mean-pooling (9/384 tokens) is the architecture bottleneck.")
        print("              Next step: per-token max Mahalanobis on existing MAE checkpoint.")
    else:
        print("  VERDICT: AUC near chance at matched energy.")
        print("           -> (a) energy-matching discards all detectable pairs, OR")
        print("              (b) injection/preprocessing bug — audit before building.")

    return b3_results


# ---------------------------------------------------------------------------
# Block B5: operational signal vs RFI (mirrors block 5 of encode_separation_test)
# ---------------------------------------------------------------------------

def block_b5(preprocessed, inj_by_snr, quiet_idx, rfi_idx, snr_list,
             ev_quiet_pos, fit_mask, seed=0):
    """OCC fit on full background (quiet + RFI), no labels.

    Mirrors operational_signal_vs_rfi() in encode_separation_test.py.
    70/30 split: held-out RFI = negatives; held-out quiet + injection = positives.
    Threshold: 90th percentile of held-out RFI scores (10% RFI-FP operating point).
    """
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  (scikit-learn not available — skipping block B5)")
        return None

    ev_rfi = [i for i in rfi_idx if not fit_mask[i]]
    ev_quiet = [i for i in quiet_idx if not fit_mask[i]]

    print(f"\n{'='*70}")
    print("B5. OPERATIONAL — signal vs RFI (scorer on FULL background, no labels)")
    print(f"{'='*70}")

    if len(ev_rfi) < 10 or len(ev_quiet) < 10:
        print(f"  SKIPPED: held-out too small (RFI {len(ev_rfi)}, quiet {len(ev_quiet)}); "
              "raise --n_samples")
        return None

    print(f"  held-out: RFI(neg)={len(ev_rfi)}  quiet→inject(pos)={len(ev_quiet)}")

    b5_results = {}

    # Scalar scorers
    for name, fn in SCALAR_SCORERS.items():
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

    # frame_stats + OneClassSVM
    try:
        fit_feats = sc_frame_stats_vec(preprocessed[fit_mask])
        sc_std = StandardScaler().fit(fit_feats)
        svm = OneClassSVM(nu=0.1, gamma="scale").fit(sc_std.transform(fit_feats))

        def svm_score(frames):
            return -svm.decision_function(sc_std.transform(sc_frame_stats_vec(frames)))

        neg_svm = svm_score(preprocessed[ev_rfi])
        thr_svm = np.percentile(neg_svm, 90)

        print(f"\n  [frame_stats_ocsvm]")
        print(f"  {'SNR':>5s}  {'AUC(sig vs RFI)':>16s}  {'TPR@10%RFI-FP':>14s}")
        print(f"  {'-'*5}  {'-'*16}  {'-'*14}")

        rows_svm = []
        floor_svm = None
        for snr in snr_list:
            pos = svm_score(inj_by_snr[snr][ev_quiet_pos])
            auc = float(roc_auc_score(
                np.concatenate([np.ones(len(pos)), np.zeros(len(neg_svm))]),
                np.concatenate([pos, neg_svm])
            ))
            tpr = float((pos > thr_svm).mean() * 100)
            if floor_svm is None and tpr >= 50:
                floor_svm = snr
            rows_svm.append((snr, auc, tpr))
            print(f"  {snr:5.0f}  {auc:16.3f}  {tpr:13.1f}%")

        tag = f"SNR≈{floor_svm:.0f}" if floor_svm else "never ≥50%"
        print(f"  -> signal beats 90% of RFI from {tag} (@10% RFI-FP)")
        b5_results["frame_stats_ocsvm"] = {"rows": rows_svm, "floor": floor_svm}

    except Exception as exc:
        print(f"  [frame_stats_ocsvm] FAILED: {exc}")

    return b5_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True,
                   help="Cache directory (contains train.npy / val.npy)")
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--n_samples", type=int, default=500,
                   help="Snippets sampled from cache (≥500 for reliable estimates)")
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50])
    p.add_argument("--drift_rate", type=float, default=0.3,
                   help="Injection drift rate in Hz/s")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "outputs/sweeps/statistical_baseline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]

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

    # ---- B5 split (done before injection so we can index inj_by_snr) ----
    rng_b5 = np.random.default_rng(args.seed + 1)
    perm = rng_b5.permutation(len(preprocessed))
    fit_mask = np.zeros(len(preprocessed), bool)
    fit_mask[perm[: int(0.7 * len(perm))]] = True

    # Map each quiet global index to its position in quiet_idx
    quiet_pos = {int(v): j for j, v in enumerate(quiet_idx)}
    ev_quiet_idx = [i for i in quiet_idx if not fit_mask[i]]
    ev_quiet_pos = np.array([quiet_pos[i] for i in ev_quiet_idx])

    # ---- Inject all SNR levels into quiet snippets ----
    print(f"Injecting {len(args.snr_list)} SNR levels × {len(quiet_idx)} quiet snippets "
          f"(N={len(args.snr_list) * len(quiet_idx)})...")
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
    print("  Done.")

    # Pool all SNRs → B3
    inj_all = np.concatenate(list(inj_by_snr.values()), axis=0)

    # ---- Run blocks ----
    b2_res = block_b2(preprocessed[quiet_idx], preprocessed[rfi_idx],
                      inj_by_snr, args.snr_list)
    b3_res = block_b3(preprocessed[quiet_idx], preprocessed[rfi_idx],
                      inj_all, seed=args.seed)
    b5_res = block_b5(preprocessed, inj_by_snr, quiet_idx, rfi_idx,
                      args.snr_list, ev_quiet_pos, fit_mask, seed=args.seed)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY — compare to ViT-MAE (B3 emb=0.746, B5 AUC@SNR10≈0.927)")
    print(f"{'='*70}")
    if b3_res:
        print("\n  Block B3 — matched-energy AUC (injected vs RFI):")
        for name, auc in b3_res.items():
            bar = " <- beats ViT trivial (0.529)" if auc > 0.529 else ""
            beats = " [beats ViT-MAE embed!]" if auc > 0.746 else ""
            print(f"    {name:<22s}: AUC={auc:.3f}{bar}{beats}")
    if b5_res:
        print("\n  Block B5 — operational floor (signal beats 90% of RFI):")
        snr10_col = next((snr for snr in args.snr_list if snr == 10.0), None)
        for name, r in b5_res.items():
            floor = r.get("floor")
            tag = f"SNR≈{floor:.0f}" if floor else "never ≥50%"
            snr10_auc = ""
            if snr10_col is not None:
                row10 = next((row for row in r["rows"] if row[0] == snr10_col), None)
                if row10:
                    cmp = " [> 0.927 beats ViT-MAE!]" if row10[1] > 0.927 else ""
                    snr10_auc = f"  AUC@SNR10={row10[1]:.3f}{cmp}"
            print(f"    {name:<22s}: floor {tag}{snr10_auc}")

    # ---- Plots ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snrs = sorted(args.snr_list)

    # B2: det@3σ and @5σ vs SNR
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in SCALAR_SCORERS:
        rows = b2_res[name]["rows"]
        det3 = [r[4] for r in rows]
        det5 = [r[5] for r in rows]
        axes[0].plot(snrs, det3, marker="o", label=name)
        axes[1].plot(snrs, det5, marker="o", label=name)
    axes[0].set(xlabel="Injection SNR", ylabel="Detection rate (%)",
                title="B2: det@3σ vs SNR (quiet-calibrated)", ylim=(-5, 105))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Injection SNR", ylabel="Detection rate (%)",
                title="B2: det@5σ vs SNR (quiet-calibrated)", ylim=(-5, 105))
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_dir / "b2_detection_vs_snr.png", dpi=150)
    plt.close()
    print(f"\nSaved → {args.out_dir / 'b2_detection_vs_snr.png'}")

    # B5: TPR@10% RFI-FP vs SNR
    if b5_res:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, r in b5_res.items():
            tprs = [row[2] for row in r["rows"]]
            ax.plot(snrs, tprs, marker="o", label=name)
        ax.axhline(50, ls="--", color="gray", alpha=0.5, label="50% TPR")
        ax.set(xlabel="Injection SNR", ylabel="TPR@10% RFI-FP (%)",
               title="B5: operational signal detection vs SNR", ylim=(-5, 105))
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_dir / "b5_tpr_vs_snr.png", dpi=150)
        plt.close()
        print(f"Saved → {args.out_dir / 'b5_tpr_vs_snr.png'}")

        # B5: AUC vs SNR
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, r in b5_res.items():
            aucs = [row[1] for row in r["rows"]]
            ax.plot(snrs, aucs, marker="o", label=name)
        ax.axhline(0.927, ls="--", color="red", alpha=0.6, label="ViT-MAE ref (0.927)")
        ax.axhline(0.5, ls=":", color="gray", alpha=0.5, label="chance")
        ax.set(xlabel="Injection SNR", ylabel="AUC (signal vs RFI)",
               title="B5: AUC(signal vs RFI) vs SNR", ylim=(0.3, 1.05))
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_dir / "b5_auc_vs_snr.png", dpi=150)
        plt.close()
        print(f"Saved → {args.out_dir / 'b5_auc_vs_snr.png'}")

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
