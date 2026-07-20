"""Is the UDMA architecture actually earning its complexity?

UDMA's fused score is a weighted sum of three disagreement maps
(``src/models/udma.py: anomaly_map_components``)::

    map_cob = w1*map_st1 + w2*map_st2 + w3*map_ss
              \\_______/   \\_______/   \\______/
              teacher vs   teacher vs   student-student
              AE student   MemAE stud.  disagreement

Only ``map_ss`` requires *two* students; only ``map_st2``/``map_ss`` require the
memory unit at all. If ``map_st1`` alone detects as well as the fusion, UDMA
reduces to a plain student-teacher detector (STPM/US) and everything the memory
adds is decorative — a real, checkable claim, not a matter of taste.

This script answers three questions on ONE pass over real cached snippets, with
synthetic ON-only injections as the positive class (same injector as
``inject_recover.py`` / ``udma_anomaly_maps.py``):

  1. BUDGET      — what fraction of the fused score's magnitude each weighted
                   term contributes. A term worth <1% cannot change a ranking.
  2. REDUNDANCY  — correlation between ``st1`` and ``st2`` maps. If ~1.0 the two
                   students learned the same function, so ``ss`` -> 0 by
                   construction and the pair is one student wearing two hats.
  3. ATTRIBUTION — per-SNR ROC-AUC of each component used *alone* as the scorer
                   (same topk aggregation as production), against the fused
                   score, against trivial model-free baselines (max pixel, peak
                   SNR — the bar from ``statistical_baseline.py``). This is the
                   decisive one: a component that matches the fusion makes the
                   rest of the architecture unnecessary; a fusion that fails to
                   beat ``max_pixel`` makes the whole model unnecessary.

Paired design: the clean and injected sets are the SAME sites, so site-to-site
variability (which dominates between-cadence variance, see memory
``udma_heldout_injection_recovery``) cancels instead of inflating the spread.

Usage (server):
    PYTHONPATH=/content/filippo/BL-Exotica-AD python scripts/debug/udma_component_attribution.py \\
        --checkpoint outputs/training/20260707_093113_6d0d1ba/checkpoints/epoch=057-val_loss=0.2065.ckpt \\
        --model_config configs/model/udma_old_teacher.yaml \\
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine_exotica \\
        --split val --n_sites 400 \\
        --out_dir outputs/sweeps/udma_component_attribution
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import NarrowbandParams, NarrowbandDriftingGenerator
from scripts.debug.injection_vs_rfi_test import preprocess_raw
from scripts.debug.encode_separation_test import load_model

INPUT_SHAPE = (96, 1024, 1)
MAP_KEYS = ("st1", "st2", "ss", "cob")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/udma_old_teacher.yaml")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_sites", type=int, default=400,
                   help="Real snippets used as the clean class (and as injection hosts)")
    p.add_argument("--snr_list", type=float, nargs="+", default=[10.0, 15.0, 20.0, 30.0])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/udma_component_attribution")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def _component_maps(model, frames: np.ndarray, device: str, batch_size: int) -> dict:
    """Stack the four maps for a set of preprocessed frames -> {key: (N, nh, nw)}."""
    out = {k: [] for k in MAP_KEYS}
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        x = torch.from_numpy(np.asarray(chunk)).float().unsqueeze(1).to(device)  # (B,1,H,W)
        comps = model.anomaly_map_components(x)
        for k in MAP_KEYS:
            out[k].append(comps[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _topk_score(maps: np.ndarray, topk_frac: float) -> np.ndarray:
    """Production aggregation (UDMA.anomaly_score method='topk') on any map."""
    flat = maps.reshape(len(maps), -1)
    k = max(1, int(round(topk_frac * flat.shape[1])))
    part = np.partition(flat, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def _trivial_scores(frames: np.ndarray) -> dict:
    """Model-free bar (cf. scripts/debug/statistical_baseline.py)."""
    flat = frames.reshape(len(frames), -1)
    mx = flat.max(axis=1)
    sd = flat.std(axis=1)
    return {"max_pixel": mx, "peak_snr": mx / np.maximum(sd, 1e-8)}


def main():
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    fchans = data_cfg["frame"]["fchans"]
    nb_params = NarrowbandParams.from_config(data_cfg)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    if model_cfg.get("architecture") != "udma":
        raise SystemExit(f"--model_config must be architecture: udma, got '{model_cfg.get('architecture')}'.")
    scoring = model_cfg.get("scoring", {})
    weights = tuple(scoring.get("weights", (0.5, 0.5, 0.5)))
    topk_frac = float(scoring.get("topk_frac", 0.01))
    print(f"score_weights={weights}  topk_frac={topk_frac}")

    model = load_model(args.checkpoint, model_cfg, args.device, require_encode=False)

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    idx = rng.choice(arr.shape[0], size=min(args.n_sites, arr.shape[0]), replace=False)
    raw_sites = np.array(arr[idx])
    del arr
    print(f"  {len(raw_sites)} sites from split '{args.split}'")

    clean = np.stack([preprocess_raw(r, preproc) for r in raw_sites])
    clean_maps = _component_maps(model, clean, args.device, args.batch_size)

    # ---- 1. magnitude budget -------------------------------------------------
    print("\n=== 1. BUDGET (share of fused score magnitude, clean data) ===")
    budget_rows = []
    denom = float(np.mean(clean_maps["cob"]))
    for w, k in zip(weights, ("st1", "st2", "ss")):
        contrib = w * float(np.mean(clean_maps[k]))
        budget_rows.append({"component": k, "weight": w,
                            "mean_map": float(np.mean(clean_maps[k])),
                            "weighted_mean": contrib,
                            "share_of_cob": contrib / max(denom, 1e-12)})
        print(f"  {k:4s} w={w:.2f}  mean={budget_rows[-1]['mean_map']:.6f}  "
              f"share of cob = {budget_rows[-1]['share_of_cob']*100:6.2f}%")

    # ---- 2. student redundancy ----------------------------------------------
    a = clean_maps["st1"].ravel()
    b = clean_maps["st2"].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    ratio = float(np.mean(clean_maps["ss"]) / max(np.mean(clean_maps["st1"]), 1e-12))
    print("\n=== 2. REDUNDANCY (are the two students the same function?) ===")
    print(f"  corr(map_st1, map_st2) = {corr:.4f}   (->1.0 = redundant students)")
    print(f"  mean(ss)/mean(st1)     = {ratio:.4f}   (->0.0 = no disagreement left)")

    # ---- 3. discriminative attribution --------------------------------------
    print("\n=== 3. ATTRIBUTION (AUC of each component alone vs fusion vs trivial) ===")
    gen = NarrowbandDriftingGenerator(nb_params, seed=args.seed)
    scorers = list(MAP_KEYS) + ["max_pixel", "peak_snr"]
    clean_scores = {k: _topk_score(clean_maps[k], topk_frac) for k in MAP_KEYS}
    clean_scores.update(_trivial_scores(clean))

    rows = []
    for snr in args.snr_list:
        inj_raw = []
        for r in raw_sites:
            drift_rate, start_channel, f_profile, t_builder, _ = \
                gen.sample_cadence_signal_params(fchans, INPUT_SHAPE[0])
            frame, _ = gen.inject_on_only_cadence(
                r, snr=snr, drift_rate=drift_rate, start_channel=start_channel,
                f_profile=f_profile, t_profile_builder=t_builder, on_indices=(0, 2, 4),
            )
            inj_raw.append(frame)
        inj = np.stack([preprocess_raw(r, preproc) for r in inj_raw])
        inj_maps = _component_maps(model, inj, args.device, args.batch_size)
        inj_scores = {k: _topk_score(inj_maps[k], topk_frac) for k in MAP_KEYS}
        inj_scores.update(_trivial_scores(inj))

        y = np.concatenate([np.zeros(len(clean)), np.ones(len(inj))])
        line = {"snr": snr}
        for k in scorers:
            line[k] = float(roc_auc_score(y, np.concatenate([clean_scores[k], inj_scores[k]])))
        rows.append(line)
        print("  SNR {:>5.1f}   ".format(snr)
              + "   ".join(f"{k}={line[k]:.3f}" for k in scorers))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    pd.DataFrame(budget_rows).to_csv(args.out_dir / f"budget_{stem}.csv", index=False)
    pd.DataFrame([{"corr_st1_st2": corr, "ss_over_st1": ratio}]).to_csv(
        args.out_dir / f"redundancy_{stem}.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / f"attribution_{stem}.csv", index=False)
    print(f"\nSaved -> {args.out_dir}")

    print("\n=== READ ===")
    print("  st1 alone ~= cob at every SNR   -> MemAE student + memory are dead weight (UDMA -> STPM)")
    print("  ss alone strong                 -> the UDMA-specific term is the one doing the work")
    print("  cob <= max_pixel / peak_snr     -> the whole model is beaten by a threshold detector")


if __name__ == "__main__":
    main()
