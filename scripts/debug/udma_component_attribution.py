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

Blocks
------
  1. BUDGET      — fraction of the fused score's magnitude each weighted term
                   contributes on normal (quiet) data. A term worth <1% cannot
                   change a ranking.
  2. REDUNDANCY  — correlation between ``st1`` and ``st2``. If ~1.0 the students
                   learned the same function, so ``ss`` -> 0 by construction.
  3. EASY        — AUC of each component alone vs the fusion, positives =
                   injected ETI, negatives = the SAME quiet sites uninjected
                   (paired, so site variability cancels). Trivial model-free
                   scorers included as the bar.
  4. OPERATIONAL — the test that actually matters: positives = injected ETI in
                   quiet sites, negatives = REAL RFI-rich sites. Against clean
                   noise (block 3) any brightness detector looks good, because
                   ``max_pixel`` fires on any hot pixel; the search's real
                   confuser is RFI, not empty sky. Energy runs AGAINST the model
                   here (a faint injected line barely raises frame energy while
                   RFI is energetic by definition), so block 4 is pessimistic —
                   which is what block 5 corrects.
  5. MATCHED-E   — block 4 with positives/negatives caliper-matched to equal
                   frame energy (adaptive caliper tightened until the
                   energy-only AUC collapses to ~0.5). This is the number
                   comparable to the historical 0.77-0.79 for the AE-vs-MemAE
                   disagreement scorer (docs/01_scoring-history.md §3),
                   and the one that says whether the model reads MORPHOLOGY or
                   just brightness.
  6. OCCUPANCY   — confound-free control: same line, same sites, same SNR,
                   differing ONLY in cadence occupancy — ON-only (blocks 0,2,4)
                   vs persistent (all 6 blocks, as terrestrial RFI is). Blocks
                   4/5 compare two different backgrounds, so energy-matching
                   neutralises energy but not background TEXTURE; this block
                   removes that confound entirely (protocol from
                   eti_vs_rfi_separation_test.py).

Read
----
  st1 alone ~= cob everywhere        -> MemAE student + memory are dead weight
  ss alone strongest at low SNR      -> the UDMA-specific term carries the regime
                                        that matters; equal fixed weights dilute it
  cob <= max_pixel/peak_snr in 4/5   -> the model is beaten by a threshold detector
  block 5 >> block 5 trivial         -> the score is morphology, not brightness

Outcome (2026-07-20): UDMA beats trivial statistics by +0.07-0.09 AUC at matched
energy vs real RFI, so the memory unit earns its place. Two caveats: clean
negatives give the OPPOSITE answer, and equal-weight fusion dilutes ``ss``,
which is by far the best single term at SNR 10. Full record in
``docs/03_teacher-localization.md`` §6.

Usage:
    PYTHONPATH=/path/to/BL-Exotica-AD python scripts/debug/udma_component_attribution.py \\
        --checkpoint outputs/training/20260707_093113_6d0d1ba/checkpoints/epoch=057-val_loss=0.2065.ckpt \\
        --model_config configs/model/udma_old_teacher.yaml \\
        --cache /path/to/data/processed/cache_gbt_fine_exotica \\
        --split val --n_sites 1200 \\
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
from scripts.debug.encode_separation_test import load_model, frame_energy, _caliper_match

INPUT_SHAPE = (96, 1024, 1)
MAP_KEYS = ("st1", "st2", "ss", "cob")
TRIVIAL_KEYS = ("max_pixel", "peak_snr", "energy")
ALL_KEYS = MAP_KEYS + TRIVIAL_KEYS


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/udma_old_teacher.yaml")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_sites", type=int, default=1200,
                   help="Sites sampled; quiet=lowest hot-frac quartile, RFI=highest")
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
        x = torch.from_numpy(np.asarray(frames[i:i + batch_size])).float().unsqueeze(1).to(device)
        comps = model.anomaly_map_components(x)
        for k in MAP_KEYS:
            out[k].append(comps[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _topk_score(maps: np.ndarray, topk_frac: float) -> np.ndarray:
    """Production aggregation (UDMA.anomaly_score method='topk') on any map."""
    flat = maps.reshape(len(maps), -1)
    k = max(1, int(round(topk_frac * flat.shape[1])))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


def _trivial_scores(frames: np.ndarray) -> dict:
    """Model-free bar (cf. scripts/debug/statistical_baseline.py)."""
    flat = frames.reshape(len(frames), -1)
    mx = flat.max(axis=1)
    sd = flat.std(axis=1)
    return {"max_pixel": mx, "peak_snr": mx / np.maximum(sd, 1e-8),
            "energy": frame_energy(frames)}


def _all_scores(model, frames, device, batch_size, topk_frac) -> dict:
    maps = _component_maps(model, frames, device, batch_size)
    scores = {k: _topk_score(maps[k], topk_frac) for k in MAP_KEYS}
    scores.update(_trivial_scores(frames))
    return scores


def _auc_row(pos: dict, neg: dict, label) -> dict:
    y = np.concatenate([np.zeros(len(neg[ALL_KEYS[0]])), np.ones(len(pos[ALL_KEYS[0]]))])
    row = {"case": label}
    for k in ALL_KEYS:
        row[k] = float(roc_auc_score(y, np.concatenate([neg[k], pos[k]])))
    return row


def _print_row(row: dict) -> None:
    print(f"  {str(row['case']):>12}   " + "   ".join(f"{k}={row[k]:.3f}" for k in ALL_KEYS))


def _inject(gen, raw_sites, snr, fchans, on_indices, preproc):
    """ON-only (or persistent) injection into each raw site -> preprocessed stack."""
    out = []
    for r in raw_sites:
        drift_rate, start_channel, f_profile, t_builder, _ = \
            gen.sample_cadence_signal_params(fchans, INPUT_SHAPE[0])
        frame, _ = gen.inject_on_only_cadence(
            r, snr=snr, drift_rate=drift_rate, start_channel=start_channel,
            f_profile=f_profile, t_profile_builder=t_builder, on_indices=on_indices,
        )
        out.append(preprocess_raw(frame, preproc))
    return np.stack(out)


def _subset(scores: dict, idx) -> dict:
    return {k: v[idx] for k, v in scores.items()}


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

    # Same quiet/RFI split as the sweep harness (encode_separation_test.py): hot-fraction quartiles.
    pre_all = np.stack([preprocess_raw(r, preproc) for r in raw_sites])
    hot_fracs = np.array([float((f > 5.0).sum()) / f.size for f in pre_all])
    quiet_idx = np.where(hot_fracs <= np.percentile(hot_fracs, 25))[0]
    rfi_idx = np.where(hot_fracs >= np.percentile(hot_fracs, 75))[0]
    print(f"  {len(raw_sites)} sites -> quiet {len(quiet_idx)}, RFI {len(rfi_idx)}")

    quiet_raw = raw_sites[quiet_idx]
    quiet_pre = pre_all[quiet_idx]
    rfi_pre = pre_all[rfi_idx]

    quiet_maps = _component_maps(model, quiet_pre, args.device, args.batch_size)
    quiet_scores = {k: _topk_score(quiet_maps[k], topk_frac) for k in MAP_KEYS}
    quiet_scores.update(_trivial_scores(quiet_pre))
    rfi_scores = _all_scores(model, rfi_pre, args.device, args.batch_size, topk_frac)

    # ---- 1. magnitude budget (normal data = quiet) ---------------------------
    print("\n=== 1. BUDGET (share of fused score magnitude, quiet data) ===")
    budget_rows = []
    denom = float(np.mean(quiet_maps["cob"]))
    for w, k in zip(weights, ("st1", "st2", "ss")):
        contrib = w * float(np.mean(quiet_maps[k]))
        budget_rows.append({"component": k, "weight": w,
                            "mean_map": float(np.mean(quiet_maps[k])),
                            "weighted_mean": contrib,
                            "share_of_cob": contrib / max(denom, 1e-12)})
        print(f"  {k:4s} w={w:.2f}  mean={budget_rows[-1]['mean_map']:.6f}  "
              f"share of cob = {budget_rows[-1]['share_of_cob']*100:6.2f}%")

    # ---- 2. student redundancy ----------------------------------------------
    corr = float(np.corrcoef(quiet_maps["st1"].ravel(), quiet_maps["st2"].ravel())[0, 1])
    ratio = float(np.mean(quiet_maps["ss"]) / max(np.mean(quiet_maps["st1"]), 1e-12))
    print("\n=== 2. REDUNDANCY (are the two students the same function?) ===")
    print(f"  corr(map_st1, map_st2) = {corr:.4f}   (->1.0 = redundant students)")
    print(f"  mean(ss)/mean(st1)     = {ratio:.4f}   (->0.0 = no disagreement left)")

    gen = NarrowbandDriftingGenerator(nb_params, seed=args.seed)
    inj_scores_by_snr, inj_pre_by_snr = {}, {}
    for snr in args.snr_list:
        inj_pre = _inject(gen, quiet_raw, snr, fchans, (0, 2, 4), preproc)
        inj_pre_by_snr[snr] = inj_pre
        inj_scores_by_snr[snr] = _all_scores(model, inj_pre, args.device, args.batch_size, topk_frac)

    # ---- 3. easy: injected vs the SAME quiet sites, uninjected ---------------
    print("\n=== 3. EASY (positives=injected ETI, negatives=same quiet sites, uninjected) ===")
    easy_rows = []
    for snr in args.snr_list:
        row = _auc_row(inj_scores_by_snr[snr], quiet_scores, f"SNR {snr:g}")
        easy_rows.append(row)
        _print_row(row)

    # ---- 4. operational: injected vs REAL RFI --------------------------------
    print("\n=== 4. OPERATIONAL (positives=injected ETI in quiet, negatives=REAL RFI) ===")
    print("  (energy runs against the model here -- see block 5)")
    oper_rows = []
    for snr in args.snr_list:
        row = _auc_row(inj_scores_by_snr[snr], rfi_scores, f"SNR {snr:g}")
        oper_rows.append(row)
        _print_row(row)

    # ---- 5. matched energy: the decisive one ---------------------------------
    print("\n=== 5. MATCHED-ENERGY (block 4, caliper-matched to equal frame energy) ===")
    matched_rows = []
    en_rfi = rfi_scores["energy"]
    for snr in args.snr_list:
        en_inj = inj_scores_by_snr[snr]["energy"]
        best = None
        for caliper in [0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
            pr, pi = _caliper_match(en_rfi, en_inj, caliper, np.random.default_rng(args.seed))
            if len(pr) < 30:
                continue
            y = np.concatenate([np.zeros(len(pr)), np.ones(len(pi))])
            ea = float(roc_auc_score(y, np.concatenate([en_rfi[pr], en_inj[pi]])))
            cand = {"caliper": caliper, "n_per_class": len(pr), "energy_only": ea,
                    "pr": pr, "pi": pi}
            if ea <= 0.58:
                best = cand
                break
            if best is None or ea < best["energy_only"]:
                best = cand
        if best is None:
            print(f"  SNR {snr:g}: too few energy-matched pairs — skipped")
            continue
        row = _auc_row(_subset(inj_scores_by_snr[snr], best["pi"]),
                       _subset(rfi_scores, best["pr"]), f"SNR {snr:g}")
        row.update({"n_per_class": best["n_per_class"], "caliper": best["caliper"]})
        matched_rows.append(row)
        _print_row(row)
        print(f"               (n={best['n_per_class']}/class, caliper={best['caliper']}, "
              f"energy_only={best['energy_only']:.3f} -> ~0.5 means matching worked)")

    # ---- 6. occupancy control: ON-only vs persistent, same line, same sites ---
    print("\n=== 6. OCCUPANCY (ON-only 3/6 vs persistent 6/6 -- same line, same sites) ===")
    occ_rows = []
    for snr in args.snr_list:
        persistent = _inject(gen, quiet_raw, snr, fchans, (0, 1, 2, 3, 4, 5), preproc)
        pers_scores = _all_scores(model, persistent, args.device, args.batch_size, topk_frac)
        row = _auc_row(inj_scores_by_snr[snr], pers_scores, f"SNR {snr:g}")
        occ_rows.append(row)
        _print_row(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    pd.DataFrame(budget_rows).to_csv(args.out_dir / f"budget_{stem}.csv", index=False)
    pd.DataFrame([{"corr_st1_st2": corr, "ss_over_st1": ratio,
                   "n_quiet": len(quiet_idx), "n_rfi": len(rfi_idx)}]).to_csv(
        args.out_dir / f"redundancy_{stem}.csv", index=False)
    for name, rows in (("easy", easy_rows), ("operational", oper_rows),
                       ("matched_energy", matched_rows), ("occupancy", occ_rows)):
        if rows:
            pd.DataFrame(rows).to_csv(args.out_dir / f"{name}_{stem}.csv", index=False)
    print(f"\nSaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
