"""Occupancy scorer evaluation harness — Stadio B pass/fail decision for 0000.fil.

Implements ``docs/2026-07-02_occupancy_scorer_plan.md`` section 3. Every
frame-level scoring line on 0000.fil is closed or capped (recon at chance on
the one axis that IS separable: cadence occupancy). This harness judges
``OccupancyScorer`` (src/search/vetting/occupancy.py) against the
PRE-REGISTERED bars in the plan's section 1 — not renegotiable after seeing
the numbers:

  - PRIMARY (operational): TPR >= 80% at FP <= 10% (threshold = 90th
    percentile of scores on REAL negatives), SNR=10, split val.
  - SECONDARY (controlled): AUC >= 0.75 on the ETI-vs-RFI-control matched-
    energy readout (same test as ``eti_vs_rfi_separation_test.py``, scored
    with the occupancy statistic instead of recon/embedding) — directly
    comparable to the earlier 0.631 (supervised embedding) / 0.497 (recon).

Two arms, same populations, same seed (plan section 2.3):
  - raw      : preprocessed frame, no model.
  - residual : (x - MemAE(x))^2, pinned checkpoint (see --checkpoint default).

Populations (plan section 3.1):
  - Negatives: >=2000 real val frames AS-IS (no hot-frac filter) — includes
    quiet, persistent-RFI and intermittent-RFI with their natural statistics.
  - Positives: ETI ON-only injected into 500 quiet-quartile + 500 RFI-quartile
    sites, one frozen morphology per site, swept over snr_list.
  - Controlled secondary population: 150 quiet sites, ETI (ON-only) vs
    RFI-control (persistent), same frozen morphology per site (``inject_both``
    from eti_vs_rfi_separation_test.py).

Usage (server, not dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/occupancy_scorer_test.py \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \
        --checkpoint outputs/training/20260630_181920_7911978/checkpoints/epoch=017-val_loss=1.3410.ckpt \
        --out_dir outputs/sweeps/occupancy_scorer
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import NarrowbandParams, NarrowbandDriftingGenerator
from src.models.autoencoder import build_autoencoder
from src.search.vetting import OccupancyScorer
from scripts.debug.injection_vs_rfi_test import preprocess_raw
from scripts.debug.eti_vs_rfi_separation_test import inject_both
from scripts.debug.encode_separation_test import (
    frame_energy, frame_stats, morphology_matched_energy_recon,
)

INPUT_SHAPE = (96, 1024, 1)
ON_INDICES = (0, 2, 4)
OFF_INDICES = (1, 3, 5)
N_OBS = 6

# Pre-registered bars (plan section 1) — do not renegotiate post-hoc.
PRIMARY_SNR = 10.0
PRIMARY_FP = 0.10
PRIMARY_TPR_BAR = 0.80
SECONDARY_AUC_BAR = 0.75


def load_memae(checkpoint_path: Path, model_cfg: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_cfg, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def memae_residual_maps(model, maps: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(N, 96, 1024) preprocessed -> (N, 96, 1024) per-pixel squared residual."""
    out = []
    for i in range(0, len(maps), batch):
        x = torch.from_numpy(maps[i:i + batch]).float().unsqueeze(1).to(device)
        recon = model(x)
        out.append(((x - recon) ** 2).squeeze(1).cpu().numpy())
    return np.concatenate(out, axis=0)


def hot_frac(pp: np.ndarray) -> float:
    return float((pp > 5.0).sum()) / pp.size


def select_populations(raw_pool: np.ndarray, preproc: dict, rng: np.random.Generator,
                        n_negatives: int, n_quiet: int, n_rfi: int, n_controlled: int):
    """Disjoint index partition of ``raw_pool``: negatives (unfiltered, as-is),
    quiet-quartile injection sites, RFI-quartile injection sites, and a
    separate quiet-quartile block for the controlled secondary test."""
    n = len(raw_pool)
    perm = rng.permutation(n)
    neg_take = min(n_negatives, max(0, n - (n_quiet + n_rfi + n_controlled)))
    neg_idx = perm[:neg_take]
    remaining = perm[neg_take:]

    hot = np.array([hot_frac(preprocess_raw(raw_pool[i], preproc)) for i in remaining])
    lo, hi = np.percentile(hot, 25), np.percentile(hot, 75)
    quiet_pool = remaining[hot <= lo]
    rfi_pool = remaining[hot >= hi]

    quiet_inj = quiet_pool[:n_quiet]
    quiet_ctrl = quiet_pool[n_quiet:n_quiet + n_controlled]
    rfi_inj = rfi_pool[:n_rfi]
    return neg_idx, quiet_inj, quiet_ctrl, rfi_inj


def build_injected_maps(raw_pool, site_idx, nb_params, fchans, total_tchans,
                         snr_list, preproc, seed):
    """One frozen morphology per site; returns dict snr -> (preprocessed maps,
    per-site meta) and the flat list of per-site meta (drift/profile), shared
    across all SNRs for that site."""
    maps_by_snr = {snr: [] for snr in snr_list}
    meta_per_site = []
    for j, i in enumerate(site_idx):
        gen = NarrowbandDriftingGenerator(nb_params, seed=seed + j)
        drift_rate, start_channel, f_profile, t_profile_builder, meta = \
            gen.sample_cadence_signal_params(fchans, total_tchans)
        meta_per_site.append(meta)
        for snr in snr_list:
            out, _ = gen.inject_on_only_cadence(
                raw_pool[i], snr=snr, drift_rate=drift_rate, start_channel=start_channel,
                f_profile=f_profile, t_profile_builder=t_profile_builder,
                on_indices=ON_INDICES,
            )
            maps_by_snr[snr].append(preprocess_raw(out, preproc))
    maps_by_snr = {snr: np.array(v) for snr, v in maps_by_snr.items()}
    return maps_by_snr, meta_per_site


def arm_transform(maps: np.ndarray, arm: str, model, device: str) -> np.ndarray:
    if arm == "raw":
        return maps
    if arm == "residual":
        return memae_residual_maps(model, maps, device)
    raise ValueError(f"Unknown arm {arm!r}")


def roc_table(neg_scores: np.ndarray, pos_by_snr: dict, snr_list, tpr_bar: float,
              primary_snr: float) -> list:
    from sklearn.metrics import roc_auc_score

    thr1 = np.percentile(neg_scores, 99)
    thr10 = np.percentile(neg_scores, 90)
    print(f"  n_neg={len(neg_scores)}  threshold@1%FP={thr1:.4f}  threshold@10%FP={thr10:.4f}")
    print(f"  {'SNR':>5}  {'AUC':>8}  {'TPR@1%FP':>10}  {'TPR@10%FP':>10}")
    rows = []
    floor = None
    for snr in snr_list:
        pos = pos_by_snr[snr]
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_scores))])
        s = np.concatenate([pos, neg_scores])
        auc = float(roc_auc_score(y, s))
        tpr1 = float((pos > thr1).mean())
        tpr10 = float((pos > thr10).mean())
        if floor is None and tpr10 >= 0.5:
            floor = snr
        rows.append((snr, auc, tpr1, tpr10))
        flag = ""
        if snr == primary_snr:
            flag = "  <-- PRIMARY BAR" + (" PASS" if tpr10 >= tpr_bar else " FAIL")
        print(f"  {snr:5.0f}  {auc:8.3f}  {tpr1:9.1%}  {tpr10:10.1%}{flag}")
    tag = f"SNR≈{floor:.0f}" if floor else "never ≥50%"
    print(f"  -> sensitivity floor (TPR@10%FP >= 50%): {tag}")
    return rows


def controlled_secondary_auc(raw_pool, ctrl_idx, nb_params, fchans, total_tchans,
                              snr_list, preproc, seed, scorer, arm, model, device):
    eti_frames, rfi_frames = [], []
    for j, i in enumerate(ctrl_idx):
        gen = NarrowbandDriftingGenerator(nb_params, seed=seed + j)
        eti_raw, rfi_raw, _ = inject_both(gen, raw_pool[i], fchans, total_tchans, snr_list, seed_offset=j)
        for snr in snr_list:
            eti_frames.append(preprocess_raw(eti_raw[snr], preproc))
            rfi_frames.append(preprocess_raw(rfi_raw[snr], preproc))
    eti_frames = np.array(eti_frames)
    rfi_frames = np.array(rfi_frames)

    en_eti, en_rfi = frame_energy(eti_frames), frame_energy(rfi_frames)
    st_eti, st_rfi = frame_stats(eti_frames), frame_stats(rfi_frames)

    eti_scored = arm_transform(eti_frames, arm, model, device)
    rfi_scored = arm_transform(rfi_frames, arm, model, device)
    occ_eti, _ = scorer.score_frames(eti_scored)
    occ_rfi, _ = scorer.score_frames(rfi_scored)

    result = morphology_matched_energy_recon(occ_eti, en_eti, st_eti, occ_rfi, en_rfi, st_rfi, seed=seed)
    return result


def top_negative_diagnostics(neg_maps, neg_scores, info, top_k):
    order = np.argsort(neg_scores)[::-1][:top_k]
    return {
        "top_maps": neg_maps[order],
        "top_scores": neg_scores[order],
        "top_start_channel": info.start_channel[order],
        "top_drift_chans": info.drift_chans[order],
        "top_obs_means": info.obs_means[order],
        "top_site_idx": order,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/memae.yaml")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "outputs/training/20260630_181920_7911978/checkpoints/"
                                  "epoch=017-val_loss=1.3410.ckpt",
                   help="Pinned MemAE checkpoint for the residual arm (arm B).")
    p.add_argument("--arms", nargs="+", default=["raw", "residual"], choices=["raw", "residual"])
    p.add_argument("--n_negatives", type=int, default=2000)
    p.add_argument("--n_sites_quiet", type=int, default=500)
    p.add_argument("--n_sites_rfi", type=int, default=500)
    p.add_argument("--n_sites_controlled", type=int, default=150)
    p.add_argument("--n_pool", type=int, default=40000,
                   help="Raw snippets drawn from cache before partitioning into disjoint populations.")
    p.add_argument("--snr_list", type=float, nargs="+", default=[3, 5, 7, 10, 15, 20, 30])
    p.add_argument("--top_k_negatives", type=int, default=20)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/occupancy_scorer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boxcar_width", type=int, default=3)
    p.add_argument("--drift_step", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    fchans = data_cfg["frame"]["fchans"]
    total_tchans = INPUT_SHAPE[0]
    tchans_per_obs = total_tchans // N_OBS
    nb_params = NarrowbandParams.from_config(data_cfg)

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    pool_idx = rng.choice(arr.shape[0], size=min(args.n_pool, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[pool_idx])
    del arr
    print(f"  Pool: {len(raw_pool)} raw snippets")

    neg_idx, quiet_inj_idx, quiet_ctrl_idx, rfi_inj_idx = select_populations(
        raw_pool, preproc, rng, args.n_negatives, args.n_sites_quiet,
        args.n_sites_rfi, args.n_sites_controlled)
    print(f"  Negatives: {len(neg_idx)}  quiet-inject: {len(quiet_inj_idx)}  "
          f"RFI-inject: {len(rfi_inj_idx)}  controlled-quiet: {len(quiet_ctrl_idx)}")
    for name, need, got in [("negatives", args.n_negatives, len(neg_idx)),
                             ("quiet-inject", args.n_sites_quiet, len(quiet_inj_idx)),
                             ("RFI-inject", args.n_sites_rfi, len(rfi_inj_idx)),
                             ("controlled-quiet", args.n_sites_controlled, len(quiet_ctrl_idx))]:
        if got < need:
            print(f"  WARNING: {name} short ({got} < {need}) — raise --n_pool.")

    print("Preprocessing negatives...")
    neg_maps = np.array([preprocess_raw(raw_pool[i], preproc) for i in neg_idx])

    print(f"Injecting ETI ON-only into {len(quiet_inj_idx)} quiet + {len(rfi_inj_idx)} "
          f"RFI-rich sites, SNRs={args.snr_list}...")
    quiet_maps_by_snr, quiet_meta = build_injected_maps(
        raw_pool, quiet_inj_idx, nb_params, fchans, total_tchans, args.snr_list, preproc, args.seed)
    rfi_maps_by_snr, rfi_meta = build_injected_maps(
        raw_pool, rfi_inj_idx, nb_params, fchans, total_tchans, args.snr_list, preproc, args.seed + 100000)

    model = None
    if "residual" in args.arms:
        print(f"Loading MemAE checkpoint: {args.checkpoint}")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        model = load_memae(args.checkpoint, model_cfg, args.device)

    scorer = OccupancyScorer(fchans=fchans, tchans_per_obs=tchans_per_obs, n_obs=N_OBS,
                              on_indices=ON_INDICES, off_indices=OFF_INDICES,
                              boxcar_width=args.boxcar_width, drift_step=args.drift_step,
                              device=args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        print(f"\n{'='*72}\nARM: {arm}\n{'='*72}")

        neg_scored = arm_transform(neg_maps, arm, model, args.device)
        neg_scores, neg_info = scorer.score_frames(neg_scored)

        pos_pooled = {}
        pos_quiet = {}
        pos_rfi = {}
        for snr in args.snr_list:
            pos_quiet[snr], _ = scorer.score_frames(
                arm_transform(quiet_maps_by_snr[snr], arm, model, args.device))
            pos_rfi[snr], _ = scorer.score_frames(
                arm_transform(rfi_maps_by_snr[snr], arm, model, args.device))
            # Reuse instead of a third (redundant) score_frames/MemAE-forward pass.
            pos_pooled[snr] = np.concatenate([pos_quiet[snr], pos_rfi[snr]])

        print("\n-- Primary readout: pooled (quiet + RFI-rich sites) --")
        rows_pooled = roc_table(neg_scores, pos_pooled, args.snr_list, PRIMARY_TPR_BAR, PRIMARY_SNR)
        print("\n-- Stratified: quiet-background sites only --")
        roc_table(neg_scores, pos_quiet, args.snr_list, PRIMARY_TPR_BAR, PRIMARY_SNR)
        print("\n-- Stratified: RFI-rich-background sites only --")
        roc_table(neg_scores, pos_rfi, args.snr_list, PRIMARY_TPR_BAR, PRIMARY_SNR)

        # Stratify by t_profile and |drift| at the pre-registered SNR (plan section 3.2).
        # Reuses pos_quiet/pos_rfi already scored above — no extra scoring pass.
        thr10 = np.percentile(neg_scores, 90)
        if PRIMARY_SNR not in pos_quiet:
            print(f"\n  (skipping t_profile/|drift| stratification — SNR={PRIMARY_SNR:.0f} "
                  f"not in --snr_list)")
        else:
            for meta_list, label, scores_at_bar in [(quiet_meta, "quiet", pos_quiet[PRIMARY_SNR]),
                                                      (rfi_meta, "rfi", pos_rfi[PRIMARY_SNR])]:
                t_profiles = np.array([m["t_profile"] for m in meta_list])
                drifts = np.abs(np.array([m["drift_rate"] for m in meta_list]))
                print(f"\n-- {label} @ SNR={PRIMARY_SNR:.0f}: stratified by t_profile / |drift| --")
                for tp in np.unique(t_profiles):
                    sel = t_profiles == tp
                    if sel.sum() > 0:
                        print(f"  t_profile={tp:<14} n={sel.sum():4d}  TPR@10%FP={float((scores_at_bar[sel] > thr10).mean()):.1%}")
                med_drift = np.median(drifts) if len(drifts) else 0.0
                for lab, sel in [("|drift|<=median", drifts <= med_drift), ("|drift|>median", drifts > med_drift)]:
                    if sel.sum() > 0:
                        print(f"  {lab:<18} n={sel.sum():4d}  TPR@10%FP={float((scores_at_bar[sel] > thr10).mean()):.1%}")

        print("\n-- Secondary readout: controlled ETI-vs-RFI-control, matched energy --")
        ctrl = controlled_secondary_auc(raw_pool, quiet_ctrl_idx, nb_params, fchans, total_tchans,
                                         args.snr_list, preproc, args.seed, scorer, arm, model, args.device)
        if ctrl is not None and "error" in ctrl:
            print(f"  SKIPPED: {ctrl['error']}")
            controlled_auc = float("nan")
        else:
            print(f"  matched pairs: n/class={ctrl['n_per_class']}  caliper={ctrl['caliper']:.3f}")
            print(f"  AUC energy-only: {ctrl['energy_only']:.3f}   AUC trivial: {ctrl['trivial']:.3f}")
            print(f"  AUC occupancy  : {ctrl['recon']:.3f}  (bar >= {SECONDARY_AUC_BAR})"
                  f"  {'PASS' if ctrl['recon'] >= SECONDARY_AUC_BAR else 'FAIL'}")
            controlled_auc = ctrl["recon"]

        primary_row = next((r for r in rows_pooled if r[0] == PRIMARY_SNR), None)
        primary_tpr10 = primary_row[3] if primary_row else float("nan")
        primary_pass = primary_row is not None and primary_tpr10 >= PRIMARY_TPR_BAR
        secondary_pass = not np.isnan(controlled_auc) and controlled_auc >= SECONDARY_AUC_BAR
        print(f"\n{'='*72}\nDECISION [{arm}]: primary(TPR@10%FP@SNR{PRIMARY_SNR:.0f}>={PRIMARY_TPR_BAR:.0%})="
              f"{'PASS' if primary_pass else 'FAIL'} ({primary_tpr10:.1%})   "
              f"secondary(AUC>={SECONDARY_AUC_BAR})={'PASS' if secondary_pass else 'FAIL'} "
              f"({controlled_auc:.3f})\n{'='*72}")

        print(f"\nSaving top-{args.top_k_negatives} highest-scoring negatives for manual triage...")
        diag = top_negative_diagnostics(neg_maps, neg_scores, neg_info, args.top_k_negatives)

        # Parametrized by arm + checkpoint (plan section 1 discipline) so a rerun
        # against a different pinned checkpoint doesn't silently overwrite results.
        tag = arm if arm == "raw" else f"{arm}_{args.checkpoint.stem}".replace("=", "").replace(".", "p")
        out_npz = args.out_dir / f"occupancy_{tag}.npz"
        np.savez(
            out_npz,
            snr_list=np.array(args.snr_list),
            neg_scores=neg_scores,
            pos_pooled=np.stack([pos_pooled[snr] for snr in args.snr_list]),
            pos_quiet=np.stack([pos_quiet[snr] for snr in args.snr_list]),
            pos_rfi=np.stack([pos_rfi[snr] for snr in args.snr_list]),
            rows_pooled=np.array(rows_pooled),
            controlled_auc=controlled_auc,
            primary_pass=primary_pass,
            secondary_pass=secondary_pass,
            **{f"diag_{k}": v for k, v in diag.items()},
        )
        print(f"Saved -> {out_npz}")


if __name__ == "__main__":
    main()
