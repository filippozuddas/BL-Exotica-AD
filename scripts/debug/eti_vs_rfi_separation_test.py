"""ETI-vs-RFI separation test — the decisive diagnostic for 0000.fil.

B3 (encode_separation_test.py, block 3) reported AUC=0.845 for "injected ETI
vs real RFI" at matched energy, taken as evidence that narrowband morphology
lives in the ViT-MAE encoder. But B3's negative class is REAL RFI-rich
background and its positive class is REAL quiet background + injected ETI —
two different backgrounds. Energy-matching neutralises total energy but not
the background TEXTURE the encoder learned from real data, so B3 may be
reading "quiet-type vs RFI-type background", not "ETI line vs RFI line".

This script removes that confound: it injects an ETI-like signal AND an
RFI-like control into the SAME quiet backgrounds, using the SAME generator,
SAME frozen morphology (drift, width, profile) and SAME SNR per site — the
ONLY thing that differs is presence pattern across the 6-observation cadence:

  - ETI-like : on_indices=(0, 2, 4)             -> ON-only (present in 3/6 blocks)
  - RFI-like : on_indices=(0, 1, 2, 3, 4, 5)    -> persistent (present in 6/6 blocks,
               independent of telescope pointing, as real terrestrial RFI is)

Both classes are pooled across SNRs, caliper-matched at equal total frame
energy, then compared with:
  - a SUPERVISED logistic probe on the ViT-MAE embedding (the upper bound —
    can the representation tell them apart AT ALL, given labels?)
  - the model's UNSUPERVISED reconstruction-MSE AUC (the deployable score)
  - optionally (give --ae_checkpoint + --memae_checkpoint), the UDMA
    student-DISAGREEMENT score ||AE(x) - MemAE(x)||^2 (Qi et al. 2024, λ3
    term; zero-training probe from two frozen checkpoints), in both frame-mean
    and top-k aggregation. In THIS controlled test both classes carry the same
    injected line and differ only in cadence occupancy (3/6 vs 6/6 blocks), so
    this asks "does student disagreement see occupancy"; the operational
    injected-ETI-vs-REAL-RFI readout for the same scorer lives in
    encode_separation_test.py --scoring disagree.

Decisive readout:
  - supervised AUC ~ 0.5  -> ETI and RFI are genuinely indistinguishable on a
    single cadence-frame once background texture is controlled -> the earlier
    B3=0.845 was a background-texture artifact -> structural ceiling on
    0000.fil confirmed -> pivot to 0001.fil is justified, not just empirically
    (MemAE failure) but architecturally (single-frame morphology cannot carry
    this distinction; only true cross-observation cadence context could, and
    that is barred by the no-ON/OFF-training-objective constraint).
  - supervised AUC high -> the information IS genuinely present in a single
    frame (drift/tilt is a real single-frame cue, see CLAUDE.md dt/df) -> B3
    was not purely an artifact -> the problem is unsupervised extraction, not
    a structural ceiling, and 0000.fil is still alive.

Usage (server, not dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/eti_vs_rfi_separation_test.py \
        --checkpoint outputs/training/<run_id>/checkpoints/<best>.ckpt \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \
        --out_dir outputs/sweeps/eti_vs_rfi

    # add the disagreement scorer (with or without --checkpoint):
    ... --ae_checkpoint outputs/training/<ae_run>/checkpoints/<best>.ckpt \
        --memae_checkpoint outputs/training/<memae_run>/checkpoints/<best>.ckpt
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
from scripts.debug.injection_vs_rfi_test import preprocess_raw
from scripts.debug.encode_separation_test import (
    DisagreementPair, load_model, embed, recon_score, frame_energy, frame_stats,
    morphology_matched_energy, morphology_matched_energy_recon,
)

INPUT_SHAPE = (96, 1024, 1)


def inject_both(gen: NarrowbandDriftingGenerator, obs_windows: np.ndarray,
                 fchans: int, total_tchans: int, snr_list, seed_offset: int):
    """One frozen signal morphology per site; ETI (ON-only) and RFI-control
    (persistent, all 6 blocks) share drift/width/profile/SNR, differing ONLY
    in which observations receive the signal. Returns two dicts snr -> raw."""
    drift_rate, start_channel, f_profile, t_profile_builder, meta = \
        gen.sample_cadence_signal_params(fchans, total_tchans)

    eti_raw, rfi_raw = {}, {}
    for snr in snr_list:
        eti_raw[snr], _ = gen.inject_on_only_cadence(
            obs_windows, snr=snr, drift_rate=drift_rate, start_channel=start_channel,
            f_profile=f_profile, t_profile_builder=t_profile_builder,
            on_indices=(0, 2, 4),
        )
        rfi_raw[snr], _ = gen.inject_on_only_cadence(
            obs_windows, snr=snr, drift_rate=drift_rate, start_channel=start_channel,
            f_profile=f_profile, t_profile_builder=t_profile_builder,
            on_indices=(0, 1, 2, 3, 4, 5),
        )
    return eti_raw, rfi_raw, meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="ViT-MAE checkpoint for the embedding-probe + recon blocks; "
                        "optional if the disagreement pair is given.")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--ae_checkpoint", type=Path, default=None,
                   help="Plain-AE checkpoint (with --memae_checkpoint) — adds the UDMA "
                        "student-disagreement scorer ||AE(x)-MemAE(x)||^2.")
    p.add_argument("--ae_config", type=Path, default=ROOT / "configs/model/convae.yaml")
    p.add_argument("--memae_checkpoint", type=Path, default=None,
                   help="MemAE checkpoint (with --ae_checkpoint).")
    p.add_argument("--memae_config", type=Path, default=ROOT / "configs/model/memae.yaml")
    p.add_argument("--topk_frac", type=float, default=0.02,
                   help="Top-k fraction for the disagreement scorer's topk aggregation.")
    p.add_argument("--n_sites", type=int, default=150,
                   help="Quiet background sites; pooled with len(snr_list) gives "
                        "n_sites*len(snr_list) samples per class before caliper matching "
                        "(sampling guardrail: keep pooled n >= 500).")
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[5, 7, 10, 15, 20, 25, 30, 40, 50])
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/eti_vs_rfi")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    fchans = data_cfg["frame"]["fchans"]
    total_tchans = INPUT_SHAPE[0]
    nb_params = NarrowbandParams.from_config(data_cfg)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    if bool(args.ae_checkpoint) != bool(args.memae_checkpoint):
        raise SystemExit("--ae_checkpoint and --memae_checkpoint must be given together.")
    if args.checkpoint is None and args.ae_checkpoint is None:
        raise SystemExit("Provide --checkpoint (embedding/recon blocks) and/or "
                         "--ae_checkpoint + --memae_checkpoint (disagreement block).")

    model = None
    if args.checkpoint is not None:
        print(f"Loading model from {args.checkpoint}")
        # UDMA has no encode() (no embedding readout, see udma.py) — the
        # supervised-probe block below is skipped for it, only the recon
        # (anomaly_score) block runs.
        require_encode = model_cfg.get("architecture") != "udma"
        model = load_model(args.checkpoint, model_cfg, args.device, require_encode=require_encode)

    pair = None
    if args.ae_checkpoint is not None:
        with open(args.ae_config) as f:
            ae_cfg = yaml.safe_load(f)
        with open(args.memae_config) as f:
            memae_cfg = yaml.safe_load(f)
        print(f"Loading AE from {args.ae_checkpoint}")
        print(f"Loading MemAE from {args.memae_checkpoint}")
        pair = DisagreementPair(
            load_model(args.ae_checkpoint, ae_cfg, args.device, require_encode=False),
            load_model(args.memae_checkpoint, memae_cfg, args.device, require_encode=False),
        )

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    idx_pool = rng.choice(arr.shape[0], size=min(args.n_sites * 3, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[idx_pool])
    del arr

    # Same quiet split convention as encode_separation_test.py (bottom hot-frac
    # quartile of a probe pool), so injection sites are free of real RFI.
    hot_fracs = np.array([
        float((preprocess_raw(raw_pool[i], preproc) > 5.0).sum()) /
        preprocess_raw(raw_pool[i], preproc).size
        for i in range(len(raw_pool))
    ])
    quiet_idx = np.where(hot_fracs <= np.percentile(hot_fracs, 25))[0]
    site_idx = quiet_idx[:args.n_sites]
    print(f"  Quiet sites: {len(site_idx)} (of {len(raw_pool)} probed)")

    print(f"\nInjecting ETI (ON-only, blocks 0,2,4) vs RFI-control (persistent, "
          f"blocks 0-5) — same generator, same frozen morphology per site, "
          f"same SNR, only presence pattern differs.")

    eti_frames, rfi_frames = [], []
    for j, i in enumerate(site_idx):
        gen = NarrowbandDriftingGenerator(nb_params, seed=args.seed + j)
        eti_raw, rfi_raw, meta = inject_both(gen, raw_pool[i], fchans, total_tchans,
                                             args.snr_list, seed_offset=j)
        for snr in args.snr_list:
            eti_frames.append(preprocess_raw(eti_raw[snr], preproc))
            rfi_frames.append(preprocess_raw(rfi_raw[snr], preproc))
    eti_frames = np.array(eti_frames)
    rfi_frames = np.array(rfi_frames)
    print(f"  Pooled: {len(eti_frames)} ETI frames, {len(rfi_frames)} RFI-control frames "
          f"({len(site_idx)} sites x {len(args.snr_list)} SNRs)")

    print("\nComputing scores (embedding/recon/disagreement), energy, trivial-stats...")
    en_eti = frame_energy(eti_frames)
    en_rfi = frame_energy(rfi_frames)
    st_eti = frame_stats(eti_frames)
    st_rfi = frame_stats(rfi_frames)
    has_encode = model is not None and hasattr(model, "encode")
    if has_encode:
        emb_eti = embed(model, eti_frames, args.device)
        emb_rfi = embed(model, rfi_frames, args.device)
    if model is not None:
        rec_eti = recon_score(model, eti_frames, args.device)
        rec_rfi = recon_score(model, rfi_frames, args.device)
    if pair is not None:
        dis_eti = recon_score(pair, eti_frames, args.device, method="recon")
        dis_rfi = recon_score(pair, rfi_frames, args.device, method="recon")
        dis_topk_eti = recon_score(pair, eti_frames, args.device,
                                   method="topk", topk_frac=args.topk_frac)
        dis_topk_rfi = recon_score(pair, rfi_frames, args.device,
                                   method="topk", topk_frac=args.topk_frac)

    m = mr = None
    if model is None:
        print("\n(no --checkpoint: skipping the supervised-probe and recon blocks)")
        # fall through to the disagreement block
    elif not has_encode:
        print(f"\n(model has no encode() — e.g. UDMA — skipping the supervised-probe block, "
              f"only the recon/anomaly_score block below runs)")
    if has_encode:
        print(f"\n{'='*72}\nSUPERVISED PROBE (primary) — ETI (ON-only) vs RFI-control (persistent), "
              f"SAME quiet backgrounds, matched energy\n{'='*72}")
        m = morphology_matched_energy(emb_eti, en_eti, st_eti, emb_rfi, en_rfi, st_rfi, seed=args.seed)
    if m is not None and "error" in m:
        print(f"  SKIPPED: {m['error']}")
    elif m is not None:
        print(f"  matched pairs    : n/class = {m['n_per_class']}  (caliper = {m['caliper']:.3f})")
        print(f"  AUC energy-only  : {m['energy_only']:.3f}   (sanity: ~0.5 means matching worked)")
        print(f"  AUC trivial-stats: {m['trivial']:.3f}   (peakiness/kurtosis/top-pixel)")
        print(f"  AUC embedding    : {m['embedding']:.3f}   <-- the decisive number")
        if m["n_per_class"] < 25:
            print("  WARNING: few matched samples — AUC underpowered; treat as indicative only.")
        if m["embedding"] <= 0.58:
            print("  VERDICT: ETI and RFI-control are NOT separable in the embedding once "
                  "background texture is controlled -> B3=0.845 was (at least largely) a "
                  "background-texture artifact, not ETI-vs-RFI morphology -> structural "
                  "ceiling on 0000.fil confirmed -> pivot to 0001.fil is architecturally "
                  "justified, not just empirically.")
        elif m["embedding"] < 0.70:
            print("  VERDICT: WEAK/borderline separation -> some single-frame information "
                  "survives (likely drift/tilt) but is far below B3=0.845 -> 0000.fil is not "
                  "fully dead, but the ceiling is much lower than B3 suggested.")
        else:
            print("  VERDICT: ETI and RFI-control DO separate at matched energy, same "
                  "background -> the information is genuinely present in a single frame -> "
                  "B3 was not purely an artifact -> the problem is unsupervised extraction, "
                  "not a structural ceiling; 0000.fil is still alive.")

    if model is not None:
        print(f"\n{'='*72}\nUNSUPERVISED RECON AUC (secondary/confirmatory) — same matched pairs, "
              f"ViT-MAE reconstruction-MSE\n{'='*72}")
        mr = morphology_matched_energy_recon(rec_eti, en_eti, st_eti, rec_rfi, en_rfi, st_rfi, seed=args.seed)
        if mr is not None and "error" in mr:
            print(f"  SKIPPED: {mr['error']}")
        elif mr is not None:
            print(f"  matched pairs    : n/class = {mr['n_per_class']}  (caliper = {mr['caliper']:.3f})")
            print(f"  AUC energy-only  : {mr['energy_only']:.3f}")
            print(f"  AUC trivial-stats: {mr['trivial']:.3f}")
            print(f"  AUC recon        : {mr['recon']:.3f}")

    dis_aucs = {}
    if pair is not None:
        print(f"\n{'='*72}\nSTUDENT-DISAGREEMENT AUC (UDMA λ3 probe) — ||AE(x) − MemAE(x)||², "
              f"same matched-energy harness\n{'='*72}")
        print("  Both classes carry the SAME injected line, differing only in occupancy\n"
              "  (3/6 vs 6/6 cadence blocks) -> this reads 'does student disagreement see\n"
              "  occupancy'. AUC > 0.5: ON-only scores higher; a consistent AUC < 0.5 is\n"
              "  ALSO separability (disagreement tracks total line extent instead). The\n"
              "  operational injected-ETI-vs-REAL-RFI readout is in\n"
              "  encode_separation_test.py --scoring disagree.")
        for agg_name, d_eti, d_rfi in [
            ("mean", dis_eti, dis_rfi),
            (f"topk{args.topk_frac}", dis_topk_eti, dis_topk_rfi),
        ]:
            md = morphology_matched_energy_recon(d_eti, en_eti, st_eti,
                                                 d_rfi, en_rfi, st_rfi, seed=args.seed)
            print(f"\n  [{agg_name}]")
            if md is not None and "error" in md:
                print(f"    SKIPPED: {md['error']}")
            elif md is not None:
                print(f"    matched pairs    : n/class = {md['n_per_class']}  (caliper = {md['caliper']:.3f})")
                print(f"    AUC energy-only  : {md['energy_only']:.3f}")
                print(f"    AUC trivial-stats: {md['trivial']:.3f}")
                print(f"    AUC disagreement : {md['recon']:.3f}")
                dis_aucs[agg_name] = md["recon"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    if args.checkpoint is not None:
        parts.append(f"{args.model_config.stem}_{args.checkpoint.stem}")
    if pair is not None:
        parts.append(f"dis_{args.ae_checkpoint.stem}_{args.memae_checkpoint.stem}")
    tag = "__".join(parts).replace("=", "").replace(".", "p")
    out_npz = args.out_dir / f"eti_vs_rfi_{tag}.npz"
    out = {"en_eti": en_eti, "en_rfi": en_rfi}
    if model is not None:
        out.update(
            rec_eti=rec_eti, rec_rfi=rec_rfi,
            supervised_auc=m.get("embedding", np.nan) if m and "error" not in m else np.nan,
            recon_auc=mr.get("recon", np.nan) if mr and "error" not in mr else np.nan)
        if has_encode:
            out.update(emb_eti=emb_eti, emb_rfi=emb_rfi)
    if pair is not None:
        out.update(
            dis_eti=dis_eti, dis_rfi=dis_rfi,
            dis_topk_eti=dis_topk_eti, dis_topk_rfi=dis_topk_rfi,
            disagree_auc_mean=dis_aucs.get("mean", np.nan),
            disagree_auc_topk=dis_aucs.get(f"topk{args.topk_frac}", np.nan))
    np.savez(out_npz, **out)
    print(f"\nSaved → {out_npz}")


if __name__ == "__main__":
    main()
