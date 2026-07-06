"""
ON/OFF residual-energy ratio analysis on inference candidates.

For each candidate from a prior inference run, computes:
    r = MSE(residual, ON blocks) / MSE(residual, OFF blocks)

where ON blocks = time bins [0:16, 32:48, 64:80] and OFF = [16:32, 48:64, 80:96].

Expected:
  - Persistent RFI (present in all obs): r ≈ 1 → reject
  - ON-only signal (ETI-like):           r ≫ 1 → keep

Also computes the ratio for synthetic ON-only injections at various SNR
to confirm that real signals produce r ≫ 1.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/on_off_ratio_analysis.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --inference_csv outputs/inference/srt_T1/inference_scores.csv \
        --cadence_list data/processed/inference_cadences.txt \
        --out_dir outputs/diagnostics/on_off_ratio
"""

import argparse
import csv
import sys
import time
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
from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs
from scripts.debug.injection_vs_rfi_test import inject_narrowband_on_only

INPUT_SHAPE = (96, 1024, 1)
ON_SLICES = [slice(0, 16), slice(32, 48), slice(64, 80)]
OFF_SLICES = [slice(16, 32), slice(48, 64), slice(80, 96)]


def load_model(checkpoint_path, model_config, device):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def compute_residual(model, snippet, device):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    residual = (x - recon).squeeze().cpu().numpy()
    return residual


def on_off_ratio_whole(residual):
    on_energy = np.mean([np.mean(residual[s, :] ** 2) for s in ON_SLICES])
    off_energy = np.mean([np.mean(residual[s, :] ** 2) for s in OFF_SLICES])
    return on_energy / max(off_energy, 1e-10)


def on_off_ratio_peak(residual, n_cols=5):
    """ON/OFF ratio on the top-n peak residual columns only."""
    col_energy = np.mean(residual ** 2, axis=0)
    peak_cols = np.argsort(col_energy)[-n_cols:]
    on_vals = np.concatenate([residual[s, :][:, peak_cols] for s in ON_SLICES])
    off_vals = np.concatenate([residual[s, :][:, peak_cols] for s in OFF_SLICES])
    return np.mean(on_vals ** 2) / max(np.mean(off_vals ** 2), 1e-10)


def preprocess_window(obs_arrays, f_start, fchans, preproc, tchans=96):
    frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
    stacked = np.concatenate(frames, axis=0)[:tchans, :]
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    stacked = core_transform(stacked, mad_epsilon)
    return stacked


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--inference_csv", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/diagnostics/on_off_ratio")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_candidates", type=int, default=500,
                   help="Max candidates to analyze (sorted by cadence score desc)")
    p.add_argument("--n_injections", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    fchans = data_cfg["frame"]["fchans"]
    downsample_factor = data_cfg["frame"].get("downsample_factor", 1)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    # Load candidates from inference CSV
    candidates = []
    with open(args.inference_csv) as f:
        for row in csv.DictReader(f):
            candidates.append(row)

    # Sort by cadence score descending, take top N
    candidates.sort(key=lambda r: float(r["cadence_score"]), reverse=True)
    candidates = candidates[:args.max_candidates]
    print(f"Loaded {len(candidates)} top candidates from {args.inference_csv}")
    print(f"  Score range: {float(candidates[-1]['cadence_score']):.4f} — "
          f"{float(candidates[0]['cadence_score']):.4f}")

    # Load cadence observations
    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]

    # Group candidates by cadence_idx
    cad_groups = {}
    for c in candidates:
        idx = int(c["cadence_idx"])
        cad_groups.setdefault(idx, []).append(c)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Compute ON/OFF ratio for real candidates ----
    ratios_whole = []
    ratios_peak = []
    scores = []

    for cad_idx, cand_list in sorted(cad_groups.items()):
        if cad_idx >= len(cadence_lines):
            continue
        obs_paths = [Path(p) for p in cadence_lines[cad_idx]]
        print(f"\nCadence {cad_idx}: loading {len(obs_paths)} obs for {len(cand_list)} candidates")

        try:
            obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]
        except OSError as e:
            print(f"  SKIPPING — corrupt file: {e}")
            continue

        t0 = time.time()
        for c in cand_list:
            f_start = int(c["f_start"])
            snip = preprocess_window(obs_arrays, f_start, fchans, preproc)
            residual = compute_residual(model, snip, args.device)
            r_whole = on_off_ratio_whole(residual)
            r_peak = on_off_ratio_peak(residual, n_cols=5)
            ratios_whole.append(r_whole)
            ratios_peak.append(r_peak)
            scores.append(float(c["cadence_score"]))

        elapsed = time.time() - t0
        print(f"  Scored {len(cand_list)} candidates in {elapsed:.1f}s")
        del obs_arrays

    ratios_whole = np.array(ratios_whole)
    ratios_peak = np.array(ratios_peak)
    scores = np.array(scores)

    for label, ratios in [("WHOLE-SNIPPET", ratios_whole), ("PEAK-COLUMN (top 5)", ratios_peak)]:
        print(f"\n{'='*60}")
        print(f"ON/OFF RATIO ({label}) — {len(ratios)} candidates")
        print(f"{'='*60}")
        print(f"  median r = {np.median(ratios):.3f}")
        print(f"  mean r   = {np.mean(ratios):.3f}")
        print(f"  r < 1.5: {(ratios < 1.5).sum()} ({(ratios < 1.5).mean()*100:.1f}%) — likely RFI")
        print(f"  r > 2.0: {(ratios > 2.0).sum()} ({(ratios > 2.0).mean()*100:.1f}%) — possible signal")
        print(f"  r > 3.0: {(ratios > 3.0).sum()} ({(ratios > 3.0).mean()*100:.1f}%)")

    # ---- Compute ON/OFF ratio for synthetic injections ----
    print(f"\nComputing injection ratios...")
    # Reload first cadence for injections
    first_paths = [Path(p) for p in cadence_lines[0]]
    try:
        obs_arrays = [_load_full_obs(p, downsample_factor) for p in first_paths]
    except OSError:
        print("Cannot load first cadence for injections, skipping.")
        obs_arrays = None

    inject_ratios_whole = {}
    inject_ratios_peak = {}
    if obs_arrays is not None:
        nchans = obs_arrays[0].shape[1]
        probe_fs = rng.choice(nchans - fchans, size=100, replace=False)
        probe_scores = []
        for fs in probe_fs:
            snip = preprocess_window(obs_arrays, fs, fchans, preproc)
            residual = compute_residual(model, snip, args.device)
            probe_scores.append(np.mean(residual ** 2))
        quiet_fs = probe_fs[np.argsort(probe_scores)[:args.n_injections]]

        for snr in [5, 10, 20, 50]:
            whole_rs, peak_rs = [], []
            for j, fs in enumerate(quiet_fs):
                raw = np.stack([obs[:16, fs:fs + fchans] for obs in obs_arrays])
                raw_inj = inject_narrowband_on_only(raw, snr=snr, drift_rate=0.3,
                                                     seed=args.seed + j)
                frame = np.concatenate(raw_inj, axis=0)
                frame = bandpass_correct(frame,
                                          method=preproc.get("bandpass_method", "polynomial"),
                                          poly_degree=preproc.get("poly_degree", 3))
                frame = core_transform(frame, preproc.get("mad_epsilon", 1e-6))
                residual = compute_residual(model, frame, args.device)
                whole_rs.append(on_off_ratio_whole(residual))
                peak_rs.append(on_off_ratio_peak(residual, n_cols=5))
            inject_ratios_whole[snr] = np.array(whole_rs)
            inject_ratios_peak[snr] = np.array(peak_rs)
            print(f"  SNR={snr:3d}: whole median={np.median(whole_rs):.2f}  "
                  f"peak median={np.median(peak_rs):.2f}")
        del obs_arrays

    # ---- Save CSV ----
    csv_path = args.out_dir / "on_off_ratios.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "cadence_idx", "f_start", "cadence_score",
                         "ratio_whole", "ratio_peak"])
        for i in range(len(ratios_whole)):
            c = candidates[i]
            writer.writerow(["rfi_candidate", c["cadence_idx"], c["f_start"],
                             c["cadence_score"],
                             f"{ratios_whole[i]:.4f}", f"{ratios_peak[i]:.4f}"])
        for snr in inject_ratios_whole:
            for w, p in zip(inject_ratios_whole[snr], inject_ratios_peak[snr]):
                writer.writerow([f"inject_snr{snr}", 0, 0, 0, f"{w:.4f}", f"{p:.4f}"])
    print(f"\nSaved -> {csv_path}")

    # ---- Plot: side-by-side histograms whole vs peak ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (label, rfi_r, inj_r) in zip(axes, [
        ("Whole-snippet", ratios_whole, inject_ratios_whole),
        ("Peak-column (top 5)", ratios_peak, inject_ratios_peak),
    ]):
        ax.hist(rfi_r, bins=100, alpha=0.7, color="gray", edgecolor="black",
                linewidth=0.3, label=f"RFI candidates (n={len(rfi_r)})")

        for snr, rs in sorted(inj_r.items()):
            ax.axvline(np.median(rs), ls="--", lw=2,
                       label=f"Inject SNR={snr} (r={np.median(rs):.1f})")

        ax.axvline(1.0, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("ON/OFF residual energy ratio (r)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=7)
        max_inj = max((np.median(v) for v in inj_r.values()), default=5)
        ax.set_xlim(0, max(5, max_inj * 1.3))

    plt.suptitle("ON/OFF ratio: whole-snippet vs peak-column", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out_dir / "on_off_ratio_comparison.png", dpi=150)
    plt.close()
    print(f"Saved -> {args.out_dir / 'on_off_ratio_comparison.png'}")

    # ---- Plot 2: Score vs peak ratio scatter ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ratios_peak, scores, s=3, alpha=0.5, c="gray", label="RFI candidates")

    for snr, rs in sorted(inject_ratios_peak.items()):
        ax.scatter(rs, np.full_like(rs, snr), s=20, marker="x",
                   label=f"Inject SNR={snr}")

    ax.axvline(2.0, color="red", ls="--", lw=1, label="r=2.0 (proposed cutoff)")
    ax.set_xlabel("ON/OFF peak-column ratio (r)")
    ax.set_ylabel("Cadence anomaly score")
    ax.set_title("Peak-column ON/OFF ratio: RFI vs injections")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_dir / "score_vs_peak_ratio.png", dpi=150)
    plt.close()
    print(f"Saved -> {args.out_dir / 'score_vs_peak_ratio.png'}")

    # ---- Verdict ----
    if len(inject_ratios_peak) > 0:
        rfi_med_peak = np.median(ratios_peak)
        inj5_peak = np.median(inject_ratios_peak.get(5, [0]))
        inj10_peak = np.median(inject_ratios_peak.get(10, [0]))
        print(f"\n{'='*60}")
        print(f"VERDICT (peak-column)")
        print(f"{'='*60}")
        print(f"  RFI candidates median r_peak = {rfi_med_peak:.3f}")
        print(f"  Injection SNR=5  median r_peak = {inj5_peak:.2f}")
        print(f"  Injection SNR=10 median r_peak = {inj10_peak:.2f}")
        if inj5_peak > rfi_med_peak * 1.5:
            print(f"  -> GOOD SEPARATION at SNR=5")
            reject_pct = (ratios_peak < 2.0).mean() * 100
            print(f"     r_peak < 2.0 rejects {reject_pct:.1f}% of RFI candidates")
        elif inj10_peak > rfi_med_peak * 1.5:
            print(f"  -> MODERATE: separates at SNR>=10 but not SNR=5")
        else:
            print(f"  -> POOR SEPARATION even with peak-column")

    print("\nDone.")


if __name__ == "__main__":
    main()
