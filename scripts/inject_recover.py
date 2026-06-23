"""
Injection-recovery test on real cadences (Phase 2).

Injects ON-only narrowband signals at varying SNR into clean frequency
windows of a real cadence, scores with both recon and cadence methods,
and compares against the RFI-inclusive background distribution from a
prior inference run.

The key question: at what SNR does an ON-only injection rank above the
real RFI candidates? This is the operationally relevant metric —
not the clean-baseline σ (already measured by cadence_snr_sweep.py).

Requires a prior inference run to have produced inference_scores.csv.

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/inject_recover.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --cadence_list data/processed/inject_recovery_cadences.txt \
        --inference_csv outputs/inference/srt_T1/inference_scores.csv \
        --out_dir outputs/inject_recovery/T1
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import build_autoencoder
from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs
from scripts.debug.injection_vs_rfi_test import inject_narrowband_on_only

INPUT_SHAPE = (96, 1024, 1)
METHODS = ["recon", "cadence"]
MAD_SCALE = 1.4826


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def robust_stats(scores):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return median, mad * MAD_SCALE


def score_snippet(model, snippet, method, device):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        s = model.anomaly_score(x, method=method)
    return float(s.item())


def reconstruct_snippet(model, snippet, device):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze().cpu().numpy()


def preprocess_raw_window(obs_arrays, f_start, fchans, preproc, tchans=96):
    """Slice a (tchans, fchans) window from loaded obs arrays and preprocess."""
    frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
    stacked = np.concatenate(frames, axis=0)[:tchans, :]
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    stacked = core_transform(stacked, mad_epsilon)
    return stacked


def inject_into_obs(obs_arrays, f_start, fchans, snr, drift_rate, seed, tchans_per_obs=16):
    """Extract raw window from obs arrays, inject ON-only signal, return raw snippet."""
    raw = np.stack([obs[:tchans_per_obs, f_start:f_start + fchans] for obs in obs_arrays])
    return inject_narrowband_on_only(raw, snr=snr, drift_rate=drift_rate, seed=seed)


def preprocess_injected(raw_snippet, preproc, tchans=96):
    """Preprocess an injected raw snippet (n_obs, tchans_per_obs, fchans)."""
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    frame = np.concatenate(raw_snippet, axis=0)[:tchans, :]
    frame = bandpass_correct(frame, method=method, poly_degree=poly_degree)
    frame = core_transform(frame, mad_epsilon)
    return frame.astype(np.float32)


def load_background(csv_path):
    """Load score distribution from a prior inference run."""
    recon_scores = []
    cadence_scores = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            recon_scores.append(float(row["recon_score"]))
            cadence_scores.append(float(row["cadence_score"]))
    return np.array(recon_scores), np.array(cadence_scores)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--inference_csv", type=Path, required=True,
                   help="CSV from a prior inference run (background distribution)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/inject_recovery")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=3)
    p.add_argument("--n_injections", type=int, default=20,
                   help="Injections per SNR level per cadence")
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[3, 5, 7, 10, 15, 20, 30, 50])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    # Load background distribution from inference run
    print(f"Loading background from {args.inference_csv}")
    bg_recon, bg_cadence = load_background(args.inference_csv)
    bg = {}
    for name, scores in [("recon", bg_recon), ("cadence", bg_cadence)]:
        median, mad_sigma = robust_stats(scores)
        bg[name] = {"median": median, "mad_sigma": mad_sigma, "scores": scores,
                    "thresh_3s": median + 3 * mad_sigma,
                    "thresh_5s": median + 5 * mad_sigma}
        n_3s = (scores > bg[name]["thresh_3s"]).sum()
        print(f"  {name}: median={median:.4f}  MAD_s={mad_sigma:.4f}  "
              f"3s={bg[name]['thresh_3s']:.4f} ({n_3s} real candidates)")

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect results across all cadences
    all_results = {m: {snr: [] for snr in args.snr_list} for m in METHODS}

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]
        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx} ({len(obs_paths)} obs)")
        print(f"{'='*70}")

        obs_arrays = []
        try:
            for obs_path in obs_paths:
                arr = _load_full_obs(obs_path, downsample_factor)
                obs_arrays.append(arr)
        except OSError as e:
            print(f"  SKIPPING — corrupt file: {e}")
            del obs_arrays
            continue
        nchans = obs_arrays[0].shape[1]
        print(f"  Loaded {len(obs_arrays)} obs, nchans={nchans}")

        # Find clean windows: score a random sample, keep the quietest
        n_probe = min(200, (nchans - fchans) // fchans)
        probe_fstarts = rng.choice((nchans - fchans), size=n_probe, replace=False)
        probe_scores = []
        for fs in probe_fstarts:
            snip = preprocess_raw_window(obs_arrays, fs, fchans, preproc)
            s = score_snippet(model, snip, "recon", args.device)
            probe_scores.append(s)
        probe_scores = np.array(probe_scores)

        # Use the quietest 50% as injection sites
        quiet_mask = probe_scores <= np.median(probe_scores)
        quiet_fstarts = probe_fstarts[quiet_mask]
        print(f"  Probed {n_probe} windows, {quiet_mask.sum()} quiet (score <= {np.median(probe_scores):.4f})")

        injection_fstarts = rng.choice(quiet_fstarts,
                                        size=min(args.n_injections, len(quiet_fstarts)),
                                        replace=False)

        for snr in args.snr_list:
            recon_scores = []
            cadence_scores = []

            for j, fs in enumerate(injection_fstarts):
                raw_inj = inject_into_obs(obs_arrays, fs, fchans,
                                           snr=snr, drift_rate=args.drift_rate,
                                           seed=args.seed + cad_idx * 1000 + j)
                snip_inj = preprocess_injected(raw_inj, preproc)

                r = score_snippet(model, snip_inj, "recon", args.device)
                c = score_snippet(model, snip_inj, "cadence", args.device)
                recon_scores.append(r)
                cadence_scores.append(c)

            recon_scores = np.array(recon_scores)
            cadence_scores = np.array(cadence_scores)
            all_results["recon"][snr].extend(recon_scores.tolist())
            all_results["cadence"][snr].extend(cadence_scores.tolist())

        del obs_arrays
        print(f"  Done cadence {cad_idx}")

    # ---- Analysis against RFI-inclusive background ----
    print(f"\n{'='*70}")
    print(f"INJECTION RECOVERY vs RFI-INCLUSIVE BACKGROUND")
    print(f"{'='*70}")

    csv_rows = []

    for method in METHODS:
        b = bg[method]
        median, mad_sigma = b["median"], b["mad_sigma"]
        thresh_3 = b["thresh_3s"]
        thresh_5 = b["thresh_5s"]
        n_bg_candidates = (b["scores"] > thresh_3).sum()

        print(f"\n  {method} (bg: median={median:.4f}, MAD_s={mad_sigma:.4f}, "
              f"3s={thresh_3:.4f}, {n_bg_candidates} RFI candidates)")
        print(f"  {'SNR':>5s}  {'mean':>8s}  {'std':>8s}  {'sigma':>8s}  "
              f"{'det@3s':>8s}  {'det@5s':>8s}  {'rank%':>8s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        for snr in args.snr_list:
            scores = np.array(all_results[method][snr])
            mean_s = scores.mean()
            std_s = scores.std()
            sigma = (mean_s - median) / mad_sigma if mad_sigma > 0 else 0
            det_3 = (scores > thresh_3).mean() * 100
            det_5 = (scores > thresh_5).mean() * 100

            # Rank: what percentile of the background does the mean injection score fall at?
            rank_pct = (b["scores"] < mean_s).mean() * 100

            print(f"  {snr:5.0f}  {mean_s:8.4f}  {std_s:8.4f}  {sigma:8.2f}s  "
                  f"{det_3:7.1f}%  {det_5:7.1f}%  {rank_pct:7.2f}%")

            csv_rows.append({
                "method": method, "snr": snr,
                "mean_score": mean_s, "std_score": std_s,
                "sigma": sigma, "det_3s": det_3, "det_5s": det_5,
                "rank_pct": rank_pct,
            })

    # Save CSV
    csv_path = args.out_dir / "inject_recovery_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved -> {csv_path}")

    # ---- Plot 1: Detection rate vs SNR (head-to-head) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"recon": "steelblue", "cadence": "crimson"}
    snrs = sorted(args.snr_list)

    # Left: score vs SNR
    for method in METHODS:
        b = bg[method]
        means = [np.mean(all_results[method][s]) for s in snrs]
        stds = [np.std(all_results[method][s]) for s in snrs]
        axes[0].errorbar(snrs, means, yerr=stds, marker="o", color=colors[method],
                         capsize=3, label=f"{method} inject-ON")
        axes[0].axhline(b["thresh_3s"], ls="--", color=colors[method], alpha=0.4,
                        label=f"{method} 3s = {b['thresh_3s']:.3f}")

    axes[0].set_xlabel("Injection SNR")
    axes[0].set_ylabel("Anomaly score")
    axes[0].set_title("Injected score vs RFI-inclusive 3s threshold")
    axes[0].legend(fontsize=7)

    # Right: detection rate
    for method in METHODS:
        b = bg[method]
        for n_sigma, ls in [(3, "-"), (5, "--")]:
            thresh = b["median"] + n_sigma * b["mad_sigma"]
            rates = [(np.array(all_results[method][s]) > thresh).mean() * 100 for s in snrs]
            axes[1].plot(snrs, rates, ls, color=colors[method], marker="o",
                         label=f"{method} @ {n_sigma}s", markersize=4)

    axes[1].set_xlabel("Injection SNR")
    axes[1].set_ylabel("Detection rate (%)")
    axes[1].set_title("Recovery rate vs SNR (RFI-inclusive threshold)")
    axes[1].set_ylim(-5, 105)
    axes[1].legend(fontsize=7)

    plt.suptitle(f"Injection Recovery — {sum(len(all_results['recon'][s]) for s in snrs)} "
                 f"injections across {len(cadence_lines)} cadences", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inject_recovery_detection.png", dpi=150)
    plt.close()
    print(f"Saved -> {args.out_dir / 'inject_recovery_detection.png'}")

    # ---- Plot 2: Injection scores overlaid on background distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, method in zip(axes, METHODS):
        b = bg[method]
        clipped = b["scores"][b["scores"] < np.percentile(b["scores"], 99.5)]
        ax.hist(clipped, bins=200, alpha=0.5, color="gray", label="Background (real)",
                edgecolor="none")

        for snr in [5, 10, 20, 50]:
            if snr in all_results[method]:
                scores = all_results[method][snr]
                ax.axvline(np.mean(scores), ls="-", lw=1.5,
                           label=f"SNR={snr} (mean={np.mean(scores):.3f})")

        ax.axvline(b["thresh_3s"], color="orange", ls="--", lw=1, label="3s threshold")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.set_title(f"{method}")
        ax.legend(fontsize=7)

    plt.suptitle("Injected scores vs background distribution", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inject_vs_background.png", dpi=150)
    plt.close()
    print(f"Saved -> {args.out_dir / 'inject_vs_background.png'}")

    # ---- Plot 3: Example candidate plots for select SNR levels ----
    print("\nGenerating example injection plots...")
    example_dir = args.out_dir / "examples"
    example_dir.mkdir(exist_ok=True)

    # Reload first cadence for examples
    first_paths = [Path(p) for p in cadence_lines[0]]
    obs_arrays = [_load_full_obs(p, downsample_factor) for p in first_paths]
    nchans = obs_arrays[0].shape[1]

    probe_fstarts = rng.choice((nchans - fchans), size=50, replace=False)
    probe_scores = [score_snippet(model,
                    preprocess_raw_window(obs_arrays, fs, fchans, preproc),
                    "recon", args.device) for fs in probe_fstarts]
    quiet_fs = probe_fstarts[np.argmin(probe_scores)]

    for snr in [5, 10, 20]:
        raw_inj = inject_into_obs(obs_arrays, quiet_fs, fchans,
                                   snr=snr, drift_rate=args.drift_rate,
                                   seed=args.seed + 9999)
        snip_inj = preprocess_injected(raw_inj, preproc)
        snip_clean = preprocess_raw_window(obs_arrays, quiet_fs, fchans, preproc)
        recon_arr = reconstruct_snippet(model, snip_inj, args.device)

        r_score = score_snippet(model, snip_inj, "recon", args.device)
        c_score = score_snippet(model, snip_inj, "cadence", args.device)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        vmin, vmax = np.percentile(snip_clean, [1, 99])

        # Top row: clean
        axes[0, 0].imshow(snip_clean, aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap="viridis")
        axes[0, 0].set_title("Clean original")
        axes[0, 0].set_ylabel("Time bin")

        # Bottom row: injected
        axes[1, 0].imshow(snip_inj, aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1, 0].set_title(f"Injected SNR={snr}")
        axes[1, 0].set_ylabel("Time bin")

        axes[1, 1].imshow(recon_arr, aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1, 1].set_title("Reconstruction")

        error = np.abs(snip_inj - recon_arr)
        axes[1, 2].imshow(error, aspect="auto", origin="lower", cmap="hot")
        axes[1, 2].set_title("Residual")

        diff = np.abs(snip_inj - snip_clean)
        axes[0, 1].imshow(diff, aspect="auto", origin="lower", cmap="hot")
        axes[0, 1].set_title("Injected - Clean (ground truth)")

        axes[0, 2].axis("off")
        axes[0, 2].text(0.1, 0.7, f"SNR = {snr}\n"
                        f"recon score = {r_score:.4f}\n"
                        f"cadence score = {c_score:.4f}\n\n"
                        f"recon 3s threshold = {bg['recon']['thresh_3s']:.4f}\n"
                        f"cadence 3s threshold = {bg['cadence']['thresh_3s']:.4f}\n\n"
                        f"recon detected: {'YES' if r_score > bg['recon']['thresh_3s'] else 'NO'}\n"
                        f"cadence detected: {'YES' if c_score > bg['cadence']['thresh_3s'] else 'NO'}",
                        transform=axes[0, 2].transAxes, fontsize=12,
                        verticalalignment="top", fontfamily="monospace")

        f_mhz = quiet_fs * df / 1e6
        fig.suptitle(f"Injection example — SNR={snr}, f_start={quiet_fs} (~{f_mhz:.3f} MHz)",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(example_dir / f"injection_snr{snr:02d}.png", dpi=120, bbox_inches="tight")
        plt.close()

    del obs_arrays
    print(f"Saved example plots -> {example_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
