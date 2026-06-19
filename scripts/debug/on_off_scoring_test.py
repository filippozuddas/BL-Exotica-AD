"""
ON/OFF cadence-aware scoring test (Zhang et al. 2019 approach).

Exploits the ABACAD cadence: ETI appears only in ON obs (0,2,4), RFI in all 6.
The ViT-MAE's patch grid (6 rows × 64 cols) maps 1:1 to observations.

Inference: ON visible (obs 0,2,4) → predict OFF (obs 1,3,5).
  - ETI case: model sees signal in ON, propagates it into OFF prediction,
    but real OFF has NO signal → mismatch → high error on OFF prediction
  - RFI case: model sees RFI in ON, propagates into OFF prediction,
    real OFF also has RFI → match → low error
  - Quiet: no signal anywhere → predictions match → low error

The anomaly score is the error on the predicted OFF observations compared
to the real OFF observations, measured on the signal's frequency track.

Runs an SNR sweep to characterize detection sensitivity.

Usage:
    python scripts/debug/on_off_scoring_test.py \
        --checkpoint outputs/.../best_model.ckpt \
        --cache /path/to/cache_gbt_fine.npz \
        --model_config configs/model/vit_mae.yaml \
        --out_dir outputs/on_off_test
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

from src.data.preprocessing import bandpass_correct, core_transform
from src.models.autoencoder import build_autoencoder
from src.models.vit_mae import patchify, unpatchify


def load_model(checkpoint_path, model_config, device="cpu"):
    model = build_autoencoder((96, 1024, 1), model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def preprocess_raw(raw_snippet, cfg_preproc):
    method = cfg_preproc.get("bandpass_method", "polynomial")
    poly_degree = cfg_preproc.get("poly_degree", 3)
    mad_epsilon = cfg_preproc.get("mad_epsilon", 1e-6)
    frame = np.concatenate(raw_snippet, axis=0)
    frame = bandpass_correct(frame, method=method, poly_degree=poly_degree)
    frame = core_transform(frame, mad_epsilon)
    return frame.astype(np.float32)


def inject_narrowband_on_only(raw_snippet, snr=25.0, drift_rate=0.3, seed=42):
    rng = np.random.default_rng(seed)
    raw = raw_snippet.copy()
    n_obs, tchans_per_obs, fchans = raw.shape
    noise_std = np.median(np.abs(raw - np.median(raw))) * 1.4826
    signal_amplitude = snr * noise_std
    start_chan = rng.integers(100, fchans - 100)
    width_chans = 3.0
    dt = 18.25361108
    df = 2.7939677238464355
    drift_chans_per_bin = (drift_rate * dt) / df
    for obs_idx in [0, 2, 4]:
        global_t_start = obs_idx * tchans_per_obs
        for t in range(tchans_per_obs):
            center = start_chan + (global_t_start + t) * drift_chans_per_bin
            chans = np.arange(fchans)
            profile = signal_amplitude * np.exp(-0.5 * ((chans - center) / width_chans) ** 2)
            raw[obs_idx, t] += profile
    return raw, start_chan


def on_to_off_inference(model, x):
    """ON visible (obs 0,2,4) → predict OFF (obs 1,3,5).

    Returns the predicted OFF observations as a full (B,C,H,W) image.
    Only OFF rows (1,3,5) in the output are meaningful predictions;
    ON rows are filled from a reverse pass for completeness.
    """
    b = x.shape[0]
    nh, nw = model.grid_size
    n = model.num_patches
    device = x.device

    # ON rows visible → predict OFF
    on_patches = []
    for row in [0, 2, 4]:
        on_patches.append(torch.arange(row * nw, (row + 1) * nw, device=device))
    ids_keep_on = torch.cat(on_patches).unsqueeze(0).expand(b, -1)

    off_patches = []
    for row in [1, 3, 5]:
        off_patches.append(torch.arange(row * nw, (row + 1) * nw, device=device))
    ids_mask_off = torch.cat(off_patches)

    ids_shuffle = torch.cat([ids_keep_on[0], ids_mask_off])
    ids_restore = ids_shuffle.argsort().unsqueeze(0).expand(b, -1)

    pred_on2off = model._decode_from_keep(x, ids_keep_on, ids_restore)

    # Reverse: OFF visible → predict ON (for display completeness)
    ids_keep_off = torch.cat(off_patches).unsqueeze(0).expand(b, -1)
    ids_shuffle2 = torch.cat([ids_keep_off[0], torch.cat(on_patches)])
    ids_restore2 = ids_shuffle2.argsort().unsqueeze(0).expand(b, -1)
    pred_off2on = model._decode_from_keep(x, ids_keep_off, ids_restore2)

    # Combine: ON→OFF prediction for OFF rows, OFF→ON for ON rows
    combined = torch.zeros(b, n, model.patch_dim, device=device, dtype=x.dtype)
    for row in [1, 3, 5]:
        s, e = row * nw, (row + 1) * nw
        combined[:, s:e] = pred_on2off[:, s:e]
    for row in [0, 2, 4]:
        s, e = row * nw, (row + 1) * nw
        combined[:, s:e] = pred_off2on[:, s:e]

    return unpatchify(combined, model.patch_size, (b, *model.input_shape))


def compute_track_error_per_obs(snippet, recon, start_chan, drift_rate, width=10):
    """Compute MSE along the signal's frequency track in each observation."""
    tchans, fchans = snippet.shape
    tchans_per_obs = 16
    n_obs = tchans // tchans_per_obs
    dt = 18.25361108
    df = 2.7939677238464355
    drift_chans_per_bin = (drift_rate * dt) / df

    obs_track_errors = []
    obs_bg_errors = []
    for obs in range(n_obs):
        t_start = obs * tchans_per_obs
        track_pixels = []
        bg_pixels = []
        for t in range(tchans_per_obs):
            global_t = t_start + t
            center = int(start_chan + global_t * drift_chans_per_bin)
            lo = max(0, center - width)
            hi = min(fchans, center + width + 1)
            if hi > lo:
                err = (snippet[global_t, lo:hi] - recon[global_t, lo:hi]) ** 2
                track_pixels.extend(err.tolist())

            # Background: offset by 200 channels (safe margin)
            track_width = hi - lo
            bg_center = center + 200
            bg_lo = max(0, bg_center - width)
            bg_hi = min(fchans, bg_center + width + 1)
            if bg_hi > bg_lo and bg_lo >= hi:
                bg_err = (snippet[global_t, bg_lo:bg_hi] - recon[global_t, bg_lo:bg_hi]) ** 2
                bg_pixels.extend(bg_err.tolist())

        obs_track_errors.append(np.mean(track_pixels) if track_pixels else 0.0)
        obs_bg_errors.append(np.mean(bg_pixels) if bg_pixels else 0.0)

    return np.array(obs_track_errors), np.array(obs_bg_errors)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True)
    p.add_argument("--data_config", type=Path,
                   default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[5, 10, 15, 20, 25, 30, 40, 50])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/on_off_test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)
    preproc = cfg["preprocessing"]

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    print(f"Loading NPZ: {args.cache}")
    archive = np.load(str(args.cache), mmap_mode="r")
    arr = archive[args.split]
    n_total = arr.shape[0]

    n_scan = min(200, n_total)
    scan_idx = rng.choice(n_total, size=n_scan, replace=False)
    scan_raw = np.array(arr[scan_idx])
    del arr, archive

    hot_fracs = []
    preprocessed = []
    for i in range(n_scan):
        snip = preprocess_raw(scan_raw[i], preproc)
        preprocessed.append(snip)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)
    hot_fracs = np.array(hot_fracs)
    preprocessed = np.array(preprocessed)

    quiet_order = np.argsort(hot_fracs)
    quiet_idx = quiet_order[:args.n_samples]
    rfi_idx = quiet_order[-min(args.n_samples, 15):]

    print(f"Selected {len(quiet_idx)} quiet, {len(rfi_idx)} RFI snippets")

    # --- Baseline: quiet snippets (no injection) ---
    # Measure OFF prediction error when there's no signal anywhere
    print("\nBaseline: ON→OFF on quiet snippets...")
    baseline_off_errors = []
    for i in quiet_idx:
        x = torch.from_numpy(preprocessed[i]).float().unsqueeze(0).unsqueeze(0).to(args.device)
        with torch.no_grad():
            recon = on_to_off_inference(model, x).squeeze().cpu().numpy()
        snip = preprocessed[i]
        # Error only on predicted OFF observations (the ones we care about)
        off_err = np.mean([
            np.mean((snip[16:32] - recon[16:32]) ** 2),
            np.mean((snip[48:64] - recon[48:64]) ** 2),
            np.mean((snip[80:96] - recon[80:96]) ** 2),
        ])
        baseline_off_errors.append(off_err)

    baseline_off = np.mean(baseline_off_errors)
    baseline_off_std = np.std(baseline_off_errors)
    print(f"  Baseline OFF prediction MSE: {baseline_off:.4f} ± {baseline_off_std:.4f}")

    # --- RFI snippets: ON→OFF should predict RFI in OFF (low error) ---
    print("\nRFI: ON→OFF scoring...")
    rfi_off_errors = []
    rfi_track_off = []
    for i in rfi_idx:
        x = torch.from_numpy(preprocessed[i]).float().unsqueeze(0).unsqueeze(0).to(args.device)
        with torch.no_grad():
            recon = on_to_off_inference(model, x).squeeze().cpu().numpy()
        snip = preprocessed[i]
        off_err = np.mean([
            np.mean((snip[16:32] - recon[16:32]) ** 2),
            np.mean((snip[48:64] - recon[48:64]) ** 2),
            np.mean((snip[80:96] - recon[80:96]) ** 2),
        ])
        rfi_off_errors.append(off_err)

        fake_start = 500
        track_err, _ = compute_track_error_per_obs(snip, recon, fake_start, args.drift_rate)
        rfi_track_off.append(np.mean(track_err[[1, 3, 5]]))

    rfi_off_mean = np.mean(rfi_off_errors)
    print(f"  RFI OFF prediction MSE: {rfi_off_mean:.4f} ± {np.std(rfi_off_errors):.4f}")

    # --- SNR sweep: inject ON-only, measure OFF prediction error ---
    # Key idea: signal in ON → model propagates to OFF prediction →
    # but real OFF has no signal → error on OFF track should rise with SNR
    print(f"\nSNR sweep (ON→predict OFF, measure OFF error): {args.snr_list}")
    print(f"  {'SNR':>5s}  {'OFF-track':>10s}  {'OFF-bg':>10s}  {'track/bg':>10s}  "
          f"{'OFF-full':>10s}  {'vs baseline':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    snr_results = {}
    for snr in args.snr_list:
        off_track_errors = []
        off_bg_errors = []
        off_full_errors = []

        for j, i in enumerate(quiet_idx):
            raw_inj, start_chan = inject_narrowband_on_only(
                scan_raw[i], snr=snr, drift_rate=args.drift_rate, seed=args.seed + j
            )
            snip_inj = preprocess_raw(raw_inj, preproc)

            x = torch.from_numpy(snip_inj).float().unsqueeze(0).unsqueeze(0).to(args.device)
            with torch.no_grad():
                recon = on_to_off_inference(model, x).squeeze().cpu().numpy()

            # Track error per observation
            track_err, bg_err = compute_track_error_per_obs(
                snip_inj, recon, start_chan, args.drift_rate
            )
            # We care about error on OFF observations (predicted from ON)
            off_track_errors.append(np.mean(track_err[[1, 3, 5]]))
            off_bg_errors.append(np.mean(bg_err[[1, 3, 5]]))

            # Full OFF observation error
            off_full = np.mean([
                np.mean((snip_inj[16:32] - recon[16:32]) ** 2),
                np.mean((snip_inj[48:64] - recon[48:64]) ** 2),
                np.mean((snip_inj[80:96] - recon[80:96]) ** 2),
            ])
            off_full_errors.append(off_full)

        snr_results[snr] = {
            "off_track": np.array(off_track_errors),
            "off_bg": np.array(off_bg_errors),
            "off_full": np.array(off_full_errors),
        }

        ot = np.mean(off_track_errors)
        ob = np.mean(off_bg_errors)
        of_ = np.mean(off_full_errors)
        ratio = ot / ob if ob > 0 else float("inf")
        vs_bl = of_ / baseline_off if baseline_off > 0 else float("inf")
        print(f"  {snr:5.1f}  {ot:10.4f}  {ob:10.4f}  {ratio:10.3f}x  "
              f"{of_:10.4f}  {vs_bl:10.3f}x")

    # --- Plots ---
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snrs = sorted(snr_results.keys())

    # Plot 1: SNR sweep
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ON→OFF cadence scoring (Zhang et al. 2019 approach)\n"
                 "ON visible → predict OFF; error measured on OFF predictions",
                 fontsize=11)

    # Panel 1: OFF track error vs SNR
    ax = axes[0]
    off_track_means = [snr_results[s]["off_track"].mean() for s in snrs]
    off_track_stds = [snr_results[s]["off_track"].std() for s in snrs]
    off_bg_means = [snr_results[s]["off_bg"].mean() for s in snrs]
    off_bg_stds = [snr_results[s]["off_bg"].std() for s in snrs]

    ax.errorbar(snrs, off_track_means, yerr=off_track_stds, marker="o",
                color="crimson", label="OFF track (signal propagated)", capsize=3)
    ax.errorbar(snrs, off_bg_means, yerr=off_bg_stds, marker="s",
                color="steelblue", label="OFF background", capsize=3)
    ax.axhline(np.mean(rfi_track_off), ls="--", color="orange", alpha=0.7,
               label=f"RFI OFF track ({np.mean(rfi_track_off):.2f})")
    ax.set_xlabel("Injection SNR")
    ax.set_ylabel("Track MSE on OFF predictions")
    ax.set_title("Signal track error on predicted OFF obs")
    ax.legend(fontsize=7)
    ax.set_xscale("linear")

    # Panel 2: track/background ratio vs SNR
    ax = axes[1]
    ratios = [off_track_means[i] / off_bg_means[i] if off_bg_means[i] > 0 else 0
              for i in range(len(snrs))]
    ax.plot(snrs, ratios, "o-", color="crimson", label="Injected ON-only")
    ax.axhline(1.0, ls=":", color="gray", alpha=0.5, label="No discrimination")
    ax.set_xlabel("Injection SNR")
    ax.set_ylabel("Track / Background error ratio")
    ax.set_title("Discrimination ratio on OFF predictions")
    ax.legend(fontsize=7)
    ax.set_xscale("linear")

    # Panel 3: full OFF error vs SNR
    ax = axes[2]
    off_full_means = [snr_results[s]["off_full"].mean() for s in snrs]
    off_full_stds = [snr_results[s]["off_full"].std() for s in snrs]
    ax.errorbar(snrs, off_full_means, yerr=off_full_stds, marker="o",
                color="crimson", label="Injected (full OFF MSE)", capsize=3)
    ax.axhline(baseline_off, ls="--", color="steelblue",
               label=f"Quiet baseline ({baseline_off:.3f})")
    ax.axhline(rfi_off_mean, ls="--", color="orange",
               label=f"RFI ({rfi_off_mean:.3f})")
    ax.fill_between(snrs, baseline_off - baseline_off_std,
                     baseline_off + baseline_off_std, alpha=0.15, color="steelblue")
    ax.set_xlabel("Injection SNR")
    ax.set_ylabel("Full OFF prediction MSE")
    ax.set_title("Full OFF observation error")
    ax.legend(fontsize=7)
    ax.set_xscale("linear")

    plt.tight_layout()
    plt.savefig(args.out_dir / "snr_sweep.png", dpi=150)
    plt.close()
    print(f"\nSaved → {args.out_dir / 'snr_sweep.png'}")

    # Plot 2: Example reconstructions at SNR=25
    n_ex = min(3, len(quiet_idx))
    fig, axes = plt.subplots(n_ex, 5, figsize=(25, 5 * n_ex))
    if n_ex == 1:
        axes = [axes]

    snr_example = 25.0
    for row, i in enumerate(quiet_idx[:n_ex]):
        raw_inj, start_chan = inject_narrowband_on_only(
            scan_raw[i], snr=snr_example, drift_rate=args.drift_rate, seed=args.seed + row
        )
        snip_orig = preprocessed[i]
        snip_inj = preprocess_raw(raw_inj, preproc)

        x = torch.from_numpy(snip_inj).float().unsqueeze(0).unsqueeze(0).to(args.device)
        with torch.no_grad():
            recon = on_to_off_inference(model, x).squeeze().cpu().numpy()

        error = (snip_inj - recon) ** 2
        track_err, bg_err = compute_track_error_per_obs(
            snip_inj, recon, start_chan, args.drift_rate
        )

        vmin, vmax = np.percentile(snip_orig, [1, 99])

        # Col 0: input with injection
        axes[row][0].imshow(snip_inj, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row][0].set_title("Input (ON-only injection)", fontsize=9)

        # Col 1: ON→OFF reconstruction
        axes[row][1].imshow(recon, aspect="auto", origin="upper",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row][1].set_title("ON→OFF prediction", fontsize=9)

        # Col 2: error map
        axes[row][2].imshow(error, aspect="auto", origin="upper",
                            cmap="hot", vmin=0, vmax=np.percentile(error, 99))
        axes[row][2].set_title("Squared error", fontsize=9)

        # Col 3: per-obs full MSE bar chart
        obs_errors = []
        for obs in range(6):
            s, e = obs * 16, (obs + 1) * 16
            obs_errors.append(np.mean(error[s:e]))
        colors = ["crimson" if obs in [0, 2, 4] else "steelblue" for obs in range(6)]
        labels_obs = ["ON" if obs in [0, 2, 4] else "OFF" for obs in range(6)]
        axes[row][3].bar(range(6), obs_errors, color=colors)
        axes[row][3].set_xticks(range(6))
        axes[row][3].set_xticklabels(labels_obs, fontsize=8)
        axes[row][3].set_title("MSE per observation", fontsize=9)
        axes[row][3].set_ylabel("MSE")

        # Col 4: track MSE per obs
        axes[row][4].bar(range(6), track_err,
                         color=["crimson" if obs in [0, 2, 4] else "steelblue"
                                for obs in range(6)])
        bg_vals = bg_err
        axes[row][4].bar(range(6), bg_vals, alpha=0.3,
                         color=["crimson" if obs in [0, 2, 4] else "steelblue"
                                for obs in range(6)],
                         label="background")
        axes[row][4].set_xticks(range(6))
        axes[row][4].set_xticklabels(labels_obs, fontsize=8)
        axes[row][4].set_title("Track vs background MSE", fontsize=9)
        axes[row][4].set_ylabel("Track MSE")
        if row == 0:
            axes[row][4].legend(["track", "background"], fontsize=7)

        for col in range(3):
            for obs_b in range(1, 6):
                axes[row][col].axhline(obs_b * 16 - 0.5, color="red", ls="--", lw=0.6, alpha=0.5)
            axes[row][col].set_ylabel("time bin")
            axes[row][col].set_xlabel("freq channel")

    plt.suptitle(f"ON→OFF scoring examples (SNR={snr_example})", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out_dir / "examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {args.out_dir / 'examples.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
