"""
Diagnose the two symptoms reported on the first real-GBT-exotica inference run:

1. A cadence with a large ON/OFF anomaly-map contrast but no actual signal
   (e.g. MESSIER42) — check whether this is a real instrumental ON/OFF power
   step that survives the current *global* (concat-then-normalize) bandpass
   correction / core_transform, by comparing against a *per-observation*
   normalized version of the same snippet (concat happens after each obs is
   independently bandpass-corrected + log/MAD standardized — the behaviour
   documented in configs/data/gbt_fine.yaml and the CachedDataset docstring,
   but not what src/data/torch_dataset.py / scripts/inference.py actually do).

2. A pure-noise cadence where a quiet frame still crosses the per-cadence
   3-sigma threshold — check whether that's a real per-cadence MAD too tight
   rather than a normalization artifact (score cadence-wide, report where the
   target f_start's score falls relative to median/MAD).

Does NOT change any pipeline code — this only lays the two normalizations
side by side and reports numbers so a retrain decision can be made with
evidence instead of speculation. Run on the data host (needs the real .h5
cadence files in RAM).

Usage:
    python scripts/debug/normalization_diagnostic.py \
        --checkpoint outputs/<run>/checkpoints/best.ckpt \
        --model_config configs/model/udma.yaml \
        --data_config configs/data/gbt_fine.yaml \
        --obs_paths <6 space-separated .h5 paths for the cadence> \
        --f_start 23123456 \
        --out_dir outputs/inference/<run>/diagnostics \
        --method topk
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs
import inference as inf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True)
    p.add_argument("--data_config", type=Path, required=True)
    p.add_argument("--obs_paths", nargs=6, required=True)
    p.add_argument("--f_start", type=int, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--method", default="topk")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_cadence_scan", action="store_true",
                    help="Skip the full cadence sliding-window scan (part 2) "
                         "and only run the normalization comparison (part 1).")
    return p.parse_args()


def per_obs_normalized(obs_frames, method, poly_degree, mad_epsilon):
    """The *intended* behaviour: bandpass_correct + core_transform applied to
    each observation independently, then concatenated — as documented in
    configs/data/gbt_fine.yaml and the CachedDataset docstring."""
    normed = []
    for frame in obs_frames:
        f = bandpass_correct(frame, method=method, poly_degree=poly_degree)
        f = core_transform(f, mad_epsilon)
        normed.append(f)
    return np.concatenate(normed, axis=0)


def main():
    args = parse_args()
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans, tchans = frame["fchans"], frame["tchans"]
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]
    bp_method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_shape = (tchans, fchans, 1)
    model = inf.load_model(args.checkpoint, model_cfg, input_shape, args.device)

    obs_paths = [Path(p) for p in args.obs_paths]
    print(f"Loading {len(obs_paths)} observations...")
    obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]

    # ---- Part 1: raw per-observation level check + normalization comparison ----
    print(f"\n{'='*70}\nPart 1: per-observation raw level check at f_start={args.f_start}\n{'='*70}")
    obs_frames = [arr[:, args.f_start:args.f_start + fchans] for arr in obs_arrays]
    raw_medians = [float(np.median(f)) for f in obs_frames]
    raw_mads = [float(np.median(np.abs(f - m))) for f, m in zip(obs_frames, raw_medians)]
    labels = ["ON1", "OFF1", "ON2", "OFF2", "ON3", "OFF3"]
    for lbl, m, mad in zip(labels, raw_medians, raw_mads):
        print(f"  {lbl}: raw median={m:.4f}  raw MAD={mad:.4f}")
    level_spread = max(raw_medians) - min(raw_medians)
    mean_mad = float(np.mean(raw_mads))
    print(f"  Level spread across obs (max-min median) = {level_spread:.4f}  "
          f"(mean per-obs MAD = {mean_mad:.4f}, ratio = {level_spread / max(mean_mad, 1e-9):.2f}x)")
    print("  -> ratio >> 1 means a real instrumental power step between "
          "observations, not just noise fluctuation.")

    # Current (buggy) pipeline: concat raw frames first, normalize once globally.
    file_specs = []
    scratch = args.out_dir / "_scratch"
    scratch.mkdir(exist_ok=True)
    for i, arr in enumerate(obs_arrays):
        path = scratch / f"obs{i}.f32"
        arr.tofile(str(path))
        file_specs.append((str(path), arr.shape, arr.dtype))
    inf._worker_init(file_specs, fchans, tchans, preproc)
    snippet_global = inf._preprocess_at(args.f_start)

    # Proposed fix: normalize each observation independently, then concat.
    snippet_perobs = per_obs_normalized(obs_frames, bp_method, poly_degree, mad_epsilon)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, snip, title in [(axes[0], snippet_global, "Current: global norm (concat-then-normalize)"),
                             (axes[1], snippet_perobs, "Proposed fix: per-obs norm (normalize-then-concat)")]:
        vmin, vmax = np.percentile(snip, [1, 99])
        im = ax.imshow(snip, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Freq channel")
        ax.set_ylabel("Time bin")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"f_start={args.f_start}  f~{args.f_start * df / 1e6:.4f} MHz")
    out_path = args.out_dir / f"norm_compare_f{args.f_start}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved comparison plot -> {out_path}")

    # Quantify striping: block-mean stddev is a proxy for surviving level steps.
    def block_striping(snip):
        n_obs = len(obs_frames)
        block_h = snip.shape[0] // n_obs
        block_means = [snip[i * block_h:(i + 1) * block_h].mean() for i in range(n_obs)]
        return float(np.std(block_means))

    striping_global = block_striping(snippet_global)
    striping_perobs = block_striping(snippet_perobs)
    print(f"  Block-mean std (striping proxy): global={striping_global:.4f}  "
          f"per-obs={striping_perobs:.4f}  "
          f"(per-obs should be much smaller if the fix removes the artifact)")

    if args.skip_cadence_scan:
        return

    # ---- Part 2: full cadence scan — is the target score really above threshold? ----
    print(f"\n{'='*70}\nPart 2: cadence-wide {args.method} score distribution\n{'='*70}")
    stride = frame["stride_infer"]
    nchans = obs_arrays[0].shape[1]
    n_snippets = max(0, (nchans - fchans) // stride + 1)
    f_starts = [i * stride for i in range(n_snippets)]
    print(f"  Scoring {n_snippets} snippets (stride={stride})...")

    scores = []
    batch_snippets, batch_fstarts = [], []
    batch_size = 64
    target_score = None
    for i, fs in enumerate(f_starts):
        snip = inf._preprocess_at(fs)
        batch_snippets.append(snip)
        batch_fstarts.append(fs)
        if len(batch_snippets) == batch_size or i == len(f_starts) - 1:
            s = inf.score_batch(model, batch_snippets, args.method, args.device)
            scores.extend(s.tolist())
            for fs_b, sc in zip(batch_fstarts, s):
                if fs_b == args.f_start:
                    target_score = float(sc)
            batch_snippets, batch_fstarts = [], []

    scores = np.array(scores)
    median, mad_sigma = inf.robust_stats(scores)
    thresh_3 = median + 3 * mad_sigma
    thresh_5 = median + 5 * mad_sigma
    print(f"  median={median:.4f}  MAD_sigma={mad_sigma:.4f}  "
          f"thresh_3s={thresh_3:.4f}  thresh_5s={thresh_5:.4f}")
    if target_score is not None:
        n_sigma = (target_score - median) / mad_sigma if mad_sigma > 0 else float("inf")
        print(f"  Target f_start={args.f_start} score={target_score:.4f}  "
              f"-> {n_sigma:.2f} sigma above median "
              f"({'ABOVE' if target_score > thresh_3 else 'below'} 3-sigma threshold)")
    else:
        print(f"  WARNING: f_start={args.f_start} not on the stride grid "
              f"({stride}) — no exact score match; use one of: "
              f"{f_starts[max(0, args.f_start // stride - 1)]}, "
              f"{f_starts[min(len(f_starts)-1, args.f_start // stride + 1)]}")
    print(f"  -> a tiny mad_sigma relative to typical scores means the "
          f"per-cadence threshold is intrinsically tight on quiet cadences, "
          f"independent of the normalization bug in Part 1.")

    for spec in file_specs:
        Path(spec[0]).unlink()


if __name__ == "__main__":
    main()
