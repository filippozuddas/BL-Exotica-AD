"""ON-row vs OFF-row anomaly-map diagnostic (UDMA, real held-out cadences).

Motivation: SRT cadences repeat the SAME OFF target 3x per cadence; the real
Exotica GBT batch uses 3 DIFFERENT OFF targets per cadence (see memory
exotica_0000_dist_analysis). UDMA's teacher token grid has shape (nh=6, nw=64)
where each grid ROW corresponds 1:1 to one of the 6 stacked observations
(ON1, OFF1, ON2, OFF2, ON3, OFF3) — patch height == one block, so no patch
ever mixes content from two blocks (see udma_full_design_spec, "lucky 16x/16px
alignment"). This lets us cleanly attribute anomaly-map activation to ON rows
(0,2,4 -- always the same target, familiar) vs OFF rows (1,3,5 -- three
different sky pointings in GBT, unlike SRT).

Hypothesis: the frozen teacher (trained on SRT, where OFF rows resemble each
other because they're the same target) may have learned an implicit "OFF rows
look like each other" prior that GBT's 3-different-OFF-targets structure
violates -- producing elevated background disagreement specifically on OFF
rows, inflating the false-positive rate independently of any real anomaly.

For a large pool of real background probe windows (no injection), this script:
  1. Computes the per-position anomaly map (map_cob, plus st1/st2/ss) via
     UDMA.anomaly_map_components.
  2. Splits each map's 6 rows into ON rows {0,2,4} and OFF rows {1,3,5}.
  3. Compares the row-mean scores between the two groups, both over ALL probed
     windows and over the subset flagged as "candidates" (frame score above
     the pooled 3-sigma threshold, i.e. what inject_recover.py's background
     step calls an RFI candidate) -- if OFF rows dominate among candidates,
     the heterogeneity hypothesis is supported.

Usage (server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \\
    python scripts/debug/off_row_heterogeneity_test.py \\
        --checkpoint outputs/training/20260707_093113_6d0d1ba/checkpoints/last.ckpt \\
        --cadence_list data/raw/gbt_0000_heldout_cadences.txt \\
        --model_config configs/model/udma.yaml \\
        --max_cadences 40 \\
        --out_dir outputs/sweeps/off_row_heterogeneity
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
from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs

INPUT_SHAPE = (96, 1024, 1)
MAD_SCALE = 1.4826
ON_ROWS = (0, 2, 4)
OFF_ROWS = (1, 3, 5)


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    if not hasattr(model, "anomaly_map_components"):
        raise SystemExit(f"Model {type(model).__name__} has no anomaly_map_components() "
                         f"-- this diagnostic is UDMA-specific.")
    return model


def robust_stats(scores):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return median, mad * MAD_SCALE


def preprocess_raw_window(obs_arrays, f_start, fchans, preproc, tchans=96):
    frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
    stacked = np.concatenate(frames, axis=0)[:tchans, :]
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    stacked = core_transform(stacked, mad_epsilon)
    return stacked


@torch.no_grad()
def batched_maps(model, snippets: np.ndarray, device: str, batch: int = 64):
    """(N, 96, 1024) preprocessed -> dict of (N, 6, 64) maps for st1/st2/ss/cob."""
    out = {"st1": [], "st2": [], "ss": [], "cob": []}
    for i in range(0, len(snippets), batch):
        x = torch.from_numpy(snippets[i:i + batch]).float().unsqueeze(1).to(device)
        maps = model.anomaly_map_components(x)
        for k in out:
            out[k].append(maps[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/udma.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/off_row_heterogeneity")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=40)
    p.add_argument("--n_background_probe", type=int, default=2000,
                   help="Random windows per cadence probed for the background pool.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
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

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    all_frame_scores = []   # (N,) mean-over-grid score per window, RFI-inclusive
    all_row_scores = {k: [] for k in ("st1", "st2", "ss", "cob")}  # (N, 6) per window

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]
        print(f"Cadence {cad_idx} ({len(obs_paths)} obs)")
        try:
            obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]
        except OSError as e:
            print(f"  SKIPPING -- corrupt file: {e}")
            continue
        nchans = obs_arrays[0].shape[1]

        n_probe = min(args.n_background_probe, nchans - fchans)
        probe_fstarts = rng.choice((nchans - fchans), size=n_probe, replace=False)
        snippets = np.stack([
            preprocess_raw_window(obs_arrays, fs, fchans, preproc) for fs in probe_fstarts
        ]).astype(np.float32)
        del obs_arrays

        maps = batched_maps(model, snippets, args.device, batch=args.batch)
        for k, v in maps.items():
            row_mean = v.mean(axis=2)  # (N, 6) -- mean over the 64 freq columns
            all_row_scores[k].append(row_mean)
        all_frame_scores.append(maps["cob"].mean(axis=(1, 2)))  # (N,) == 'recon' method
        print(f"  Probed {n_probe} windows")

    frame_scores = np.concatenate(all_frame_scores)
    row_scores = {k: np.concatenate(v, axis=0) for k, v in all_row_scores.items()}  # (N_total, 6)

    median, mad_sigma = robust_stats(frame_scores)
    thresh_3s = median + 3 * mad_sigma
    is_candidate = frame_scores > thresh_3s
    print(f"\nBackground: n={len(frame_scores)}  median={median:.4f}  MAD_s={mad_sigma:.4f}  "
          f"3s={thresh_3s:.4f}  ({is_candidate.sum()} candidates)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nON-ROW (0,2,4) vs OFF-ROW (1,3,5) comparison\n{'='*70}")
    summary = {}
    for k, rs in row_scores.items():
        on_mean = rs[:, ON_ROWS].mean(axis=1)   # (N,)
        off_mean = rs[:, OFF_ROWS].mean(axis=1)  # (N,)

        print(f"\n[{k}] -- ALL windows (n={len(rs)})")
        print(f"  ON  rows mean: {on_mean.mean():.5f} +/- {on_mean.std():.5f}")
        print(f"  OFF rows mean: {off_mean.mean():.5f} +/- {off_mean.std():.5f}")
        d_all = cohens_d(off_mean, on_mean)
        frac_off_gt_all = float((off_mean > on_mean).mean() * 100)
        print(f"  Cohen's d (OFF vs ON): {d_all:+.3f}   "
              f"frac(OFF>ON) = {frac_off_gt_all:.1f}%")

        if is_candidate.sum() >= 5:
            on_c, off_c = on_mean[is_candidate], off_mean[is_candidate]
            print(f"  -- CANDIDATE windows only (score > 3s, n={is_candidate.sum()})")
            print(f"  ON  rows mean: {on_c.mean():.5f} +/- {on_c.std():.5f}")
            print(f"  OFF rows mean: {off_c.mean():.5f} +/- {off_c.std():.5f}")
            d_cand = cohens_d(off_c, on_c)
            frac_off_gt_cand = float((off_c > on_c).mean() * 100)
            print(f"  Cohen's d (OFF vs ON): {d_cand:+.3f}   "
                  f"frac(OFF>ON) = {frac_off_gt_cand:.1f}%")
        else:
            d_cand, frac_off_gt_cand = None, None
            print("  -- too few candidates for a separate comparison")

        summary[k] = {
            "on_mean_all": float(on_mean.mean()), "off_mean_all": float(off_mean.mean()),
            "cohens_d_all": d_all, "frac_off_gt_all": frac_off_gt_all,
            "cohens_d_candidates": d_cand, "frac_off_gt_candidates": frac_off_gt_cand,
        }

    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    d_cob_cand = summary["cob"]["cohens_d_candidates"]
    d_cob_all = summary["cob"]["cohens_d_all"]
    d_ref = d_cob_cand if d_cob_cand is not None else d_cob_all
    if d_ref > 0.3:
        print(f"  cob Cohen's d = {d_ref:+.3f} (OFF > ON) -- supports the heterogeneity "
              "hypothesis: background disagreement concentrates on OFF rows (3 different "
              "targets per cadence), not ON rows (same target x3). Consistent with a "
              "teacher prior (learned on SRT's single-OFF-target convention) mismatched "
              "to GBT's OFF-target diversity.")
    elif d_ref < -0.3:
        print(f"  cob Cohen's d = {d_ref:+.3f} (ON > OFF) -- opposite of the hypothesis; "
              "background disagreement concentrates on ON rows instead. Re-examine.")
    else:
        print(f"  cob Cohen's d = {d_ref:+.3f} -- no clear ON/OFF asymmetry. The "
              "heterogeneity hypothesis is NOT supported by this test; look elsewhere "
              "(pooled-threshold effect, teacher feature geometry, etc.).")

    # Save + plot
    np.savez(args.out_dir / "off_row_heterogeneity_results.npz",
              frame_scores=frame_scores, is_candidate=is_candidate,
              **{f"row_scores_{k}": v for k, v in row_scores.items()})

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, k in zip(axes, ("st1", "st2", "ss", "cob")):
        rs = row_scores[k]
        row_means = rs.mean(axis=0)  # (6,)
        colors = ["#4C72B0" if i in ON_ROWS else "#DD8452" for i in range(6)]
        ax.bar(range(6), row_means, color=colors)
        ax.set_xticks(range(6))
        ax.set_xticklabels(["ON1", "OFF1", "ON2", "OFF2", "ON3", "OFF3"], fontsize=8)
        ax.set_title(f"{k} (d={summary[k]['cohens_d_all']:+.2f})")
    plt.tight_layout()
    plt.savefig(args.out_dir / "on_off_row_scores.png", dpi=150)
    print(f"\nSaved -> {args.out_dir / 'on_off_row_scores.png'}")
    print(f"Saved -> {args.out_dir / 'off_row_heterogeneity_results.npz'}")


if __name__ == "__main__":
    main()
