"""
Re-plot short-listed candidates from an already-run inference pass as
individual high-resolution PNGs, showing the preprocessed waterfall (by
default no model needed — just extracts and plots the snippet, like
``scripts/debug/plot_single_candidate.py`` but for the whole short list at
once).

Takes the per-cadence ``{method}_candidates.csv`` written by
``scripts/inference.py`` (e.g. ``topk_candidates.csv``), filters to
``in_short_list == True`` (when that column is present — UDMA only; falls
back to all rows otherwise), and plots each with the 6-observation ABACAD
divider lines on the waterfall.

``cad_idx`` and the target name are parsed from the CSV's parent directory
name (``cad{idx}_{target}_{fch1}MHz_{date}``, as written by
``scripts/inference.py: make_cadence_dirname()``) unless overridden.

Pass ``--model_config`` plus either ``--maps_dir`` or ``--checkpoint`` to also
plot the model's anomaly map (UDMA's native ``anomaly_map`` grid, bilinearly
overlaid on the waterfall) or reconstruction (AE/MAE/VAE) next to each
candidate — otherwise the script only plots the preprocessed waterfall, no
model needed.

``--maps_dir`` (preferred for UDMA) points at the output of
``scripts/recompute_anomaly_maps.py`` — per-cadence ``maps.npz`` files with
the cached ``st1``/``st2``/``ss`` disagreement maps, keyed by ``f_start``.
The fused map is a linear recombination (``w1*st1 + w2*st2 + w3*ss``, weights
from ``model_config``'s ``scoring.weights``) computed offline — no checkpoint
load, no forward pass. Falls back to ``--checkpoint`` (live forward pass) for
any candidate not covered by the cache (e.g. below the cache's
``--map_quantile``).

Usage (cached maps, no model load):
    python scripts/debug/plot_shortlist_candidates.py \
        --candidates_csv outputs/inference/<run>/cad03_.../topk_candidates.csv \
        --cadence_list data/processed/inference_cadences.txt \
        --data_config configs/data/gbt_fine.yaml \
        --out_dir outputs/inference/<run>/cad03_.../shortlist_png \
        --maps_dir outputs/inference/exotica_heldout_maps \
        --model_config configs/model/udma.yaml

Usage (live forward pass):
    python scripts/debug/plot_shortlist_candidates.py \
        ... --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --model_config configs/model/udma.yaml
"""
import argparse
import csv
import re
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
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs
from src.utils.visualization import add_obs_dividers, plot_candidate
import inference as inf

CAD_DIR_RE = re.compile(r"cad(\d+)_(.+?)_[\d.]+MHz_\d+$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates_csv", type=Path, required=True,
                   help="{method}_candidates.csv written by scripts/inference.py")
    p.add_argument("--cadence_list", type=Path, required=True,
                   help="cadence list file passed to scripts/inference.py "
                        "(one line per cadence, 6 space-separated .h5 paths)")
    p.add_argument("--cad_idx", type=int, default=None,
                   help="0-based line into --cadence_list; parsed from "
                        "--candidates_csv's parent directory name if omitted")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    p.add_argument("--maps_dir", type=Path, default=None,
                   help="Output dir of scripts/recompute_anomaly_maps.py — per-cadence "
                        "maps.npz with cached st1/st2/ss disagreement maps, fused offline "
                        "with model_config's scoring.weights (no model load). Preferred "
                        "over --checkpoint when available.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Model checkpoint for a live anomaly-map/reconstruction forward "
                        "pass (optional — omit to only plot the preprocessed waterfall, "
                        "as before). Used as a fallback for candidates not covered by "
                        "--maps_dir, or as the sole source if --maps_dir is omitted.")
    p.add_argument("--model_config", type=Path, default=None,
                   help="Model config YAML (required if --maps_dir or --checkpoint is given)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--all", action="store_true",
                   help="Plot every candidate in the CSV, ignoring in_short_list "
                        "(default: short list only, or all rows if the CSV has "
                        "no in_short_list column)")
    return p.parse_args()


def main():
    args = parse_args()

    method = args.candidates_csv.stem.replace("_candidates", "")

    cad_idx = args.cad_idx
    target = None
    m = CAD_DIR_RE.match(args.candidates_csv.parent.name)
    if m:
        if cad_idx is None:
            cad_idx = int(m.group(1))
        target = m.group(2)
    if cad_idx is None:
        raise SystemExit("--cad_idx not given and could not be parsed from "
                          f"{args.candidates_csv.parent.name!r}")

    lines = [line.split() for line in args.cadence_list.read_text().splitlines() if line.strip()]
    obs_paths = [Path(p) for p in lines[cad_idx]]

    with open(args.candidates_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{args.candidates_csv} has no candidates")
        return

    if args.all or "in_short_list" not in rows[0]:
        selected = rows
        if not args.all and "in_short_list" not in rows[0]:
            print("No in_short_list column in CSV (non-UDMA model) — plotting all rows")
    else:
        selected = [r for r in rows if r["in_short_list"].strip().lower() == "true"]
    print(f"{len(selected)}/{len(rows)} candidates selected")
    if not selected:
        return
    selected = sorted(selected, key=lambda r: -float(r["peak_score"]))

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans, tchans = frame["fchans"], frame["tchans"]
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]
    method_bp = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)

    if (args.maps_dir is not None or args.checkpoint is not None) and args.model_config is None:
        raise SystemExit("--maps_dir/--checkpoint requires --model_config")
    model_cfg = None
    if args.model_config is not None:
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)

    cached_maps = None
    if args.maps_dir is not None:
        maps_npz_path = args.maps_dir / args.candidates_csv.parent.name / "maps.npz"
        if maps_npz_path.exists():
            npz = np.load(maps_npz_path)
            weights = tuple(model_cfg.get("scoring", {}).get("weights", (0.5, 0.5, 0.5)))
            f_start_idx = {int(fs): i for i, fs in enumerate(npz["f_start"])}
            cached_maps = {"f_start_idx": f_start_idx, "st1": npz["st1"], "st2": npz["st2"],
                           "ss": npz["ss"], "weights": weights}
            print(f"Loaded cached maps -> {maps_npz_path} ({len(f_start_idx)} snippets)")
        else:
            print(f"No cached maps at {maps_npz_path}, falling back to --checkpoint if given")

    model = None
    has_amap = False
    if args.checkpoint is not None:
        input_shape = (tchans, fchans, 1)
        print(f"Loading model from {args.checkpoint}")
        model = inf.load_model(args.checkpoint, model_cfg, input_shape, args.device)
        has_amap = hasattr(model, "anomaly_map")

    def get_cached_amap(f_start):
        if cached_maps is None:
            return None
        idx = cached_maps["f_start_idx"].get(f_start)
        if idx is None:
            return None
        w1, w2, w3 = cached_maps["weights"]
        return (w1 * cached_maps["st1"][idx].astype(np.float32) +
                w2 * cached_maps["st2"][idx].astype(np.float32) +
                w3 * cached_maps["ss"][idx].astype(np.float32))

    print(f"Loading {len(obs_paths)} obs files for cadence {cad_idx}...")
    obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]

    def extract_at(f_start):
        raw_frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
        normed_frames = [
            core_transform(bandpass_correct(fr, method=method_bp, poly_degree=poly_degree),
                            mad_epsilon)
            for fr in raw_frames
        ]
        return np.concatenate(normed_frames, axis=0)[:tchans, :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(selected):
        fs = int(float(row["f_start_peak"]))
        score = float(row["peak_score"])
        snippet = extract_at(fs)
        f_center_mhz = fs * df / 1e6

        amap = get_cached_amap(fs)
        recon = None
        if amap is None and model is not None:
            if has_amap:
                x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    amap = model.anomaly_map(x)[0].cpu().numpy()
            else:
                recon = inf.reconstruct_batch(model, [snippet], args.device)[0]

        if amap is not None or recon is not None:
            fig = plot_candidate(
                original=snippet, reconstruction=recon, score=score, sigma=None,
                method=method, cad_idx=cad_idx, target=target or "unknown",
                f_start=fs, df=df, anomaly_map=amap, show_overlay=False,
            )
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            vmin, vmax = np.percentile(snippet, [1, 99])
            im = ax.imshow(snippet, aspect="auto", origin="upper", vmin=vmin, vmax=vmax,
                            cmap="viridis")
            ax.set_title("Preprocessed")
            ax.set_xlabel("Freq channel")
            ax.set_ylabel("Time bin (ABACAD)")
            add_obs_dividers(ax, snippet.shape[0])
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(
                f"cad={cad_idx} ({target or 'unknown'})  f_start={fs}  "
                f"f~{f_center_mhz:.4f} MHz  {method} score={score:.4f}",
                fontsize=11,
            )
            plt.tight_layout()

        out_path = args.out_dir / f"{method}_rank{rank:02d}_f{fs}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{rank}] f_start={fs} score={score:.4f} -> {out_path}")

    print(f"Saved {len(selected)} PNGs -> {args.out_dir}")


if __name__ == "__main__":
    main()
