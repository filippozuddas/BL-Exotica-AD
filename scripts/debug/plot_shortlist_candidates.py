"""
Re-plot short-listed candidates from an already-run inference pass as
individual high-resolution PNGs, showing raw power + preprocessed waterfall
(no model needed — just extracts and plots the snippet, like
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

Usage:
    python scripts/debug/plot_shortlist_candidates.py \
        --candidates_csv outputs/inference/<run>/cad03_.../topk_candidates.csv \
        --cadence_list data/processed/inference_cadences.txt \
        --data_config configs/data/gbt_fine.yaml \
        --out_dir outputs/inference/<run>/cad03_.../shortlist_png
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs
from src.utils.visualization import add_obs_dividers

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

    print(f"Loading {len(obs_paths)} obs files for cadence {cad_idx}...")
    obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]

    def extract_at(f_start):
        raw_frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
        normed_frames = [
            core_transform(bandpass_correct(fr, method=method_bp, poly_degree=poly_degree),
                            mad_epsilon)
            for fr in raw_frames
        ]
        raw = np.concatenate(raw_frames, axis=0)[:tchans, :]
        snippet = np.concatenate(normed_frames, axis=0)[:tchans, :]
        return raw, snippet

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(selected):
        fs = int(float(row["f_start_peak"]))
        score = float(row["peak_score"])
        raw, snippet = extract_at(fs)
        f_center_mhz = fs * df / 1e6

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, arr, title in zip(axes, [raw, snippet], ["Raw power", "Preprocessed"]):
            vmin, vmax = np.percentile(arr, [1, 99])
            im = ax.imshow(arr, aspect="auto", origin="upper", vmin=vmin, vmax=vmax,
                            cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("Freq channel")
            ax.set_ylabel("Time bin (ABACAD)")
            add_obs_dividers(ax, arr.shape[0])
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
