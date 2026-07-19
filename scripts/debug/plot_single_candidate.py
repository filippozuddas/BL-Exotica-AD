"""
Plot one specific candidate from an already-run inference pass, without
re-running the full sliding-window scan over the cadence.

Two ways to pick the candidate:

1. By cadence index + rank in an existing ``{method}_candidates.csv``
   (as written by ``scripts/inference.py``) — obs paths and f_start are
   resolved automatically:

    python scripts/debug/plot_single_candidate.py \
        --checkpoint outputs/<run>/checkpoints/best.ckpt \
        --model_config configs/model/udma.yaml \
        --data_config configs/data/gbt_fine.yaml \
        --cadence_list data/processed/inference_cadences.txt \
        --cad_idx 3 \
        --candidates_csv outputs/inference/<run>/cad03_.../topk_candidates.csv \
        --rank 1 \
        --out_dir outputs/inference/<run>/cad03_.../inspect \
        --method topk

2. Manually, by explicit obs paths + f_start:

    python scripts/debug/plot_single_candidate.py \
        --checkpoint outputs/<run>/checkpoints/best.ckpt \
        --model_config configs/model/udma.yaml \
        --data_config configs/data/gbt_fine.yaml \
        --obs_paths <6 space-separated .h5 paths for the cadence> \
        --f_start 42123264 \
        --out_dir outputs/inference/<run>/cad03_.../inspect \
        --method topk
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.torch_dataset import _load_full_obs
from src.search.candidates import on_off_contrast
import inference as inf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True)
    p.add_argument("--data_config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--method", default="topk", choices=inf.METHODS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--obs_paths", nargs=6, default=None,
                    help="6 space-separated .h5 paths for the cadence (manual mode)")
    p.add_argument("--f_start", type=int, default=None,
                    help="window start channel (manual mode)")
    p.add_argument("--target", default="candidate",
                    help="label for the plot title (manual mode only; auto mode "
                         "reads the real target name from the .h5 header)")

    p.add_argument("--cadence_list", type=Path, default=None,
                    help="cadence list file passed to scripts/inference.py "
                         "(one line per cadence, 6 space-separated .h5 paths)")
    p.add_argument("--cad_idx", type=int, default=None,
                    help="0-based line number into --cadence_list")
    p.add_argument("--candidates_csv", type=Path, default=None,
                    help="{method}_candidates.csv written by scripts/inference.py "
                         "for this cadence")
    p.add_argument("--rank", type=int, default=1,
                    help="1-based rank into --candidates_csv, sorted by peak_score "
                         "descending (1 = top candidate)")
    return p.parse_args()


def resolve_auto(args):
    """--cadence_list/--cad_idx/--candidates_csv/--rank -> (obs_paths, f_start, target)."""
    lines = [
        line.split() for line in args.cadence_list.read_text().splitlines() if line.strip()
    ]
    obs_paths = [Path(p) for p in lines[args.cad_idx]]

    with open(args.candidates_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{args.candidates_csv} has no candidates")
    if not (1 <= args.rank <= len(rows)):
        raise ValueError(f"--rank {args.rank} out of range (1..{len(rows)})")
    f_start = int(float(rows[args.rank - 1]["f_start_peak"]))

    meta = inf.read_cadence_meta(obs_paths[0])
    return obs_paths, f_start, meta["source"]


def main():
    args = parse_args()
    if args.cadence_list is not None:
        if args.cad_idx is None or args.candidates_csv is None:
            raise SystemExit("--cadence_list requires --cad_idx and --candidates_csv")
        obs_paths, f_start, target = resolve_auto(args)
    else:
        if args.obs_paths is None or args.f_start is None:
            raise SystemExit("manual mode requires --obs_paths and --f_start")
        obs_paths = [Path(p) for p in args.obs_paths]
        f_start = args.f_start
        target = args.target

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans, tchans = frame["fchans"], frame["tchans"]
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    input_shape = (tchans, fchans, 1)
    model = inf.load_model(args.checkpoint, model_cfg, input_shape, args.device)

    obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]
    file_specs = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.out_dir / "_scratch"
    scratch.mkdir(exist_ok=True)
    for i, arr in enumerate(obs_arrays):
        path = scratch / f"obs{i}.f32"
        arr.tofile(str(path))
        file_specs.append((str(path), arr.shape, arr.dtype))

    try:
        inf._worker_init(file_specs, fchans, tchans, preproc)
        snippet = inf._preprocess_at(f_start)

        x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(args.device)
        with torch.no_grad():
            score = float(model.anomaly_score(x, method=args.method)[0])
            amap = None
            recon = None
            if hasattr(model, "anomaly_map"):
                amap = model.anomaly_map(x)[0].cpu().numpy()
                stats = on_off_contrast(amap)
                print(f"on_off_contrast={stats['on_off_contrast']:.3f}  "
                      f"on_mean={stats['on_mean']:.3f}  off_mean={stats['off_mean']:.3f}")
            else:
                recon = model(x).squeeze(1)[0].cpu().numpy()

        fig = inf.plot_candidate(
            original=snippet, reconstruction=recon,
            score=score, sigma=0.0, method=args.method,
            cad_idx=args.cad_idx if args.cad_idx is not None else 0, target=target,
            f_start=f_start, df=df, anomaly_map=amap,
        )
        out_path = args.out_dir / f"{args.method}_f{f_start}_inspect.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved -> {out_path}")
    finally:
        for spec in file_specs:
            Path(spec[0]).unlink()


if __name__ == "__main__":
    main()
