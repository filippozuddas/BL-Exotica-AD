"""
Recompute and persist UDMA anomaly-map components for an existing inference run.

The 364-cadence heldout scan (``outputs/inference/exotica_heldout_topk001``)
saved per-snippet *scalar* scores and per-cadence candidate CSVs, but not the
``(6, 64)`` anomaly maps themselves — so every subsequent question about the
short-list rule, the fusion weights, or the FAR pre-cut required re-reading
~26 GB per cadence off a spinning disk (~30 h for a full rescan).

This script pays that disk cost **once** and writes the maps to disk, after
which all of those questions become offline array operations on a few GB.

What gets saved per cadence (``<out_dir>/<cad_dirname>/maps.npz``):

* ``st1``/``st2``/``ss`` — the three raw disagreement maps (float16,
  ``(n_sel, 6, 64)``) for every snippet above ``--map_quantile``, *not* the
  fused ``cob``. Saving them separately is the whole point: any
  ``scoring.weights`` combination is then a linear recombination, computable
  offline with no forward pass.
* ``f_start`` — the window start channel of each saved map.
* ``score`` — that snippet's scalar score from the original run.
* ``probe_*`` — the same three maps for a random cadence-wide background
  sample (``--off_ceiling_probe``), used to build the OFF-noise ceiling. The
  original run used only 300 probes, tuned on a narrowband Voyager file that
  covered ~15% of the band; on a full-band Exotica cadence that is ~0.2%
  coverage (see ``off_ceiling_probe_bandwidth_scaling`` in memory), so the
  default here is much larger.
* ``median``/``mad_sigma``/``thresh_3``/``thresh_5``/``far_thresh`` — the
  per-cadence scalars derived from the *full* score pool, so thresholds can
  be reproduced offline exactly as ``scripts/inference.py`` computed them.

Why the 95th percentile and not the 99th the original run clustered at:
selecting a *looser* pool than the pipeline's own FAR cut is what makes the
"does ``FAR_QUANTILE = 0.99`` discard real signals?" question answerable
without a second full scan — the answer lives precisely in the band between
the two quantiles.

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/BL-Exotica-AD \
    python scripts/recompute_anomaly_maps.py \
        --checkpoint outputs/training/20260707_093113_6d0d1ba/checkpoints/epoch=057*.ckpt \
        --cadence_list data/raw/gbt_0000_heldout_cadences.txt \
        --run_dir outputs/inference/exotica_heldout_topk001 \
        --model_config configs/model/udma_old_teacher.yaml \
        --out_dir outputs/inference/exotica_heldout_maps \
        --num_workers 32
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.inference as inf
from src.data.torch_dataset import _load_full_obs

SCORE_COL = "topk_score"


def split_scores(scores_csv: Path, cache_dir: Path, score_col: str) -> None:
    """One-shot pass over the run's global score CSV -> per-cadence .npz.

    The CSV is ~2.9 GB / ~48M rows, too slow to re-scan per cadence, so it is
    read once in chunks and demultiplexed by ``cadence_idx``. Idempotent: does
    nothing if the cache is already populated.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "_complete"
    if marker.exists():
        print(f"Score cache already built -> {cache_dir}")
        return

    print(f"Splitting {scores_csv} by cadence (one-shot, ~48M rows)...")
    t0 = time.time()
    buf: dict[int, list] = {}
    reader = pd.read_csv(scores_csv, usecols=["cadence_idx", "f_start", score_col],
                         chunksize=5_000_000)
    for n_chunk, chunk in enumerate(reader):
        for cad_idx, g in chunk.groupby("cadence_idx"):
            buf.setdefault(int(cad_idx), []).append(
                (g["f_start"].to_numpy(np.int64), g[score_col].to_numpy(np.float32))
            )
        print(f"  chunk {n_chunk}: {len(buf)} cadences seen  ({time.time()-t0:.0f}s)")

    for cad_idx, parts in buf.items():
        f_start = np.concatenate([p[0] for p in parts])
        score = np.concatenate([p[1] for p in parts])
        order = np.argsort(f_start)
        np.savez(cache_dir / f"cad{cad_idx:02d}.npz",
                 f_start=f_start[order], score=score[order])
    marker.touch()
    print(f"Score cache built: {len(buf)} cadences in {time.time()-t0:.0f}s")


def compute_maps(model, f_starts, batch_size, device, num_workers, file_specs,
                 fchans, tchans, preproc):
    """Preprocess ``f_starts`` in a worker pool, forward them, return st1/st2/ss.

    Snippets are streamed rather than materialised: a 5% selection on a
    full-band cadence is ~6500 snippets x (96, 1024) float32 = ~2.6 GB, which
    would otherwise sit in RAM alongside the ~26 GB of staged observations.
    """
    import multiprocessing as mp

    out = {"st1": [], "st2": [], "ss": []}
    batch = []
    ctx = mp.get_context("spawn")
    chunksize = max(1, min(64, len(f_starts) // (num_workers * 4) or 1))

    def flush(batch):
        x = torch.from_numpy(np.stack(batch)).float().unsqueeze(1).to(device)
        with torch.no_grad():
            comps = model.anomaly_map_components(x)
        for k in out:
            out[k].append(comps[k].cpu().numpy().astype(np.float16))

    with ctx.Pool(num_workers, initializer=inf._worker_init,
                  initargs=(file_specs, fchans, tchans, preproc)) as pool:
        for snippet in pool.imap(inf._preprocess_at, f_starts, chunksize=chunksize):
            batch.append(snippet)
            if len(batch) == batch_size:
                flush(batch)
                batch = []
    if batch:
        flush(batch)

    return {k: np.concatenate(v, axis=0) if v else np.zeros((0, 0, 0), np.float16)
            for k, v in out.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Existing inference run dir (must contain inference_scores.csv)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path,
                   default=ROOT / "configs/model/udma_old_teacher.yaml")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--score_col", default=SCORE_COL)
    p.add_argument("--map_quantile", type=float, default=0.95,
                   help="Save maps for every snippet at or above this quantile of "
                        "the cadence's own score pool. Must be LOOSER than the "
                        "pipeline's FAR_QUANTILE (0.99) for the pre-cut question "
                        "to be answerable offline.")
    p.add_argument("--off_ceiling_probe", type=int, default=3000,
                   help="Random cadence-wide background snippets for the OFF-noise "
                        "ceiling pool. Default scaled up from the pipeline's 300, "
                        "which was tuned on a narrowband file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--max_cadences", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute cadences that already have maps.npz (default: skip, "
                        "so an interrupted run resumes where it stopped).")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans, tchans = frame["fchans"], frame["tchans"]
    stride = frame.get("stride_infer", fchans // 2)
    downsample_factor = frame.get("downsample_factor", 1)
    input_shape = (tchans, fchans, 1)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    cadence_lines = [line.strip().split()
                     for line in args.cadence_list.read_text().splitlines()
                     if line.strip()]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "_scores"
    split_scores(args.run_dir / "inference_scores.csv", cache_dir, args.score_col)

    print(f"Loading model from {args.checkpoint}")
    model = inf.load_model(args.checkpoint, model_cfg, input_shape, args.device)
    if not hasattr(model, "anomaly_map_components"):
        raise SystemExit("Model has no anomaly_map_components — this script is UDMA-only.")

    scratch_dir = args.out_dir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]

        meta = None
        for obs_path in obs_paths:
            try:
                meta = inf.read_cadence_meta(obs_path)
                break
            except OSError:
                continue
        if meta is None:
            print(f"\nCadence {cad_idx}: SKIPPING — all files corrupt")
            continue

        cad_dir = args.out_dir / inf.make_cadence_dirname(cad_idx, meta)
        out_npz = cad_dir / "maps.npz"
        if out_npz.exists() and not args.overwrite:
            print(f"Cadence {cad_idx}: maps.npz exists, skipping")
            continue

        score_path = cache_dir / f"cad{cad_idx:02d}.npz"
        if not score_path.exists():
            print(f"Cadence {cad_idx}: no scores in cache, skipping")
            continue
        sc = np.load(score_path)
        all_f, all_s = sc["f_start"], sc["score"]

        # Thresholds reproduced exactly as scripts/inference.py derived them
        # from this same pool, so anything computed offline downstream lines up
        # with the original run's candidate CSVs.
        median, mad_sigma = inf.robust_stats(all_s)
        thresh_3 = median + 3 * mad_sigma
        thresh_5 = median + 5 * mad_sigma
        far_thresh = float(np.quantile(all_s, inf.FAR_QUANTILE))
        map_thresh = float(np.quantile(all_s, args.map_quantile))

        sel_mask = all_s >= map_thresh
        sel_f = all_f[sel_mask]
        sel_s = all_s[sel_mask]

        rng = np.random.default_rng(cad_idx)
        n_probe = min(args.off_ceiling_probe, len(all_f))
        probe_f = rng.choice(all_f, size=n_probe, replace=False)

        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx}: {meta['source']}  fch1={meta['fch1_mhz']:.1f} MHz")
        print(f"  {len(all_f)} snippets | q{args.map_quantile:.2f}={map_thresh:.4f} "
              f"-> {len(sel_f)} maps | FAR1%={far_thresh:.4f} | probe={n_probe}")

        t_load = time.time()
        obs_arrays = []
        corrupt = False
        for i, obs_path in enumerate(obs_paths):
            try:
                obs_arrays.append(_load_full_obs(obs_path, downsample_factor))
            except OSError as e:
                print(f"  SKIPPING cadence — corrupt file obs {i}: {obs_path.name} — {e}")
                corrupt = True
                break
        if corrupt:
            continue
        print(f"  Loaded {sum(a.nbytes for a in obs_arrays)/1e9:.1f} GB "
              f"in {time.time()-t_load:.1f}s")

        shm_paths, file_specs = [], []
        for i, arr in enumerate(obs_arrays):
            path = scratch_dir / f"cad{cad_idx:02d}_obs{i}.f32"
            arr.tofile(str(path))
            shm_paths.append(path)
            file_specs.append((str(path), arr.shape, arr.dtype))
        del obs_arrays

        try:
            # One pool for selection + probe together: spawning 32 workers
            # re-imports torch in each, ~10 s that would otherwise be paid twice
            # per cadence on a ~100 s budget.
            t0 = time.time()
            combined = [int(x) for x in sel_f] + [int(x) for x in probe_f]
            maps = compute_maps(model, combined, args.batch_size, args.device,
                                args.num_workers, file_specs, fchans, tchans, preproc)
            n_sel = len(sel_f)
            sel_maps = {k: v[:n_sel] for k, v in maps.items()}
            probe_maps = {k: v[n_sel:] for k, v in maps.items()}
            print(f"  Computed {n_sel}+{n_probe} maps in {time.time()-t0:.1f}s")

            cad_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_npz,
                f_start=sel_f.astype(np.int64), score=sel_s.astype(np.float32),
                st1=sel_maps["st1"], st2=sel_maps["st2"], ss=sel_maps["ss"],
                probe_f_start=probe_f.astype(np.int64),
                probe_st1=probe_maps["st1"], probe_st2=probe_maps["st2"],
                probe_ss=probe_maps["ss"],
                median=np.float32(median), mad_sigma=np.float32(mad_sigma),
                thresh_3=np.float32(thresh_3), thresh_5=np.float32(thresh_5),
                far_thresh=np.float32(far_thresh), map_thresh=np.float32(map_thresh),
                map_quantile=np.float32(args.map_quantile),
                stride=np.int64(stride), fchans=np.int64(fchans),
                n_snippets_total=np.int64(len(all_f)),
            )
            print(f"  Saved -> {out_npz}  ({out_npz.stat().st_size/1e6:.1f} MB)")
        finally:
            for path in shm_paths:
                path.unlink(missing_ok=True)

        done = cad_idx + 1
        rate = (time.time() - t_all) / done
        print(f"  [{done}/{len(cadence_lines)}]  {rate:.0f}s/cadence  "
              f"ETA {rate*(len(cadence_lines)-done)/3600:.1f}h")

    print("\nDone.")


if __name__ == "__main__":
    main()
