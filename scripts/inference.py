"""
Run inference on real cadences from a cadence list file.

Loads each cadence's .h5 files fully into RAM, slides a (tchans, fchans)
window across the frequency axis at stride_infer, preprocesses each snippet
in parallel across CPU cores, and scores batches on GPU.

Each cadence gets its own output folder named:
    cad{idx}_{target}_{fch1_MHz}MHz_{date}

Candidate snippets are saved with per-candidate plots showing
original | reconstruction | error map, either as one multi-page PDF per
cadence per method (default, fast vetting) or individual PNGs
(--plot_format png/both).

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/BL-Exotica-AD \
    python scripts/inference.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --cadence_list data/processed/inference_cadences.txt \
        --out_dir outputs/inference/run_name \
        --num_workers 32
"""

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import build_autoencoder
from src.data.preprocessing import bandpass_correct, core_transform
from src.data.torch_dataset import _load_full_obs
from src.search.candidates import (
    cluster_candidates, on_off_contrast, full_row_hits, off_noise_ceiling,
)

FAR_QUANTILE = 0.99
MIN_OFF_POOL = 30
from src.utils.visualization import plot_candidate

METHODS = ["recon", "topk", "max", "cadence"]
MAD_SCALE = 1.4826

_shared_obs = None
_shared_fchans = None
_shared_tchans = None
_shared_preproc = None


def _worker_init(file_specs, fchans, tchans, preproc):
    """Pool(initializer=...) target: memory-map the parent's observation
    files by path.

    Runs once per spawned worker process. Workers are started with the
    'spawn' context (never forked from the main process), because the main
    process already holds an initialized CUDA context by the time the pool
    is created — forking after CUDA init can deadlock silently (workers
    inherit an unusable copy of driver-internal locks). 'spawn' processes
    never inherit parent memory, so data must come in explicitly rather
    than via a global set right before a fork.

    Uses plain disk-backed np.memmap rather than multiprocessing.shared_memory:
    the latter is backed by /dev/shm, which defaults to a small quota (e.g.
    64 MB in Docker) and dies with a SIGBUS ("Bus error") once a write crosses
    it — nowhere near enough for a ~26 GB cadence.
    """
    global _shared_obs, _shared_fchans, _shared_tchans, _shared_preproc
    _shared_obs = [np.memmap(path, dtype=dtype, mode="r", shape=shape)
                   for path, shape, dtype in file_specs]
    _shared_fchans = fchans
    _shared_tchans = tchans
    _shared_preproc = preproc


def _preprocess_at(f_start):
    method = _shared_preproc.get("bandpass_method", "polynomial")
    poly_degree = _shared_preproc.get("poly_degree", 3)
    mad_epsilon = _shared_preproc.get("mad_epsilon", 1e-6)
    frames = [obs[:, f_start:f_start + _shared_fchans] for obs in _shared_obs]
    normed = [
        core_transform(bandpass_correct(f, method=method, poly_degree=poly_degree), mad_epsilon)
        for f in frames
    ]
    stacked = np.concatenate(normed, axis=0)[:_shared_tchans, :]
    return stacked


def load_model(checkpoint_path: Path, model_config: dict, input_shape: tuple, device: str):
    model = build_autoencoder(input_shape, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def score_batch(model, snippets: list, method: str, device: str) -> np.ndarray:
    x = torch.from_numpy(np.array(snippets)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        s = model.anomaly_score(x, method=method)
    return s.cpu().numpy()


def reconstruct_batch(model, snippets: list, device: str) -> np.ndarray:
    x = torch.from_numpy(np.array(snippets)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze(1).cpu().numpy()


def robust_stats(scores: np.ndarray):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    sigma = mad * MAD_SCALE
    return median, sigma


def read_cadence_meta(h5_path: Path) -> dict:
    """Read target name, start frequency and observation date from .h5 header."""
    with h5py.File(str(h5_path), 'r') as f:
        attrs = dict(f['data'].attrs) if 'data' in f and f['data'].attrs else {}
        if not attrs:
            for key in f.keys():
                if hasattr(f[key], 'attrs') and len(f[key].attrs) > 0:
                    attrs = dict(f[key].attrs)
                    break
            if not attrs and len(f.attrs) > 0:
                attrs = dict(f.attrs)

    source = attrs.get('source_name', b'unknown')
    if isinstance(source, bytes):
        source = source.decode('utf-8', errors='replace')
    source = source.strip().replace(' ', '_')

    fch1 = float(attrs.get('fch1', 0.0))

    tstart_mjd = float(attrs.get('tstart', 0.0))
    if tstart_mjd > 0:
        from astropy.time import Time
        t = Time(tstart_mjd, format='mjd')
        date_str = t.iso[:10].replace('-', '')
    else:
        date_str = "nodate"

    return {"source": source, "fch1_mhz": fch1, "date": date_str}


def make_cadence_dirname(cad_idx: int, meta: dict) -> str:
    fch1_str = f"{meta['fch1_mhz']:.1f}MHz"
    return f"cad{cad_idx:02d}_{meta['source']}_{fch1_str}_{meta['date']}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True,
                   help="Text file with one cadence per line (6 space-separated .h5 paths)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--top_k", type=int, default=30,
                   help="Number of top candidates to plot per cadence per method")
    p.add_argument("--plot_format", default="pdf", choices=["pdf", "png", "both"],
                   help="Candidate plot output: one multi-page PDF per cadence per "
                        "method (fast vetting, default), individual PNGs (old "
                        "behaviour), or both.")
    p.add_argument("--methods", nargs="+", default=["recon"], choices=METHODS,
                   help="Anomaly scoring methods to run. The plain Autoencoder/MAE/VAE "
                        "only support 'recon'/'topk'; 'cadence' requires the ViT-MAE "
                        "backbone; UDMA supports 'recon'/'topk'/'max' (its own topk_frac "
                        "default, no 'cadence').")
    p.add_argument("--ignore_short_list", action="store_true",
                   help="Diagnostic only: plot the top_k candidates by on_off_contrast "
                        "regardless of in_short_list (off_leak). The full CSV always "
                        "contains every candidate with the short-list columns either way; "
                        "this only affects which ones get a plot.")
    p.add_argument("--off_ceiling_probe", type=int, default=300,
                   help="Extra background snippets sampled across the whole cadence "
                        "(beyond the candidate clusters) to build the off_noise_ceiling "
                        "pool, so the ceiling reflects the cadence-wide OFF noise floor "
                        "rather than only the OFF-row cells co-located with already-"
                        "flagged candidates (a small, detection-biased sample).")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    tchans = frame["tchans"]
    stride = frame.get("stride_infer", fchans // 2)
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]
    input_shape = (tchans, fchans, 1)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    methods = args.methods

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, input_shape, args.device)

    print(f"Cadences: {len(cadence_lines)}")
    print(f"Input shape: {input_shape}  methods: {methods}")
    print(f"Frame: {tchans}x{fchans}, stride={stride}, downsample={downsample_factor}")
    print(f"Batch size: {args.batch_size}, workers: {args.num_workers}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {m: [] for m in methods}
    all_rows = []
    cadence_dirs = []

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]

        # Read metadata — try each obs file until one opens
        meta = None
        for obs_path in obs_paths:
            try:
                meta = read_cadence_meta(obs_path)
                break
            except OSError:
                continue
        if meta is None:
            print(f"\nCadence {cad_idx}: SKIPPING — all files corrupt")
            continue
        target_name = meta["source"]
        cad_dirname = make_cadence_dirname(cad_idx, meta)
        cad_dir = args.out_dir / cad_dirname
        cad_dir.mkdir(parents=True, exist_ok=True)
        cadence_dirs.append(cad_dir)

        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx}: {target_name}  fch1={meta['fch1_mhz']:.1f} MHz  "
              f"date={meta['date']}")
        print(f"  -> {cad_dir}")
        print(f"{'='*70}")

        # Load all observations into RAM
        t_load = time.time()
        obs_arrays = []
        corrupt = False
        for i, obs_path in enumerate(obs_paths):
            try:
                arr = _load_full_obs(obs_path, downsample_factor)
            except OSError as e:
                print(f"  SKIPPING cadence — corrupt file obs {i}: {obs_path.name} — {e}")
                corrupt = True
                break
            obs_arrays.append(arr)
            print(f"  Loaded obs {i}: {obs_path.name} -> {arr.shape}")
        if corrupt:
            continue
        load_time = time.time() - t_load

        nchans = obs_arrays[0].shape[1]
        n_snippets = max(0, (nchans - fchans) // stride + 1)
        mem_gb = sum(a.nbytes for a in obs_arrays) / 1e9
        print(f"  {mem_gb:.1f} GB in RAM, loaded in {load_time:.1f}s")
        print(f"  nchans={nchans} -> {n_snippets} snippets (stride={stride})")

        # Stage each observation as a disk-backed memmap so 'spawn' workers can
        # open it by path instead of relying on fork's copy-on-write (which
        # would require forking the main process after it already holds an
        # initialized CUDA context — see _worker_init).
        scratch_dir = args.out_dir / "_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        shm_paths = []
        file_specs = []
        for i, arr in enumerate(obs_arrays):
            path = scratch_dir / f"cad{cad_idx:02d}_obs{i}.f32"
            arr.tofile(str(path))
            shm_paths.append(path)
            file_specs.append((str(path), arr.shape, arr.dtype))
        del obs_arrays

        f_starts = [i * stride for i in range(n_snippets)]

        t0 = time.time()
        batch_snippets = []
        batch_fstarts = []
        processed_count = 0
        cad_scores = {m: [] for m in methods}
        cad_fstarts = []

        chunksize = max(1, min(256, n_snippets // (args.num_workers * 4)))
        print(f"  Pool: {args.num_workers} workers (spawn), chunksize={chunksize}")

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.num_workers, initializer=_worker_init,
                      initargs=(file_specs, fchans, tchans, preproc)) as pool:
            for f_start, snippet in zip(f_starts, pool.imap(_preprocess_at, f_starts,
                                                             chunksize=chunksize)):
                batch_snippets.append(snippet)
                batch_fstarts.append(f_start)

                if len(batch_snippets) == args.batch_size or processed_count == n_snippets - 1:
                    batch_scores = {m: score_batch(model, batch_snippets, m, args.device)
                                    for m in methods}

                    for j in range(len(batch_snippets)):
                        cad_fstarts.append(batch_fstarts[j])

                        row = {
                            "cadence_idx": cad_idx,
                            "target": target_name,
                            "f_start": batch_fstarts[j],
                            "f_center_mhz": batch_fstarts[j] * df / 1e6,
                        }
                        for m in methods:
                            score = float(batch_scores[m][j])
                            cad_scores[m].append(score)
                            all_scores[m].append(score)
                            row[f"{m}_score"] = score
                        all_rows.append(row)

                    batch_snippets = []
                    batch_fstarts = []

                processed_count += 1
                if processed_count % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = processed_count / elapsed
                    eta = (n_snippets - processed_count) / rate
                    print(f"  {processed_count}/{n_snippets} snippets  "
                          f"({rate:.0f}/s, ETA {eta:.0f}s)")

        elapsed = time.time() - t0
        print(f"  Scored {n_snippets} snippets in {elapsed:.1f}s "
              f"({n_snippets/max(elapsed,1):.0f}/s)")

        # ---- Per-cadence summary & plots ----
        cad_arrs = {m: np.array(cad_scores[m]) for m in methods}
        cad_fstarts_arr = np.array(cad_fstarts)

        # Re-populate this (main) process's globals so _preprocess_at can
        # re-derive individual snippets on demand for the cluster-peak plots
        # below, without keeping all n_snippets arrays in RAM during scoring.
        _worker_init(file_specs, fchans, tchans, preproc)

        n_plots = 0
        for method in methods:
            scores = cad_arrs[method]
            median, mad_sigma = robust_stats(scores)
            # Gaussian 3s/5s kept only as a diagnostic reference line (plots
            # below) — heavy-tailed disagreement scores put this threshold
            # inside the OFF noise tail (see inference_threshold_off_null_calibration
            # in memory: cad02 crossed 0.88% of snippets vs the ~0.13% a
            # Gaussian 3s implies). It no longer drives candidate selection.
            thresh_3 = median + 3 * mad_sigma
            thresh_5 = median + 5 * mad_sigma
            n_3s = (scores > thresh_3).sum()
            n_5s = (scores > thresh_5).sum()

            # Declared operating point: fixed 1% FAR (empirical quantile of
            # this cadence's own score pool), replacing the parametric
            # Gaussian 3s as the threshold that actually forms candidates —
            # non-parametric, so it does not assume the heavy-tailed score
            # distribution is Gaussian.
            far_thresh = float(np.quantile(scores, FAR_QUANTILE))
            n_far = (scores > far_thresh).sum()
            print(f"  {method}: median={median:.4f}  MAD_s={mad_sigma:.4f}  "
                  f"3s(ref)={thresh_3:.4f}->{n_3s}  5s(ref)={thresh_5:.4f}->{n_5s}  "
                  f"FAR{100*(1-FAR_QUANTILE):.0f}%={far_thresh:.4f}->{n_far}")

            # Frequency-adjacency dedup (src/search/candidates.py): a single
            # wide/strong line triggers many adjacent overlapping windows
            # (stride < fchans) — collapse those into one candidate per event.
            clusters = cluster_candidates(cad_fstarts_arr, scores, far_thresh, stride, fchans, df)
            print(f"  {method}: {len(clusters)} distinct candidates after "
                  f"frequency-adjacency dedup (from {n_far} raw threshold-crossings)")

            # ON/OFF contrast (src/search/candidates.py): UDMA's (6,64) grid
            # has one row per cadence observation (ABACAD order) — reuses the
            # per-candidate anomaly map, no extra forward passes beyond one
            # per cluster. Descriptive only, added as CSV columns for human
            # review — see arch_constraint_encoder_based_only in memory.
            has_amap = hasattr(model, "anomaly_map")
            amaps_by_cluster = [None] * len(clusters)
            if has_amap and len(clusters) > 0:
                off_rows_default = (1, 3, 5)
                off_pool = []
                for i, (_, crow) in enumerate(clusters.iterrows()):
                    fs_c = int(crow["f_start_peak"])
                    snippet_c = _preprocess_at(fs_c)
                    x_c = torch.from_numpy(snippet_c).float().unsqueeze(0).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        amap_c = model.anomaly_map(x_c)[0].cpu().numpy()
                    amaps_by_cluster[i] = amap_c
                    off_idx = [r for r in off_rows_default if r < amap_c.shape[0]]
                    if off_idx:
                        off_pool.append(amap_c[off_idx, :].ravel())

                # Broaden the pool beyond the (small, detection-biased) candidate
                # clusters: sample background snippets spread across the whole
                # cadence so the ceiling reflects the true cadence-wide OFF noise
                # floor. Without this, the pool only contains OFF-row cells
                # co-located in frequency with already-flagged candidates, which
                # is neither representative nor guaranteed to be quiet — on a
                # real Voyager-1 run this produced an unstable, too-low ceiling
                # that killed the real candidate via off_leak (see
                # udma_voyager_shortlist_off_leak_concern in memory).
                if args.off_ceiling_probe > 0:
                    rng = np.random.default_rng(cad_idx)
                    n_bg = min(args.off_ceiling_probe, len(cad_fstarts_arr))
                    bg_fstarts = rng.choice(cad_fstarts_arr, size=n_bg, replace=False)
                    bg_snippets = [_preprocess_at(int(fs)) for fs in bg_fstarts]
                    for bstart in range(0, len(bg_snippets), args.batch_size):
                        chunk = bg_snippets[bstart:bstart + args.batch_size]
                        x_bg = torch.from_numpy(np.stack(chunk)).float().unsqueeze(1).to(args.device)
                        with torch.no_grad():
                            amap_bg = model.anomaly_map(x_bg).cpu().numpy()
                        off_idx = [r for r in off_rows_default if r < amap_bg.shape[1]]
                        if off_idx:
                            off_pool.append(amap_bg[:, off_idx, :].ravel())

                # Per-cadence OFF-noise-core ceiling (src/search/candidates.py,
                # Fase 3.1): replaces the Gaussian thresh_3 in the row-hit test
                # below. Falls back to thresh_3 if the cluster pool is too
                # small for a robust clip+quantile estimate (rare: needs
                # >=MIN_OFF_POOL cells; MIN_OFF_POOL/len(off_rows_default)
                # clusters at minimum, since each contributes ~len(off_rows)*nw cells).
                #
                # Floored at thresh_5 (2026-07-16, udma_voyager_shortlist_off_leak_concern):
                # off_ceiling is a small (~15-cluster + probe), detection-biased sample and
                # can compute BELOW the Gaussian reference on some cadences (observed on
                # Voyager-1/topk: ceiling 0.29 < thresh_3 0.36), loosening the row-hit bar
                # below what candidate-selection itself required and letting scattered,
                # unrelated marginal cells (score ~ FAR1% line) count as coherent hits — a
                # multiple-comparisons artifact over ~2000 snippets x 6 rows x several
                # clusters. Flooring at thresh_3 first (empirically tested on Voyager-1)
                # still let 1 residual noise candidate through (score right at the 3sigma
                # line); thresh_5 (5*MAD_sigma, ~2.9e-7 nominal tail under the Gaussian
                # reference, i.e. the standard many-comparisons response of raising the
                # per-test significance bar) gives a clean 3/3 real candidates, 0 noise.
                # Raising the row-hit floor does NOT reduce raw detection sensitivity: which
                # candidates get flagged at all is still governed by FAR1% clustering
                # upstream (cluster_candidates) — this floor only affects whether an already-
                # flagged candidate also earns automatic short-list membership; anything it
                # excludes still survives in the full per-cadence CSV for human review, so a
                # genuine but fainter signal is not silently dropped from the pipeline, only
                # from the auto-shortlist convenience view. This max() only ever tightens
                # relative to the un-floored ceiling — the original cad02 rationale for
                # off_ceiling (Gaussian reference buried far below real OFF noise) is
                # unaffected since there off_ceiling >> thresh_5 already. Unlike the
                # column-coherence gate tried and rejected the same day, this is a level-only
                # threshold change — no bias against fast/nonlinear drifters.
                off_pool_n = sum(len(p) for p in off_pool)
                if off_pool and off_pool_n >= MIN_OFF_POOL:
                    off_ceiling = max(off_noise_ceiling(np.concatenate(off_pool)), thresh_5)
                else:
                    off_ceiling = thresh_5
                print(f"  {method}: OFF-noise-core ceiling={off_ceiling:.4f} "
                      f"(pooled from {len(clusters)} clusters + "
                      f"{n_bg if args.off_ceiling_probe > 0 else 0} background snippets, "
                      f"{off_pool_n} cells, vs Gaussian 3s(ref)={thresh_3:.4f}, 5s(ref)={thresh_5:.4f})")

                contrasts, on_means, off_means = [], [], []
                n_on_hits_l, n_off_hits_l = [], []
                n_on_hits_full_l, n_off_hits_full_l, off_leak_l, in_short_list_l = [], [], [], []
                for amap_c in amaps_by_cluster:
                    stats = on_off_contrast(amap_c, threshold=off_ceiling)
                    contrasts.append(stats["on_off_contrast"])
                    on_means.append(stats["on_mean"])
                    off_means.append(stats["off_mean"])
                    n_on_hits_l.append(stats["n_on_hits"])
                    n_off_hits_l.append(stats["n_off_hits"])
                    # Short-list volume reduction (full_row_hits, no column
                    # restriction) — separate from on_off_contrast, which
                    # still ranks the short-listed candidates for plotting.
                    fr = full_row_hits(amap_c, threshold=off_ceiling)
                    n_on_hits_full_l.append(fr["n_on_hits_full"])
                    n_off_hits_full_l.append(fr["n_off_hits_full"])
                    off_leak_l.append(fr["off_leak"])
                    in_short_list_l.append(fr["in_short_list"])
                clusters["on_off_contrast"] = contrasts
                clusters["on_mean"] = on_means
                clusters["off_mean"] = off_means
                clusters["n_on_hits"] = n_on_hits_l
                clusters["n_off_hits"] = n_off_hits_l
                clusters["n_on_hits_full"] = n_on_hits_full_l
                clusters["n_off_hits_full"] = n_off_hits_full_l
                clusters["off_leak"] = off_leak_l
                clusters["in_short_list"] = in_short_list_l

            clusters.to_csv(cad_dir / f"{method}_candidates.csv", index=False)

            # Plot selection: rank by on_off_contrast when available (UDMA) —
            # peak_score alone rewards persistent RFI over target-only events
            # (see search_candidate_clustering memory, 2026-07-06 finding).
            # Caveat: n_off_hits==0 is NOT proof of target-locking — the
            # fixed-column drift-tolerance window can miss fast/non-linear
            # drifters that are actually present in every OFF block too
            # (observed on a satellite-like chirp candidate) — always confirm
            # visually, this only re-prioritises what to look at first.
            #
            # Volume reduction: before ranking, restrict to in_short_list
            # (full_row_hits: n_on_hits_full>=2 AND not off_leak). The full
            # CSV below still has every candidate — this only shrinks what
            # gets plotted for manual vetting.
            if has_amap and len(clusters) > 0 and not args.ignore_short_list:
                short_mask = clusters["in_short_list"].to_numpy()
                short_idx = np.nonzero(short_mask)[0]
                order = short_idx[np.argsort(-clusters["on_off_contrast"].to_numpy()[short_idx])]
                clusters_ranked = clusters.iloc[order].reset_index(drop=True)
                amaps_by_cluster = [amaps_by_cluster[i] for i in order]
                print(f"  {method}: {len(order)}/{len(clusters)} candidates "
                      f"in short list after ON/OFF full-row filter")
            elif has_amap and len(clusters) > 0:
                order = np.argsort(-clusters["on_off_contrast"].to_numpy())
                clusters_ranked = clusters.iloc[order].reset_index(drop=True)
                amaps_by_cluster = [amaps_by_cluster[i] for i in order]
                print(f"  {method}: --ignore_short_list set, plotting top {args.top_k} "
                      f"by on_off_contrast regardless of off_leak")
            else:
                clusters_ranked = clusters

            top_clusters = clusters_ranked.head(args.top_k).reset_index(drop=True)
            n_plots += len(top_clusters)
            want_pdf = args.plot_format in ("pdf", "both")
            want_png = args.plot_format in ("png", "both")
            pdf = (PdfPages(cad_dir / f"{method}_candidates.pdf") if want_pdf and
                   len(top_clusters) > 0 else None)
            for rank, row in top_clusters.iterrows():
                fs = int(row["f_start_peak"])
                score = float(row["peak_score"])
                snippet = _preprocess_at(fs)
                sigma = (score - median) / mad_sigma if mad_sigma > 0 else 0
                amap = amaps_by_cluster[rank]
                recon = None
                if amap is None:
                    recon = reconstruct_batch(model, [snippet], args.device)[0]
                fig = plot_candidate(
                    original=snippet,
                    reconstruction=recon,
                    score=score, sigma=sigma, method=method,
                    cad_idx=cad_idx, target=target_name,
                    f_start=fs, df=df,
                    anomaly_map=amap,
                )
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                if want_png:
                    fig.savefig(cad_dir / f"{method}_rank{rank:02d}_f{fs}.png",
                                dpi=120, bbox_inches="tight")
                plt.close(fig)
            if pdf is not None:
                pdf.close()

        # Per-cadence frequency profile, one per method
        for method in methods:
            scores = cad_arrs[method]
            fig, ax = plt.subplots(figsize=(16, 4))
            med, mad_s = robust_stats(scores)
            ax.plot(cad_fstarts, scores, linewidth=0.3, alpha=0.7)
            ax.axhline(med + 3 * mad_s, color="orange", ls="--", lw=1,
                       label=f"3s = {med + 3*mad_s:.3f}")
            ax.axhline(med + 5 * mad_s, color="red", ls="--", lw=1,
                       label=f"5s = {med + 5*mad_s:.3f}")
            ax.set_xlabel("Frequency channel (f_start)")
            ax.set_ylabel(f"{method} score")
            ax.set_title(f"{target_name} — {method} score vs frequency")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(cad_dir / f"{method}_score_vs_freq.png", dpi=150)
            plt.close()

        print(f"  Saved {n_plots} candidate plots -> {cad_dir}")

        for path in shm_paths:
            path.unlink()

    # ---- Global summary ----
    all_arrs = {m: np.array(all_scores[m]) for m in methods}
    n_total = len(next(iter(all_arrs.values()))) if all_arrs else 0

    csv_path = args.out_dir / "inference_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["cadence_idx", "target", "f_start", "f_center_mhz"]
        fieldnames += [f"{m}_score" for m in methods]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved scores -> {csv_path}")

    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY -- {n_total} snippets across {len(cadence_lines)} cadences")
    print(f"{'='*70}")

    for name in methods:
        scores = all_arrs[name]
        median, mad_sigma = robust_stats(scores)
        mean, std = scores.mean(), scores.std()
        thresh_3s = median + 3 * mad_sigma
        thresh_5s = median + 5 * mad_sigma
        n_3s = (scores > thresh_3s).sum()
        n_5s = (scores > thresh_5s).sum()

        print(f"\n  {name}:")
        print(f"    mean={mean:.4f}  std={std:.4f}")
        print(f"    median={median:.4f}  MAD_sigma={mad_sigma:.4f}")
        print(f"    min={scores.min():.4f}  max={scores.max():.4f}")
        print(f"    3s (robust)={thresh_3s:.4f}  -> {n_3s} candidates ({n_3s/n_total*100:.3f}%)")
        print(f"    5s (robust)={thresh_5s:.4f}  -> {n_5s} candidates ({n_5s/n_total*100:.3f}%)")

    # Global histograms, one panel per method
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5), squeeze=False)
    for ax, name in zip(axes[0], methods):
        scores = all_arrs[name]
        median, mad_sigma = robust_stats(scores)
        thresh_3 = median + 3 * mad_sigma
        thresh_5 = median + 5 * mad_sigma
        n_3s = (scores > thresh_3).sum()

        clipped = scores[scores < np.percentile(scores, 99.5)]
        ax.hist(clipped, bins=200, alpha=0.7, edgecolor="black", linewidth=0.2)
        ax.axvline(thresh_3, color="orange", ls="--", lw=1.5, label=f"3s = {thresh_3:.3f}")
        ax.axvline(thresh_5, color="red", ls="--", lw=1.5, label=f"5s = {thresh_5:.3f}")
        ax.set_xlabel("Anomaly score (MSE)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} -- {n_3s} candidates > 3s (robust)")
        ax.legend()

    plt.suptitle(f"Score distribution -- {n_total} snippets, {len(cadence_lines)} cadences\n"
                 f"Thresholds: median + k * MAD * 1.4826", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inference_score_distributions.png", dpi=150)
    plt.close()
    print(f"\nSaved -> {args.out_dir / 'inference_score_distributions.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
