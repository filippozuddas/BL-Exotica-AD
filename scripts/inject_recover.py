"""
Injection-recovery test on real cadences (Phase 2 — run BEFORE inference).

Injects ON-only narrowband signals at varying SNR into quiet frequency
windows of real cadences, and compares against an RFI-inclusive background
built in-script from a full random probe of the same cadences (not just the
quiet half used for injection sites).

Injection goes through src/data/morphologies.py (inject_on_only_cadence
in src/data/synthetic.py): one setigen Frame per observation assembled into a
Cadence (t_overwrite=True) so the drift phase stays correct across the OFF
observations' real duration (slew is negligible, per-obs duration is not), but
the signal is only rendered into the 3 ON frames — OFF frames are returned
byte-identical. Drift/width/profile are sampled once per injection site (same
distributions as training) and held fixed across the SNR sweep at that site, so
SNR is the sweep's one independent variable.

The key question: at what SNR does an ON-only injection rank above the
real RFI in the background? This is the operationally relevant metric —
not the clean-baseline sigma (already measured by cadence_snr_sweep.py,
which uses a preprocessed training cache rather than real cadences).

This test establishes the model's detection capability and must run before
scripts/inference.py — it no longer depends on a prior inference run.

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/BL-Exotica-AD \
    python scripts/inject_recover.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --cadence_list data/processed/inject_recovery_cadences.txt \
        --out_dir outputs/inject_recovery/T1
"""

import argparse
import csv
import logging
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
from src.data.morphologies import MORPHOLOGIES, build_morphology
from src.utils.visualization import overlay_anomaly_map

INPUT_SHAPE = (96, 1024, 1)
DEFAULT_METHODS = ["recon", "cadence"]
MAD_SCALE = 1.4826
PREPROC_MODES = ("per_obs", "legacy_concat")

logger = logging.getLogger("inject_recover")


def setup_logging(out_dir: Path) -> None:
    """Tee all `log()` output to both stdout and <out_dir>/run.log."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(out_dir / "run.log", mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def log(msg: str = "") -> None:
    logger.info(msg)


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


def score_snippet(model, snippet, method, device, topk_frac=None):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    kwargs = {"topk_frac": topk_frac} if method == "topk" and topk_frac is not None else {}
    with torch.no_grad():
        s = model.anomaly_score(x, method=method, **kwargs)
    return float(s.item())


def reconstruct_snippet(model, snippet, device):
    x = torch.from_numpy(snippet).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze().cpu().numpy()


def normalize_frames(frames, preproc, tchans=96, mode="per_obs"):
    """Assemble per-observation windows into one preprocessed (tchans, fchans) frame.

    ``per_obs`` normalizes each observation on its own and *then* concatenates —
    byte-identical to ``scripts/inference.py::_preprocess_at`` and to
    ``torch_dataset``'s two ``__getitem__``s since commit ``4b5660c``
    (2026-07-09). This is the domain the production search actually runs in, so
    it is the default: a benchmark whose background lives in a different domain
    than the pipeline it is meant to characterise is not measuring that pipeline.

    ``legacy_concat`` is the pre-``4b5660c`` order — concatenate first, then
    normalize the whole 96-row block once. It leaves the real power step between
    observations in the frame and divides the whole cadence by one common MAD
    (see the ``gbt_fine_normalization_bug`` write-up). Every run under
    ``outputs/inject_recovery/`` before 2026-07-23 was measured this way, and so
    was the training of the frozen production checkpoint ``6d0d1ba``, so the mode
    is kept for reproducing those numbers — not because it is correct.
    """
    if mode not in PREPROC_MODES:
        raise ValueError(f"preproc_mode must be one of {PREPROC_MODES}, got '{mode}'")
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    if mode == "per_obs":
        normed = [
            core_transform(bandpass_correct(f, method=method, poly_degree=poly_degree),
                           mad_epsilon)
            for f in frames
        ]
        return np.concatenate(normed, axis=0)[:tchans, :]
    stacked = np.concatenate(frames, axis=0)[:tchans, :]
    stacked = bandpass_correct(stacked, method=method, poly_degree=poly_degree)
    return core_transform(stacked, mad_epsilon)


def preprocess_raw_window(obs_arrays, f_start, fchans, preproc, tchans=96, mode="per_obs"):
    """Slice a (tchans, fchans) window from loaded obs arrays and preprocess."""
    frames = [obs[:, f_start:f_start + fchans] for obs in obs_arrays]
    return normalize_frames(frames, preproc, tchans, mode)


def extract_obs_windows(obs_arrays, f_start, fchans, tchans_per_obs=16):
    """Slice the (n_obs, tchans_per_obs, fchans) raw window at f_start."""
    return np.stack([obs[:tchans_per_obs, f_start:f_start + fchans] for obs in obs_arrays])


def preprocess_injected(raw_snippet, preproc, tchans=96, mode="per_obs"):
    """Preprocess an injected raw snippet (n_obs, tchans_per_obs, fchans)."""
    return normalize_frames(list(raw_snippet), preproc, tchans, mode).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/inject_recovery")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=3)
    p.add_argument("--n_injections", type=int, default=20,
                   help="Injections per SNR level per cadence")
    p.add_argument("--n_background_probe", type=int, default=2000,
                   help="Random windows per cadence used to build the RFI-inclusive "
                        "background (also the pool from which quiet injection sites "
                        "are drawn). Keep >=500 (see sweep_baseline_sampling_guardrail: "
                        "n=200 undersamples the RFI tail and biases the threshold low).")
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[3, 5, 7, 10, 15, 20, 30, 50])
    p.add_argument("--morphology", default="narrowband_drift", choices=list(MORPHOLOGIES),
                   help="Signal class to inject. ONE per run, unlike "
                        "pipeline_sensitivity.py which sweeps all of them in a "
                        "single pass: this script's aggregation (eta^2, the "
                        "per-cadence tables, both CSVs, the plots) is indexed by "
                        "scoring method throughout, and the default keeps the "
                        "historical narrowband benchmark path bit-identical so "
                        "new runs stay comparable to the recorded "
                        "79.11/95.56/100.0 baseline. Outputs are stamped with "
                        "the morphology name, so runs do not overwrite each other.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preproc_mode", default="per_obs", choices=PREPROC_MODES,
                   help="Normalization order (see normalize_frames). 'per_obs' is "
                        "what the production search runs; 'legacy_concat' reproduces "
                        "every inject_recovery run before 2026-07-23. This choice "
                        "moves BOTH the background threshold and the injection-site "
                        "selection, so runs in different modes are not comparable.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Scoring methods to test (default: all supported by model)")
    p.add_argument("--topk_frac", type=float, default=None,
                   help="Override the model's configured topk_frac for method=topk "
                        "(e.g. 0.005 -> k~2/384 grid positions for UDMA). Default: "
                        "use the model's own config value.")
    return p.parse_args()


def main():
    args = parse_args()
    # One subdirectory per morphology. Without this a second run at a different
    # morphology silently overwrites the first's CSVs, plots and run.log, and
    # the two become indistinguishable after the fact — the same failure the
    # RESOLVED ARGS block below exists to prevent.
    args.out_dir = args.out_dir / args.morphology
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out_dir)

    # Every run is self-describing: without this, two runs whose background
    # probe happens to coincide (same checkpoint/model_config/seed, only a
    # downstream arg like topk_frac differing) are indistinguishable from a
    # stale-state bug when comparing logs after the fact.
    log(f"{'='*70}")
    log("RESOLVED ARGS")
    log(f"{'='*70}")
    for key, value in sorted(vars(args).items()):
        log(f"  {key}: {value}")
    log(f"{'='*70}")

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

    total_tchans = INPUT_SHAPE[0]

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    log(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    if args.methods is not None:
        methods = args.methods
    else:
        methods = []
        dummy = torch.zeros(1, 1, *INPUT_SHAPE[:2], device=args.device)
        for m in DEFAULT_METHODS:
            try:
                model.anomaly_score(dummy, method=m)
                methods.append(m)
            except (ValueError, AttributeError):
                log(f"  (skipping unsupported method '{m}' for this model)")
    log(f"  Methods: {methods}")
    if args.topk_frac is not None:
        log(f"  topk_frac override: {args.topk_frac}")

    # Collect results across all cadences
    all_results = {m: {snr: [] for snr in args.snr_list} for m in methods}
    # Background is built in-script from a full random probe of each cadence
    # (RFI-inclusive) — NOT from the quiet half used for injection sites.
    bg_scores = {m: [] for m in methods}
    # Per-cadence-tagged copies of the same scores (for the per-cadence-threshold
    # table + the between-vs-within-cadence variance decomposition below). Each
    # entry is one cadence's array; the two lists stay index-aligned because both
    # are appended once per successful cadence (a corrupt-file `continue` skips
    # both). bg_by_cad[m][k] and inj_by_cad[m][snr][k] refer to the same cadence k.
    bg_by_cad = {m: [] for m in methods}
    inj_by_cad = {m: {snr: [] for snr in args.snr_list} for m in methods}

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]
        log(f"\n{'='*70}")
        log(f"Cadence {cad_idx} ({len(obs_paths)} obs)")
        log(f"{'='*70}")

        obs_arrays = []
        try:
            for obs_path in obs_paths:
                arr = _load_full_obs(obs_path, downsample_factor)
                obs_arrays.append(arr)
        except OSError as e:
            log(f"  SKIPPING — corrupt file: {e}")
            del obs_arrays
            continue
        nchans = obs_arrays[0].shape[1]
        log(f"  Loaded {len(obs_arrays)} obs, nchans={nchans}")

        # Full random probe: builds the RFI-inclusive background AND
        # (via the recon score) identifies quiet windows for injection sites.
        n_probe = min(args.n_background_probe, nchans - fchans)
        probe_fstarts = rng.choice((nchans - fchans), size=n_probe, replace=False)
        probe_scores = {m: [] for m in methods}
        # Quiet-site selection below always needs a "recon" score, regardless
        # of which methods the user asked to benchmark (--methods topk alone
        # used to crash here with KeyError('recon') — 2026-07-16 fix): compute
        # it unconditionally, reusing the already-scored value when "recon" is
        # itself one of the requested methods instead of a redundant forward.
        need_recon_probe = "recon" not in methods
        recon_probe_list = [] if need_recon_probe else None
        for fs in probe_fstarts:
            snip = preprocess_raw_window(obs_arrays, fs, fchans, preproc,
                                         mode=args.preproc_mode)
            for m in methods:
                probe_scores[m].append(score_snippet(model, snip, m, args.device, args.topk_frac))
            if need_recon_probe:
                recon_probe_list.append(score_snippet(model, snip, "recon", args.device))
        probe_scores = {m: np.array(v) for m, v in probe_scores.items()}

        for m in methods:
            bg_scores[m].extend(probe_scores[m].tolist())
            bg_by_cad[m].append(probe_scores[m])

        # Use the quietest 50% (by recon score) as injection sites — recon is
        # the selection criterion regardless of --methods (see fix above): the
        # smoothest/most robust quiet-site detector (mean aggregation), kept
        # constant so injection-site selection doesn't vary with topk_frac
        # sweeps or other --methods choices.
        recon_probe = np.array(recon_probe_list) if need_recon_probe else probe_scores["recon"]
        quiet_mask = recon_probe <= np.median(recon_probe)
        quiet_fstarts = probe_fstarts[quiet_mask]
        log(f"  Probed {n_probe} windows (background), "
              f"{quiet_mask.sum()} quiet (recon <= {np.median(recon_probe):.4f})")

        injection_fstarts = rng.choice(quiet_fstarts,
                                        size=min(args.n_injections, len(quiet_fstarts)),
                                        replace=False)

        # One signal morphology (drift, width, profile) per injection site,
        # frozen and reused across the whole SNR sweep — only the amplitude
        # varies, so SNR is the sweep's one true independent variable.
        cad_inj = {m: {snr: [] for snr in args.snr_list} for m in methods}
        for j, fs in enumerate(injection_fstarts):
            site_seed = args.seed + cad_idx * 1000 + j
            injector = build_morphology(args.morphology, data_cfg, seed=site_seed)
            site = injector.sample_site(fchans, total_tchans)

            obs_windows = extract_obs_windows(obs_arrays, fs, fchans)

            for snr in args.snr_list:
                raw_inj, _ = injector.inject(obs_windows, site, snr)
                snip_inj = preprocess_injected(raw_inj, preproc, mode=args.preproc_mode)

                for m in methods:
                    s = score_snippet(model, snip_inj, m, args.device, args.topk_frac)
                    all_results[m][snr].append(s)
                    cad_inj[m][snr].append(s)

        for m in methods:
            for snr in args.snr_list:
                inj_by_cad[m][snr].append(np.array(cad_inj[m][snr]))

        del obs_arrays
        log(f"  Done cadence {cad_idx}")

    # ---- Build RFI-inclusive background from the aggregated probe ----
    log(f"\n{'='*70}")
    log(f"RFI-INCLUSIVE BACKGROUND (from {len(cadence_lines)} cadences)")
    log(f"{'='*70}")

    bg = {}
    for name in methods:
        scores = np.array(bg_scores[name])
        median, mad_sigma = robust_stats(scores)
        bg[name] = {"median": median, "mad_sigma": mad_sigma, "scores": scores,
                    "thresh_3s": median + 3 * mad_sigma,
                    "thresh_5s": median + 5 * mad_sigma}
        n_3s = (scores > bg[name]["thresh_3s"]).sum()
        log(f"  {name}: n={len(scores)}  median={median:.4f}  MAD_s={mad_sigma:.4f}  "
              f"3s={bg[name]['thresh_3s']:.4f} ({n_3s} candidates in probe)")

    # ---- Analysis against RFI-inclusive background ----
    log(f"\n{'='*70}")
    log(f"INJECTION RECOVERY vs RFI-INCLUSIVE BACKGROUND")
    log(f"{'='*70}")

    csv_rows = []

    for method in methods:
        b = bg[method]
        median, mad_sigma = b["median"], b["mad_sigma"]
        thresh_3 = b["thresh_3s"]
        thresh_5 = b["thresh_5s"]
        n_bg_candidates = (b["scores"] > thresh_3).sum()

        log(f"\n  {method} (bg: median={median:.4f}, MAD_s={mad_sigma:.4f}, "
              f"3s={thresh_3:.4f}, {n_bg_candidates} RFI candidates)")
        log(f"  {'SNR':>5s}  {'mean':>8s}  {'std':>8s}  {'sigma':>8s}  "
              f"{'det@3s':>8s}  {'det@5s':>8s}  {'rank%':>8s}")
        log(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        for snr in args.snr_list:
            scores = np.array(all_results[method][snr])
            mean_s = scores.mean()
            std_s = scores.std()
            sigma = (mean_s - median) / mad_sigma if mad_sigma > 0 else 0
            det_3 = (scores > thresh_3).mean() * 100
            det_5 = (scores > thresh_5).mean() * 100

            # Rank: what percentile of the background does the mean injection score fall at?
            rank_pct = (b["scores"] < mean_s).mean() * 100

            log(f"  {snr:5.0f}  {mean_s:8.4f}  {std_s:8.4f}  {sigma:8.2f}s  "
                  f"{det_3:7.1f}%  {det_5:7.1f}%  {rank_pct:7.2f}%")

            csv_rows.append({
                "method": method, "snr": snr,
                "mean_score": mean_s, "std_score": std_s,
                "sigma": sigma, "det_3s": det_3, "det_5s": det_5,
                "rank_pct": rank_pct,
            })

    # ---- Per-cadence threshold + variance decomposition (confound-1 test) ----
    # The pooled table above scores every injection against ONE global median+3σ,
    # merging cadences whose background RFI severity spans ~2 orders of magnitude.
    # Two questions the pooled table can't answer:
    #   (A) does a per-cadence threshold (each injection vs its OWN cadence's 3σ,
    #       the operationally honest metric — the real search runs per cadence)
    #       materially raise detection? If yes, the pooled threshold was the
    #       artifact depressing det@3σ.
    #   (B) is the huge injection-score spread (std ≈ 19× MAD at SNR50, so det@3σ
    #       is only 73% despite mean at +17σ) driven by BETWEEN-cadence variance
    #       (per-cadence threshold fixes it) or WITHIN-cadence variance (some
    #       injected signals produce almost no disagreement even at high SNR —
    #       a model/representation property no threshold can rescue)?
    #       η² = SS_between / SS_total: →1 between-dominated, →0 within-dominated.
    log(f"\n{'='*70}")
    log(f"PER-CADENCE THRESHOLD + VARIANCE DECOMPOSITION")
    log(f"{'='*70}")

    def eta_squared(groups):
        """Fraction of variance explained by cadence identity (one-way ANOVA η²).
        `groups` = list of per-cadence 1-D score arrays. Returns (eta2, n_total)."""
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return float("nan"), sum(len(g) for g in groups)
        all_x = np.concatenate(groups)
        grand = all_x.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        ss_total = ss_between + ss_within
        return (float(ss_between / ss_total) if ss_total > 0 else float("nan"),
                len(all_x))

    per_cad_rows = []

    for method in methods:
        # Each cadence's own robust threshold from its own background probe.
        cad_thresh3 = []
        cad_thresh5 = []
        for cad_bg in bg_by_cad[method]:
            med_c, mad_c = robust_stats(cad_bg)
            cad_thresh3.append(med_c + 3 * mad_c)
            cad_thresh5.append(med_c + 5 * mad_c)
        cad_thresh3 = np.array(cad_thresh3)
        cad_thresh3_min = float(cad_thresh3.min())
        cad_thresh3_median = float(np.median(cad_thresh3))
        cad_thresh3_max = float(cad_thresh3.max())

        log(f"\n  {method}: per-cadence 3σ thresholds — "
              f"min={cad_thresh3_min:.4f}, median={cad_thresh3_median:.4f}, "
              f"max={cad_thresh3_max:.4f}  (pooled 3σ={bg[method]['thresh_3s']:.4f})")
        log(f"  {'SNR':>5s}  {'det@3s_pool':>11s}  {'det@3s_cad':>11s}  "
              f"{'det@5s_cad':>11s}  {'eta2(cad)':>9s}")
        log(f"  {'-'*5}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*9}")
        for snr in args.snr_list:
            groups3 = inj_by_cad[method][snr]
            # pooled det@3σ (same as table above, recomputed for side-by-side)
            pooled_scores = np.array(all_results[method][snr])
            det_pool = (pooled_scores > bg[method]["thresh_3s"]).mean() * 100
            # per-cadence det@3σ / det@5σ: each cadence's injections vs its own thr
            hits3 = tot = hits5 = 0
            for ci, inj_arr in enumerate(groups3):
                if len(inj_arr) == 0:
                    continue
                hits3 += (inj_arr > cad_thresh3[ci]).sum()
                hits5 += (inj_arr > cad_thresh5[ci]).sum()
                tot += len(inj_arr)
            det_cad3 = 100 * hits3 / tot if tot else float("nan")
            det_cad5 = 100 * hits5 / tot if tot else float("nan")
            eta2, n_total = eta_squared(groups3)
            log(f"  {snr:5.0f}  {det_pool:10.1f}%  {det_cad3:10.1f}%  "
                  f"{det_cad5:10.1f}%  {eta2:9.3f}")

            per_cad_rows.append({
                "method": method, "snr": snr,
                "det_3s_pool": det_pool, "det_3s_cad": det_cad3, "det_5s_cad": det_cad5,
                "eta2_cad": eta2, "n_injections": n_total,
                "n_cadences": len(groups3),
                "cad_thresh3_min": cad_thresh3_min,
                "cad_thresh3_median": cad_thresh3_median,
                "cad_thresh3_max": cad_thresh3_max,
                "pooled_thresh3": bg[method]["thresh_3s"],
            })

        # High-SNR variance read: if η² stays low at SNR30/50, per-cadence
        # thresholding cannot rescue detection — the spread is within-cadence.
        hi = [s for s in args.snr_list if s >= 30]
        if hi:
            etas = [eta_squared(inj_by_cad[method][s])[0] for s in hi]
            mean_eta = np.nanmean(etas)
            if mean_eta > 0.5:
                verdict = ("BETWEEN-cadence variance dominates → per-cadence threshold "
                           "is the fix (pooled threshold was the artifact).")
            else:
                verdict = ("WITHIN-cadence variance dominates → per-cadence threshold "
                           "CANNOT rescue detection; some injections yield little "
                           "disagreement even at high SNR (model/representation, not "
                           "thresholding).")
            log(f"  → high-SNR (≥30) mean η² = {mean_eta:.3f}: {verdict}")

    # Save CSVs
    csv_path = args.out_dir / "inject_recovery_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    log(f"\nSaved -> {csv_path}")

    per_cad_csv_path = args.out_dir / "inject_recovery_per_cadence.csv"
    with open(per_cad_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_cad_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_cad_rows)
    log(f"Saved -> {per_cad_csv_path}")

    # ---- Plot 1: Detection rate vs SNR (head-to-head) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _base_colors = {"recon": "steelblue", "cadence": "crimson"}
    _cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {m: _base_colors.get(m, _cycle[i % len(_cycle)]) for i, m in enumerate(methods)}
    snrs = sorted(args.snr_list)

    # Left: score vs SNR
    for method in methods:
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
    for method in methods:
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

    m0 = methods[0]
    plt.suptitle(f"Injection Recovery — {sum(len(all_results[m0][s]) for s in snrs)} "
                 f"injections across {len(cadence_lines)} cadences", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inject_recovery_detection.png", dpi=150)
    plt.close()
    log(f"Saved -> {args.out_dir / 'inject_recovery_detection.png'}")

    # ---- Plot 2: Injection scores overlaid on background distribution ----
    fig, axes = plt.subplots(1, max(len(methods), 2), figsize=(7 * max(len(methods), 2), 5),
                             squeeze=False)
    for ax, method in zip(axes.flat, methods):
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
    log(f"Saved -> {args.out_dir / 'inject_vs_background.png'}")

    # ---- Plot 3: Example candidate plots for select SNR levels ----
    log("\nGenerating example injection plots...")
    example_dir = args.out_dir / "examples"
    example_dir.mkdir(exist_ok=True)

    # Reload first cadence for examples
    first_paths = [Path(p) for p in cadence_lines[0]]
    obs_arrays = [_load_full_obs(p, downsample_factor) for p in first_paths]
    nchans = obs_arrays[0].shape[1]

    probe_fstarts = rng.choice((nchans - fchans), size=50, replace=False)
    probe_scores = [score_snippet(model,
                    preprocess_raw_window(obs_arrays, fs, fchans, preproc,
                                          mode=args.preproc_mode),
                    "recon", args.device) for fs in probe_fstarts]
    quiet_fs = probe_fstarts[np.argmin(probe_scores)]

    example_injector = build_morphology(args.morphology, data_cfg, seed=args.seed + 9999)
    example_site = example_injector.sample_site(fchans, total_tchans)
    example_obs_windows = extract_obs_windows(obs_arrays, quiet_fs, fchans)

    for snr in [5, 10, 20]:
        raw_inj, _ = example_injector.inject(example_obs_windows, example_site, snr)
        snip_inj = preprocess_injected(raw_inj, preproc, mode=args.preproc_mode)
        snip_clean = preprocess_raw_window(obs_arrays, quiet_fs, fchans, preproc,
                                           mode=args.preproc_mode)
        try:
            recon_arr = reconstruct_snippet(model, snip_inj, args.device)
        except NotImplementedError:
            # UDMA has no pixel decoder — fall back to its native (nh,nw)
            # disagreement map for the illustrative panel below.
            recon_arr = None

        method_scores = {}
        for m in methods:
            method_scores[m] = score_snippet(model, snip_inj, m, args.device, args.topk_frac)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        vmin, vmax = np.percentile(snip_clean, [1, 99])

        # Top row: clean
        axes[0, 0].imshow(snip_clean, aspect="auto", origin="upper",
                          vmin=vmin, vmax=vmax, cmap="viridis")
        axes[0, 0].set_title("Clean original")
        axes[0, 0].set_ylabel("Time bin")

        # Bottom row: injected
        axes[1, 0].imshow(snip_inj, aspect="auto", origin="upper",
                          vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1, 0].set_title(f"Injected SNR={snr}")
        axes[1, 0].set_ylabel("Time bin")

        if recon_arr is not None:
            axes[1, 1].imshow(recon_arr, aspect="auto", origin="upper",
                              vmin=vmin, vmax=vmax, cmap="viridis")
            axes[1, 1].set_title("Reconstruction")

            error = np.abs(snip_inj - recon_arr)
            axes[1, 2].imshow(error, aspect="auto", origin="upper", cmap="hot")
            axes[1, 2].set_title("Residual")
        else:
            x_inj = torch.from_numpy(snip_inj).float().unsqueeze(0).unsqueeze(0).to(args.device)
            amap = model.anomaly_map(x_inj)[0].cpu().numpy()
            axes[1, 1].imshow(amap, aspect="auto", origin="upper", cmap="viridis")
            axes[1, 1].set_title("anomaly_map (UDMA, native (nh,nw) grid)")
            overlay_anomaly_map(axes[1, 2], snip_inj, amap,
                                title="anomaly_map (bilinear overlay)")

        diff = np.abs(snip_inj - snip_clean)
        axes[0, 1].imshow(diff, aspect="auto", origin="upper", cmap="hot")
        axes[0, 1].set_title("Injected - Clean (ground truth)")

        axes[0, 2].axis("off")
        info_lines = [f"SNR = {snr}"]
        for m in methods:
            info_lines.append(f"{m} score = {method_scores[m]:.4f}")
        info_lines.append("")
        for m in methods:
            if m in bg:
                info_lines.append(f"{m} 3s threshold = {bg[m]['thresh_3s']:.4f}")
        info_lines.append("")
        for m in methods:
            if m in bg:
                det = "YES" if method_scores[m] > bg[m]["thresh_3s"] else "NO"
                info_lines.append(f"{m} detected: {det}")
        axes[0, 2].text(0.1, 0.7, "\n".join(info_lines),
                        transform=axes[0, 2].transAxes, fontsize=12,
                        verticalalignment="top", fontfamily="monospace")

        f_mhz = quiet_fs * df / 1e6
        fig.suptitle(f"Injection example — SNR={snr}, f_start={quiet_fs} (~{f_mhz:.3f} MHz)",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(example_dir / f"injection_snr{snr:02d}.png", dpi=120, bbox_inches="tight")
        plt.close()

    del obs_arrays
    log(f"Saved example plots -> {example_dir}")
    log("\nDone.")


if __name__ == "__main__":
    main()
