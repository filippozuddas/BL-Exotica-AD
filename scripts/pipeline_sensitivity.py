"""
End-to-end pipeline sensitivity: how many injected signals reach a human's eyes?

Every injection-recovery benchmark in this project so far
(``scripts/inject_recover.py``, the topk_frac sweeps, the teacher comparisons)
measures the **scorer**: does the injected snippet's scalar score clear a
robust sigma threshold? That is not the question the search result rests on.
Between the score and a plot a human actually looks at sit four more stages,
each of which can silently drop a real signal:

    score  ->  FAR 1% cut  ->  frequency-adjacency clustering
           ->  OFF-noise ceiling + full_row_hits short list

This script injects ON-only narrowband signals into real heldout cadences and
reports the survival fraction at **every** stage, so the search can be written
up with a sensitivity statement ("complete to X% at SNR Y") rather than an
unquantified absence of detections.

The plot cap is reported too, but as a *rank distribution* rather
than a filter: it is a review-budget knob (how many plots to render), not a
pipeline decision, and can simply be raised. What the rank tells us is how
much human review budget a given completeness costs.

It is deliberately cheap. It does not re-scan anything: the background score
pool for each cadence comes from the completed run's ``inference_scores.csv``
(via the ``_scores`` cache built by ``scripts/recompute_anomaly_maps.py``), and
the OFF-noise ceiling comes from that script's saved probe maps. The only work
per cadence is one HDD load plus a few hundred forward passes on injected
windows, so the whole sweep is hours, not the ~30 h a real rescan costs.

Two stages deserve a note on why they are measured rather than assumed:

* **The FAR 1% cut runs before any ON/OFF reasoning.** It keeps the top 1% of
  ~131k snippets by fused score, and that score is known to rank *persistent*
  signals above ON-only ones (occupancy AUC 0.36-0.40, see
  ``udma_component_attribution_result`` in memory). So the pipeline's first
  irreversible cut is structurally biased against exactly the class of signal
  the search targets. ``--loose_quantile`` re-runs the same cascade at a looser
  cut so the cost of the 1% choice is a measured number.
* **The short list is where the real decision happens.** Everything above it is
  a threshold on the model's own score; ``full_row_hits`` is the only stage
  that reasons about ON vs OFF, and by the attribution result it carries 100%
  of the ETI-vs-RFI discrimination. If a stage is going to lose signals, this
  is the one to watch.

Usage (run on the server, AFTER recompute_anomaly_maps.py has finished):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/BL-Exotica-AD \
    python scripts/pipeline_sensitivity.py \
        --checkpoint outputs/training/20260707_093113_6d0d1ba/checkpoints/epoch=057-val_loss=0.2065.ckpt \
        --cadence_list data/raw/gbt_0000_heldout_cadences.txt \
        --run_dir outputs/inference/exotica_heldout_topk001 \
        --map_dir outputs/inference/exotica_heldout_maps \
        --model_config configs/model/udma_old_teacher.yaml \
        --out_dir outputs/pipeline_sensitivity/v1 \
        --max_cadences 20 --n_sites 25
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
from scripts.inject_recover import extract_obs_windows, preprocess_injected
from src.data.torch_dataset import _load_full_obs
from src.data.morphologies import MORPHOLOGIES, build_morphology
from src.search.candidates import off_noise_ceiling, on_off_contrast, full_row_hits

INPUT_SHAPE = (96, 1024, 1)
OFF_ROWS = (1, 3, 5)


def ceiling_from_probe(probe_maps: dict, weights, thresh_5: float) -> float:
    """Reproduce ``scripts/inference.py``'s OFF-noise ceiling from saved maps.

    Uses only the cadence-wide random probe, not the candidate clusters the
    pipeline also pooled in: the cluster contribution is a small,
    detection-biased sample (its OFF cells are co-located in frequency with
    already-flagged events), and the saved probe here is 10x larger than the
    one the original run could afford. The ``max(..., thresh_5)`` floor is kept
    exactly as the pipeline applies it.
    """
    w1, w2, w3 = weights
    cob = (w1 * probe_maps["st1"].astype(np.float32)
           + w2 * probe_maps["st2"].astype(np.float32)
           + w3 * probe_maps["ss"].astype(np.float32))
    off_idx = [r for r in OFF_ROWS if r < cob.shape[1]]
    return max(off_noise_ceiling(cob[:, off_idx, :].ravel()), float(thresh_5))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Completed inference run (for the per-cadence candidate CSVs)")
    p.add_argument("--map_dir", type=Path, required=True,
                   help="Output of recompute_anomaly_maps.py (probe maps + score cache)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path,
                   default=ROOT / "configs/model/udma_old_teacher.yaml")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=20)
    p.add_argument("--n_sites", type=int, default=25,
                   help="Injection sites per cadence, drawn from the quiet half of "
                        "that cadence's own score distribution.")
    p.add_argument("--snr_list", type=float, nargs="+",
                   default=[10, 15, 20, 30, 50])
    p.add_argument("--rank_caps", type=int, nargs="+", default=[30, 50, 100, 200],
                   help="Review-budget levels to report completeness at. The plot cap "
                        "is not a filter — it only decides how many plots get rendered "
                        "and can be raised freely — so it is reported as a rank "
                        "distribution: how many plots per cadence buy how much "
                        "completeness.")
    p.add_argument("--loose_quantile", type=float, default=0.95,
                   help="Alternative (looser) FAR cut, to price the 0.99 choice.")
    p.add_argument("--morphologies", nargs="+", default=list(MORPHOLOGIES),
                   choices=list(MORPHOLOGIES),
                   help="Signal classes to sweep. All morphologies share each "
                        "injection site and its background, so the comparison "
                        "between them is paired.")
    p.add_argument("--method", default="topk")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    downsample_factor = frame.get("downsample_factor", 1)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    cadence_lines = [line.strip().split()
                     for line in args.cadence_list.read_text().splitlines()
                     if line.strip()]

    print(f"Loading model from {args.checkpoint}")
    model = inf.load_model(args.checkpoint, model_cfg, INPUT_SHAPE, args.device)
    weights = model.score_weights
    print(f"  score_weights={weights}  topk_frac={model.topk_frac}")

    print(f"  morphologies: {', '.join(args.morphologies)}")

    score_cache = args.map_dir / "_scores"
    rows = []
    n_done = 0

    for cad_idx, obs_paths in enumerate(cadence_lines):
        if n_done >= args.max_cadences:
            break
        obs_paths = [Path(p) for p in obs_paths]

        meta = None
        for obs_path in obs_paths:
            try:
                meta = inf.read_cadence_meta(obs_path)
                break
            except OSError:
                continue
        if meta is None:
            continue
        cad_dirname = inf.make_cadence_dirname(cad_idx, meta)

        maps_path = args.map_dir / cad_dirname / "maps.npz"
        score_path = score_cache / f"cad{cad_idx:02d}.npz"
        cand_path = args.run_dir / cad_dirname / f"{args.method}_candidates.csv"
        if not (maps_path.exists() and score_path.exists() and cand_path.exists()):
            print(f"Cadence {cad_idx}: missing inputs, skipping")
            continue

        npz = np.load(maps_path)
        sc = np.load(score_path)
        all_f, all_s = sc["f_start"], sc["score"]

        thresh_3 = float(npz["thresh_3"])
        thresh_5 = float(npz["thresh_5"])
        far_thresh = float(npz["far_thresh"])
        loose_thresh = float(np.quantile(all_s, args.loose_quantile))
        off_ceiling = ceiling_from_probe(
            {k: npz[f"probe_{k}"] for k in ("st1", "st2", "ss")}, weights, thresh_5)

        # An injection is ordered against the cadence's *existing* short list,
        # so the incumbent contrasts are what set its rank — and therefore how
        # many plots would have to be rendered for it to be seen.
        cand = pd.read_csv(cand_path)
        if "in_short_list" in cand.columns:
            incumbent = cand.loc[cand["in_short_list"].astype(bool),
                                 "on_off_contrast"].to_numpy()
        else:
            incumbent = np.array([])
        incumbent = np.sort(incumbent)[::-1]

        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx}: {meta['source']}  {meta['fch1_mhz']:.1f} MHz")
        print(f"  far(1%)={far_thresh:.4f}  loose({args.loose_quantile})={loose_thresh:.4f}  "
              f"3s={thresh_3:.4f}  off_ceiling={off_ceiling:.4f}")
        print(f"  incumbent short list: {len(incumbent)} candidates")

        t_load = time.time()
        obs_arrays = []
        try:
            for obs_path in obs_paths:
                obs_arrays.append(_load_full_obs(obs_path, downsample_factor))
        except OSError as e:
            print(f"  SKIPPING — corrupt file: {e}")
            continue
        print(f"  Loaded in {time.time()-t_load:.1f}s")

        # Inject only into windows this cadence itself scores as quiet, so the
        # measurement is "can a signal be found in ordinary sky", not "can it be
        # found on top of RFI that would have been flagged anyway".
        quiet_f = all_f[all_s <= np.median(all_s)]
        sites = rng.choice(quiet_f, size=min(args.n_sites, len(quiet_f)), replace=False)

        # The fix proposed in docs/06_threshold-audit.md. Free to evaluate here:
        # it is a second comparison against the same scalar score, no extra
        # forward pass. Reported alongside the shipped cut rather than behind a
        # flag, because a single-arm result cannot distinguish "the model did
        # not see the signal" from "the pre-cut discarded it" — and on 32.7% of
        # cadences far_thresh sits above off_ceiling (max 162x), so that
        # distinction decides how the whole sweep reads.
        min_thresh = min(far_thresh, off_ceiling)

        for site_i, f_start in enumerate(sites):
            f_start = int(f_start)
            # The site (and therefore its background) is drawn ONCE and reused by
            # every morphology, so morphologies are compared paired: the between-
            # cadence and between-site variance that dominates this measurement
            # (eta^2 ~ 0.94, see udma_heldout_injection_recovery) cancels in the
            # comparison instead of being sampled independently per class.
            raw = extract_obs_windows(obs_arrays, f_start, fchans)

            for m_idx, morph_name in enumerate(args.morphologies):
                # Deterministic per (cadence, site, morphology): reproducible and
                # independent of how many sites/SNRs/morphologies ran before it.
                site_seed = args.seed + 1000 * cad_idx + site_i + 100000 * m_idx
                injector = build_morphology(morph_name, data_cfg, seed=site_seed)
                site = injector.sample_site(fchans, INPUT_SHAPE[0])

                for snr in args.snr_list:
                    injected, inj_info = injector.inject(
                        raw, site, snr, on_indices=(0, 2, 4))
                    snip = preprocess_injected(injected, preproc)
                    x = torch.from_numpy(snip).float().unsqueeze(0).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        score = float(model.anomaly_score(x, method=args.method).item())
                        amap = model.anomaly_map(x)[0].cpu().numpy()

                    fr = full_row_hits(amap, threshold=off_ceiling)
                    contrast = on_off_contrast(amap, threshold=off_ceiling)["on_off_contrast"]

                    pass_far = score > far_thresh
                    pass_min = score > min_thresh
                    pass_loose = score > loose_thresh
                    pass_rows = fr["n_on_hits_full"] >= 2
                    pass_short = bool(fr["in_short_list"])
                    survives = pass_short and pass_far
                    survives_min = pass_short and pass_min
                    # Where the injection would land in this cadence's plot ordering.
                    # Not a filter — it prices review budget: rank r means it needs
                    # a cap of r+1 to be rendered.
                    rank = int((incumbent > contrast).sum()) if survives else -1

                    rows.append({
                        "cadence_idx": cad_idx, "target": meta["source"],
                        "morphology": morph_name,
                        "site": site_i, "f_start": f_start, "snr": snr,
                        "drift_rate": inj_info.get("drift_rate"),
                        "min_thresh": min_thresh,
                        "s1_min": bool(pass_min),
                        "survives_pipeline_min": bool(survives_min),
                        "score": score, "on_off_contrast": contrast,
                        "n_on_hits_full": fr["n_on_hits_full"],
                        "n_off_hits_full": fr["n_off_hits_full"],
                        "far_thresh": far_thresh, "loose_thresh": loose_thresh,
                        "thresh_3": thresh_3, "off_ceiling": off_ceiling,
                        "rank_in_shortlist": rank,
                        "n_incumbent": len(incumbent),
                        "s0_sigma3": bool(score > thresh_3),
                        "s1_far1pct": bool(pass_far),
                        "s1b_loose": bool(pass_loose),
                        "s2_on_rows": bool(pass_rows),
                        "s3_short_list": bool(pass_short),
                        "survives_pipeline": bool(survives),
                        **{f"sig_{k}": v for k, v in inj_info.items()
                           if k not in ("snr", "drift_rate")},
                    })

        del obs_arrays
        n_done += 1

        df = pd.DataFrame(rows)
        df.to_csv(args.out_dir / "pipeline_sensitivity.csv", index=False)

    # ---- Survival cascade ----
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo results.")
        return

    stages = ["s0_sigma3", "s1_far1pct", "s1b_loose", "s1_min", "s2_on_rows",
              "s3_short_list", "survives_pipeline", "survives_pipeline_min"]
    labels = {
        "s0_sigma3":         "score > 3sigma (scorer only, historical metric)",
        "s1_far1pct":        "survives FAR 1% cut          [pipeline stage 1]",
        "s1b_loose":         f"  ...would survive FAR {100*(1-args.loose_quantile):.0f}% cut",
        "s1_min":            "  ...would survive min(far,ceiling)  [docs/06 fix]",
        "s2_on_rows":        ">=2 ON rows over OFF ceiling [pipeline stage 2]",
        "s3_short_list":     "in short list (no off_leak)  [pipeline stage 3]",
        "survives_pipeline": "SURVIVES THE PIPELINE        [as shipped]",
        "survives_pipeline_min": "SURVIVES THE PIPELINE        [with docs/06 fix]",
    }
    summary = df.groupby("snr")[stages].mean() * 100

    def table(title, index, cols, fmt="{:>7.1f}"):
        print(f"\n{'='*78}\n{title}\n{'='*78}")
        print(f"{'':<48}  " + " ".join(f"{c:>7}" for c in cols))
        for label, vals in index:
            print(f"{label:<48}  " + " ".join(fmt.format(v) for v in vals))

    n_per_snr = len(df) // len(args.snr_list)
    table(f"END-TO-END SURVIVAL (%)   n={n_per_snr} injections/SNR, {n_done} cadences"
          f"  [all morphologies pooled]",
          [(labels[st], [summary.loc[s, st] for s in summary.index]) for st in stages],
          [f"SNR{int(s)}" for s in summary.index])

    # Per morphology. Pooling hides the thing the sweep exists to measure: the
    # thesis is that this search is sensitive to signal classes turboSETI is
    # not, which is a statement about the SPREAD between these rows, not about
    # their average.
    for morph in df["morphology"].unique():
        sub = df[df.morphology == morph]
        msum = sub.groupby("snr")[stages].mean() * 100
        n_m = len(sub) // max(len(args.snr_list), 1)
        table(f"MORPHOLOGY: {morph}   n={n_m} injections/SNR",
              [(labels[st], [msum.loc[s, st] for s in msum.index]) for st in stages],
              [f"SNR{int(s)}" for s in msum.index])

    # The gap the two arms open is the cost of the shipped pre-cut, per class.
    table("COST OF THE FAR 1% PRE-CUT (percentage points recovered by docs/06 fix)",
          [(f"  {m:<44}",
            [(df[(df.morphology == m) & (df.snr == s)]["survives_pipeline_min"].mean()
              - df[(df.morphology == m) & (df.snr == s)]["survives_pipeline"].mean()) * 100
             for s in summary.index])
           for m in df["morphology"].unique()],
          [f"SNR{int(s)}" for s in summary.index])

    # Review budget: of the injections that survive the pipeline, how many are
    # rendered at a given plot cap. Denominator is all injections, so this reads
    # as end-to-end completeness including the review step.
    surv = df[df["survives_pipeline"]]
    rank_rows = []
    for cap in args.rank_caps:
        vals = []
        for s in summary.index:
            sub = df[df.snr == s]
            hit = surv[(surv.snr == s) & (surv.rank_in_shortlist < cap)]
            vals.append(100 * len(hit) / max(len(sub), 1))
        rank_rows.append((f"completeness at plot cap {cap:>4}", vals))
    table("REVIEW BUDGET — completeness (%) vs plots rendered per cadence",
          rank_rows, [f"SNR{int(s)}" for s in summary.index])

    if len(surv):
        q = surv.groupby("snr")["rank_in_shortlist"].quantile([0.5, 0.9, 0.99]).unstack()
        table("RANK OF SURVIVING INJECTIONS within the cadence's short list",
              [(f"  {int(100*p)}th percentile", [q.loc[s, p] for s in q.index])
               for p in (0.5, 0.9, 0.99)],
              [f"SNR{int(s)}" for s in q.index], fmt="{:>7.0f}")

    summary.to_csv(args.out_dir / "survival_summary.csv")
    print(f"\nSaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
