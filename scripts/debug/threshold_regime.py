"""Which of the two pipeline cuts actually binds, per cadence?

Offline analysis over the ``maps.npz`` files written by
``scripts/recompute_anomaly_maps.py``. Reads no raw data, touches no HDD-backed
observation file, and runs happily on a *partial* recompute — every cadence
already written is usable.

Background. The search applies two cuts in series:

  stage 1  ``far_thresh``  = quantile(all scores, FAR_QUANTILE=0.99)
           a *relative* cut: "the most anomalous 1% of this cadence"
  stage 3  ``off_ceiling`` = max(off_noise_ceiling(OFF cells), thresh_5)
           an *absolute* cut, in anomaly-map units

They live in the same units (the scalar score is a topk reduction of the same
map), so they are directly comparable. ``scripts/pipeline_sensitivity.py``
found on a 2-cadence smoke test that they can disagree by 4x, and that when
``far_thresh`` wins, injected signals die at stage 1 *even at SNR 50* — the
ON/OFF logic never gets to see them. This script measures how common that is.

Two questions, one pass:

  Q1  Per cadence, does ``far_thresh`` sit above or below ``off_ceiling``?
      Above  -> the 1% pre-cut is the binding constraint (the "3C125 regime"),
                and real signals can be discarded before any ON/OFF reasoning.
      Below  -> the pre-cut is inert, ``off_ceiling`` decides everything
                (the regime the 364-CSV aggregation implied on average).

  Q2  Inside ``off_ceiling = max(off_noise_ceiling(probe), thresh_5)``, which
      term wins? ``thresh_5`` is a floor that was tuned on a single narrowband
      Voyager file; if it binds on most full-band cadences, a single-case
      threshold is governing the whole search
      (see docs/04_candidate-filtering.md §3.1).

Usage:
    python scripts/debug/threshold_regime.py \
        --map_dir outputs/inference/exotica_heldout_maps
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.search.candidates import off_noise_ceiling

OFF_ROWS = (1, 3, 5)


def _fuse(npz, weights, prefix: str) -> np.ndarray:
    w1, w2, w3 = weights
    return (w1 * npz[f"{prefix}st1"].astype(np.float32)
            + w2 * npz[f"{prefix}st2"].astype(np.float32)
            + w3 * npz[f"{prefix}ss"].astype(np.float32))


def _off_cells(cob: np.ndarray) -> np.ndarray:
    off_idx = [r for r in OFF_ROWS if r < cob.shape[1]]
    return cob[:, off_idx, :].ravel()


def ceilings(npz, weights) -> tuple[float, float]:
    """Raw OFF-noise ceiling under two pooling choices.

    Neither reproduces ``inference.py`` exactly, and the gap between them is
    the point. Production pools ~15 cluster maps + ``off_ceiling_probe``
    background maps, i.e. ~95% probe. The ``.npz`` holds a far larger selected
    set (everything above ``map_quantile``, ~5%), so pooling it in shifts the
    mix toward high-scoring, detection-biased snippets and biases the ceiling
    up. ``probe`` is the cleaner unbiased noise estimate and the closer
    analogue of production; ``pooled`` is what ``recover_off_ceiling.py``
    reports. If the two disagree materially, the ceiling is being set by the
    candidates it is supposed to be judging.
    """
    probe = _off_cells(_fuse(npz, weights, "probe_"))
    pooled = np.concatenate([_off_cells(_fuse(npz, weights, "")), probe])
    return off_noise_ceiling(probe), off_noise_ceiling(pooled)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map_dir", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs=3, default=(0.5, 0.5, 0.5),
                   help="score_weights the maps are fused with (production: 0.5 0.5 0.5)")
    p.add_argument("--out_csv", type=Path, default=None)
    args = p.parse_args()

    rows = []
    for npz_path in sorted(args.map_dir.glob("cad*/maps.npz")):
        label = npz_path.parent.name
        cad_idx = int(re.match(r"cad(\d+)", label).group(1))
        with np.load(npz_path) as z:
            if "probe_st1" not in z:
                continue
            probe_ceiling, pooled_ceiling = ceilings(z, args.weights)
            thresh_5 = float(z["thresh_5"])
            rows.append({
                "cad_idx": cad_idx,
                "label": label,
                "far_thresh": float(z["far_thresh"]),
                "probe_ceiling": probe_ceiling,
                "pooled_ceiling": pooled_ceiling,
                "thresh_5": thresh_5,
                "off_ceiling": max(probe_ceiling, thresh_5),
                "off_ceiling_pooled": max(pooled_ceiling, thresh_5),
                "n_snippets_total": int(z["n_snippets_total"]),
            })

    if not rows:
        raise SystemExit(f"No usable maps.npz found in {args.map_dir}")

    df = pd.DataFrame(rows).sort_values("cad_idx").reset_index(drop=True)
    df["far_over_ceiling"] = df["far_thresh"] / df["off_ceiling"]
    df["binding_cut"] = np.where(df["far_over_ceiling"] > 1.0, "FAR_1%", "off_ceiling")
    df["ceiling_term"] = np.where(df["thresh_5"] > df["probe_ceiling"],
                                  "thresh_5 (Voyager floor)", "OFF noise")

    n = len(df)
    print(f"\n{'=' * 78}\nTHRESHOLD REGIME — {n} cadences with maps "
          f"(recompute may still be running)\n{'=' * 78}")

    print("\nQ1  Which cut binds?")
    for name, k in df["binding_cut"].value_counts().items():
        print(f"  {name:<14} {k:4d}  ({100 * k / n:5.1f}%)")

    r = df["far_over_ceiling"]
    print("\n    far_thresh / off_ceiling  (>1 = the 1% pre-cut is the gate)")
    print("      min {:.2f}   p25 {:.2f}   median {:.2f}   p75 {:.2f}   "
          "p90 {:.2f}   max {:.2f}".format(
              r.min(), *np.quantile(r, [0.25, 0.5, 0.75, 0.9]), r.max()))
    severe = (r > 2.0).sum()
    print(f"      cadences where FAR sits >2x above the ceiling: {severe} "
          f"({100 * severe / n:.1f}%)  <- signals lost before ON/OFF")

    print("\nQ2  Inside max(OFF noise, thresh_5), which term wins?")
    for name, k in df["ceiling_term"].value_counts().items():
        print(f"  {name:<26} {k:4d}  ({100 * k / n:5.1f}%)")

    infl = df["pooled_ceiling"] / df["probe_ceiling"]
    print("\n    Pooling sensitivity: pooled_ceiling / probe_ceiling")
    print("      median {:.3f}   p90 {:.3f}   max {:.3f}   "
          "(>>1 = the ceiling is set by the candidates it should judge)".format(
              infl.median(), np.quantile(infl, 0.9), infl.max()))
    flipped = (df["binding_cut"].values
               != np.where(df["far_thresh"] > df["off_ceiling_pooled"],
                           "FAR_1%", "off_ceiling"))
    print(f"      cadences whose Q1 verdict flips under the other pooling: "
          f"{flipped.sum()} / {n}")

    print("\nWorst offenders (highest far/ceiling):")
    cols = ["cad_idx", "label", "far_thresh", "off_ceiling",
            "far_over_ceiling", "ceiling_term"]
    print(df.nlargest(10, "far_over_ceiling")[cols].to_string(index=False))

    out = args.out_csv or args.map_dir.parent / "threshold_regime.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
