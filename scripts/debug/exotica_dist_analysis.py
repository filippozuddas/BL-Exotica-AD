"""
Deep distributional analysis of the real Exotica .0000.h5 batch, run on the
data host (has blimpy). Targets three dataset-construction decisions:

  1. Normalization: does bandpass_correct(poly_degree=3) + core_transform
     (log1p + median/MAD) hold up on real noise/RFI, or does arcsinh /
     something else become necessary?
  2. RFI characterization per band: occupancy fraction, intensity vs noise,
     static-vs-transient -- the thing that dominates recon-MSE.
  3. Band/cadence completeness -> usable training budget, and ON vs OFF
     pixel-distribution comparison -> is training-on-everything safe.

Samples a handful of files per band-setup (not the whole ~5400-file batch).
Prints tables to stdout and saves CSVs/a PNG under data/processed/dist_analysis/.

Run on the machine where DATA_DIR is mounted (see notebooks/04), e.g.:
    python3 scripts/debug/exotica_dist_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"pyproject.toml not found above {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scan_headers import cluster_by_gap, infer_on_off  # noqa: E402
from src.data.preprocessing import bandpass_correct, core_transform  # noqa: E402

import blimpy  # noqa: E402

DATA_DIR = Path("/content/wd_mybook")
HEADERS_CSV = REPO_ROOT / "data" / "processed" / "exotica_0000_headers.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "dist_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GAP_THRESHOLD_S = 700
SUBBAND_CHANNELS = 4096
N_SUBBANDS_PER_FILE = 3  # low edge, center, high edge
N_BANDS_TO_SAMPLE = 8

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_subbands(file_path, n_channels=SUBBAND_CHANNELS, n_positions=N_SUBBANDS_PER_FILE):
    """Load a few narrow frequency windows spanning the file's covered band."""
    wf_hdr = blimpy.Waterfall(str(file_path), load_data=False)
    fch1 = wf_hdr.header["fch1"]
    foff = wf_hdr.header["foff"]
    nchans = wf_hdr.header["nchans"]
    f_lo = min(fch1, fch1 + foff * nchans)
    f_hi = max(fch1, fch1 + foff * nchans)
    span = f_hi - f_lo
    frac_positions = np.linspace(0.1, 0.9, n_positions)
    frames = []
    for frac in frac_positions:
        f_center = f_lo + frac * span
        half_bw = n_channels * abs(foff) / 2
        f_start = f_center - half_bw
        f_stop = f_center + half_bw
        wf = blimpy.Waterfall(str(file_path), f_start=f_start, f_stop=f_stop)
        data = wf.data.squeeze()
        if data.ndim == 1:
            data = data[np.newaxis, :]
        frames.append(("%.0f%%" % (frac * 100), data.astype(np.float32)))
    return frames


def main():
    df = pd.read_csv(HEADERS_CSV)
    good = df[df["nchans"].notna()].sort_values("abs_time_s").reset_index(drop=True)

    # --- header assumption check -------------------------------------------------
    print("=== 0. Header assumption check ===")
    print("nchans unique:", sorted(good["nchans"].unique()))
    print("ntime unique :", sorted(good["ntime"].unique()))
    print("df (foff_Hz) stats:\n", good["foff_Hz"].describe())
    print("dt (tsamp_s) stats:\n", good["tsamp_s"].describe())
    print("duration_s stats:\n", good["duration_s"].describe())

    # --- cadence / band-setup grouping --------------------------------------------
    rows_sorted = good.to_dict("records")
    clusters = cluster_by_gap(rows_sorted, GAP_THRESHOLD_S)

    def rounded_band(row):
        return (round(row["f_start_MHz"], 0), round(row["f_stop_MHz"], 0))

    session_band_rows = []
    for session_idx, cluster in enumerate(clusters, 1):
        band_groups = defaultdict(list)
        for r in cluster:
            band_groups[rounded_band(r)].append(r)
        for band, rows_in_band in band_groups.items():
            oi = infer_on_off(rows_in_band)
            session_band_rows.append({
                "session": session_idx, "band": band, "n_files": len(rows_in_band),
                "rows": rows_in_band, "on_target": oi["on_target"],
            })

    session_band_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "rows"} for r in session_band_rows]
    )

    print("\n=== 3. Band / cadence completeness (training budget) ===")
    complete = session_band_df[session_band_df["n_files"] == 6]
    by_band = complete.groupby("band").size().sort_values(ascending=False)
    print(f"{len(complete)}/{len(session_band_df)} session-band entries are complete (6 files)")
    print("Complete session-bands per band (top 15):")
    print(by_band.head(15))
    by_band.to_csv(OUT_DIR / "complete_cadences_per_band.csv")

    # --- pick representative sample: 1 complete session-band per distinct band,
    #     covering both ON and OFF files -------------------------------------------
    seen_bands = set()
    sample_entries = []
    for entry in session_band_rows:
        if entry["n_files"] != 6 or entry["band"] in seen_bands:
            continue
        seen_bands.add(entry["band"])
        oi = infer_on_off(entry["rows"])
        on_row = next((r for r in entry["rows"] if r["target"] == oi["on_target"]), None)
        off_row = next((r for r in entry["rows"] if r["target"] != oi["on_target"]), None)
        if on_row and off_row:
            sample_entries.append({"band": entry["band"], "on": on_row, "off": off_row})
        if len(sample_entries) >= N_BANDS_TO_SAMPLE:
            break

    print(f"\nSelected {len(sample_entries)} bands for deep sampling: "
          f"{[e['band'] for e in sample_entries]}")

    # --- 1 & 2: raw / bandpass-corrected / core-transformed distributions --------
    stats_rows = []
    raw_pixel_samples = {"ON": [], "OFF": []}
    transformed_pixel_samples = {"ON": [], "OFF": []}

    for entry in sample_entries:
        band = entry["band"]
        for role in ("on", "off"):
            row = entry[role]
            try:
                subbands = load_subbands(row["file"])
            except Exception as e:
                print(f"  [skip] {row['file']}: {e}")
                continue
            for pos, frame in subbands:
                corrected = bandpass_correct(frame, method="polynomial", poly_degree=3)
                transformed = core_transform(corrected)

                # RFI channel flag: per-channel temporal median > 5-sigma (robust) above baseline
                chan_med = np.median(frame.astype(np.float64), axis=0)
                base = np.median(chan_med)
                mad = np.median(np.abs(chan_med - base)) + 1e-10
                rfi_mask = np.abs(chan_med - base) > 5 * 1.4826 * mad
                rfi_frac = rfi_mask.mean()
                rfi_intensity_ratio = (
                    (chan_med[rfi_mask].mean() / base) if rfi_mask.any() else np.nan
                )

                stats_rows.append({
                    "band": f"{band[0]:.0f}-{band[1]:.0f}", "role": role.upper(),
                    "target": row["target"], "subband_pos": pos,
                    "raw_min": float(frame.min()), "raw_max": float(frame.max()),
                    "raw_median": float(np.median(frame)),
                    "rfi_frac": float(rfi_frac),
                    "rfi_intensity_ratio": float(rfi_intensity_ratio),
                    "post_bp_min": float(corrected.min()), "post_bp_max": float(corrected.max()),
                    "post_transform_min": float(transformed.min()),
                    "post_transform_max": float(transformed.max()),
                    "post_transform_p99": float(np.percentile(transformed, 99)),
                    "post_transform_p01": float(np.percentile(transformed, 1)),
                })
                raw_pixel_samples[role.upper()].append(frame.ravel())
                transformed_pixel_samples[role.upper()].append(transformed.ravel())

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_DIR / "subband_stats.csv", index=False)

    print("\n=== 1&2. Normalization + RFI stats (per sub-band sample) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(stats_df.describe())
        print("\nRFI fraction by role:\n", stats_df.groupby("role")["rfi_frac"].describe())
        print("\npost_transform_max by role (RFI blow-up check):\n",
              stats_df.groupby("role")["post_transform_max"].describe())

    # --- ON vs OFF comparison ------------------------------------------------------
    print("\n=== ON vs OFF pixel-distribution comparison ===")
    from scipy import stats as sstats
    on_concat = np.concatenate(raw_pixel_samples["ON"]) if raw_pixel_samples["ON"] else np.array([])
    off_concat = np.concatenate(raw_pixel_samples["OFF"]) if raw_pixel_samples["OFF"] else np.array([])
    if on_concat.size and off_concat.size:
        # subsample for KS test tractability
        rng = np.random.default_rng(42)
        n = min(200_000, on_concat.size, off_concat.size)
        ks = sstats.ks_2samp(rng.choice(on_concat, n, replace=False),
                             rng.choice(off_concat, n, replace=False))
        print(f"raw pixel KS test ON vs OFF: stat={ks.statistic:.4f} p={ks.pvalue:.3g}")
        print(f"ON  median={np.median(on_concat):.4g} OFF median={np.median(off_concat):.4g}")

    # --- plots ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for role, color in [("ON", "tab:orange"), ("OFF", "tab:blue")]:
        if transformed_pixel_samples[role]:
            sample = np.concatenate(transformed_pixel_samples[role])
            sub = np.random.default_rng(0).choice(sample, min(500_000, sample.size), replace=False)
            axes[0].hist(sub, bins=200, alpha=0.5, label=role, color=color, density=True)
    axes[0].set_title("post core_transform pixel distribution")
    axes[0].set_xlabel("normalized value")
    axes[0].legend()

    axes[1].hist(stats_df[stats_df.role == "ON"]["rfi_frac"], bins=20, alpha=0.5, label="ON", color="tab:orange")
    axes[1].hist(stats_df[stats_df.role == "OFF"]["rfi_frac"], bins=20, alpha=0.5, label="OFF", color="tab:blue")
    axes[1].set_title("RFI channel fraction per sub-band (>5σ robust)")
    axes[1].set_xlabel("fraction of channels flagged RFI")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "distributions_summary.png", dpi=130)
    print(f"\nSaved plot -> {OUT_DIR / 'distributions_summary.png'}")

    summary = {
        "n_complete_session_bands": int(len(complete)),
        "n_session_band_entries": int(len(session_band_df)),
        "bands_sampled": [f"{b[0]:.0f}-{b[1]:.0f}" for b in seen_bands],
        "post_transform_max_overall": float(stats_df["post_transform_max"].max()),
        "post_transform_min_overall": float(stats_df["post_transform_min"].min()),
        "rfi_frac_mean_ON": float(stats_df[stats_df.role == "ON"]["rfi_frac"].mean()),
        "rfi_frac_mean_OFF": float(stats_df[stats_df.role == "OFF"]["rfi_frac"].mean()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
