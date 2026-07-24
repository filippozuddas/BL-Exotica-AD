"""Make the off_leak mechanism visible: UDMA anomaly maps, morphology x SNR.

Companion to `udma_anomaly_maps.py` (which inspects one narrowband injection at
one SNR from the preprocessed cache). This one injects ALL registered
morphologies into ONE shared quiet site of a real cadence, across an SNR sweep,
and lays the fused (6,64) disagreement maps out as a grid so the reason
end-to-end completeness peaks at SNR30 and falls at SNR50 is legible rather than
inferred from a CSV.

What to look for (measured 2026-07-23, see
`.scratch/test-harness-unification/results-log.md` and the
`udma_off_leak_short_list_filter` memory):

  - The map is (6, 64): one ROW per observation. ON = rows 0/2/4 (signal
    injected there), OFF = rows 1/3/5 (byte-identical raw data — nothing is
    injected). A hot OFF row is therefore response the MODEL generated from ON
    content, never anything carried by the data.
  - `narrowband_drift` / `narrowband_accel`: response concentrates on the ON
    rows in a tight column; OFF rows stay dark at every SNR. These survive.
  - `narrowband_sine` / `wideband_pulsed`: the response is diffuse, and as SNR
    rises the OFF rows light up too. When >=2 OFF rows clear the per-cadence
    off_ceiling (and reach `leak_frac` of the weakest ON row), `full_row_hits`
    flags `off_leak` and the short list DROPS the candidate — a red box marks
    each leaking OFF row, and the cell title reads LEAK.

So the stronger the signal, the more likely the diffuse morphologies are thrown
away. The one stage carrying all the ETI-vs-RFI discrimination misreads the
project's target class as RFI.

This is purely diagnostic — no AUC, no pass/fail. It reuses the production
scoring path (same off_ceiling construction as `scripts/inference.py`, same
`full_row_hits` rule) so what the picture shows is what the pipeline does.

Usage (server, AFTER commit+push so --preproc_mode exists):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/BL-Exotica-AD \
    python scripts/debug/morphology_anomaly_maps.py \
        --checkpoint outputs/training/20260707_093113_6d0d1ba/checkpoints/epoch=057-val_loss=0.2065.ckpt \
        --cadence_list data/raw/gbt_0000_heldout_cadences.txt \
        --cad_idx 0 \
        --model_config configs/model/udma_old_teacher.yaml \
        --out_dir outputs/sweeps/morphology_anomaly_maps
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.inference as inf
from scripts.inject_recover import (
    extract_obs_windows, preprocess_injected, preprocess_raw_window, robust_stats,
)
from src.data.torch_dataset import _load_full_obs
from src.data.morphologies import MORPHOLOGIES, build_morphology
from src.search.candidates import off_noise_ceiling
from src.utils.visualization import add_obs_dividers, overlay_anomaly_map

INPUT_SHAPE = (96, 1024, 1)
ON_ROWS = (0, 2, 4)
OFF_ROWS = (1, 3, 5)
LEAK_FRAC = 0.3  # must match src/search/candidates.py::full_row_hits default


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path,
                   default=ROOT / "data/raw/gbt_0000_heldout_cadences.txt")
    p.add_argument("--cad_idx", type=int, default=0,
                   help="Which cadence in the list to draw the shared quiet site from.")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path,
                   default=ROOT / "configs/model/udma_old_teacher.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/morphology_anomaly_maps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--snr_list", type=float, nargs="+", default=[0, 15, 30, 50])
    p.add_argument("--morphologies", nargs="+", default=list(MORPHOLOGIES),
                   choices=list(MORPHOLOGIES))
    p.add_argument("--n_probe", type=int, default=400,
                   help="Random windows used to build the off_ceiling and pick the "
                        "quiet site. Same role as inference's probe, smaller because "
                        "this is one illustrative cadence, not a benchmark.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--preproc_mode", default="per_obs",
                   help="Must match the production search (per_obs). legacy_concat "
                        "would let ON injections shift the shared block stats into "
                        "OFF rows via a DATA path, masking the model-generated leak "
                        "this figure exists to show.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def anomaly_maps_batched(model, snippets, device, batch_size):
    """Fused (N,6,64) maps for a list of (96,1024) snippets."""
    out = []
    for i in range(0, len(snippets), batch_size):
        chunk = np.asarray(snippets[i:i + batch_size], dtype=np.float32)
        x = torch.from_numpy(chunk).unsqueeze(1).to(device)  # (b,1,96,1024)
        out.append(model.anomaly_map(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def topk_scores(maps, topk_frac):
    """Per-map top-k mean, matching UDMA.anomaly_score(method='topk')."""
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, int(round(topk_frac * flat.shape[1])))
    part = np.partition(flat, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def leaking_off_rows(amap, off_ceiling):
    """Which OFF rows count as leak hits under the full_row_hits rule.

    Reproduces src/search/candidates.py::full_row_hits inline so the figure does
    not depend on that function also returning the row peaks (a separate,
    later commit). Returns (list of leaking OFF row indices, off_leak bool,
    n_on_hits, on_off_contrast).
    """
    on_max = amap[list(ON_ROWS), :].max(axis=1)
    off_max = amap[list(OFF_ROWS), :].max(axis=1)
    n_on_hits = int((on_max > off_ceiling).sum())
    on_ref = float(on_max.min())
    leak_mask = (off_max > off_ceiling) & (off_max >= LEAK_FRAC * on_ref)
    leaking = [OFF_ROWS[i] for i in range(len(OFF_ROWS)) if leak_mask[i]]
    off_leak = leak_mask.sum() >= 2
    # ON/OFF row-max ratio — the quantity the leak gate actually keys on (a
    # signal is dropped once OFF rows reach LEAK_FRAC of the weakest ON row).
    # NOT src.search.candidates.on_off_contrast, which reduces inside a shared
    # col-window; that one drives plot ranking, this one drives short-listing.
    ratio = on_max.mean() / off_max.mean() if off_max.mean() > 0 else float("inf")
    return leaking, off_leak, n_on_hits, ratio


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("RESOLVED ARGS")
    print("=" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 70)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    fchans = data_cfg["frame"]["fchans"]
    downsample_factor = data_cfg["frame"].get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    cadence_lines = [line.strip().split()
                     for line in args.cadence_list.read_text().splitlines()
                     if line.strip() and not line.strip().startswith("#")]
    obs_paths = [Path(p) for p in cadence_lines[args.cad_idx]]

    meta = None
    for p in obs_paths:
        try:
            meta = inf.read_cadence_meta(p)
            break
        except OSError:
            continue
    source = meta["source"] if meta else f"cad{args.cad_idx}"

    print(f"\nLoading model from {args.checkpoint}")
    model = inf.load_model(args.checkpoint, model_cfg, INPUT_SHAPE, args.device)
    topk_frac = model.topk_frac
    print(f"  topk_frac={topk_frac}  score_weights={model.score_weights}")

    print(f"\nLoading cadence {args.cad_idx} ({source})")
    obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]
    nchans = obs_arrays[0].shape[1]

    # ---- Probe: off_ceiling (as inference derives it) + quiet-site pick ----
    n_probe = min(args.n_probe, nchans - fchans)
    probe_fstarts = rng.choice(nchans - fchans, size=n_probe, replace=False)
    probe_snips = [preprocess_raw_window(obs_arrays, int(fs), fchans, preproc,
                                         mode=args.preproc_mode)
                   for fs in probe_fstarts]
    probe_maps = anomaly_maps_batched(model, probe_snips, args.device, args.batch_size)
    probe_scores = topk_scores(probe_maps, topk_frac)

    median, mad_sigma = robust_stats(probe_scores)
    thresh_5 = median + 5 * mad_sigma
    off_cells = probe_maps[:, list(OFF_ROWS), :].ravel()
    off_ceiling = max(off_noise_ceiling(off_cells), float(thresh_5))
    print(f"  probe n={n_probe}  off_ceiling={off_ceiling:.4f} "
          f"(noise_core={off_noise_ceiling(off_cells):.4f}, thresh_5={thresh_5:.4f})")

    quiet_fs = int(probe_fstarts[np.argmin(probe_scores)])
    print(f"  shared quiet site: f_start={quiet_fs} (~{quiet_fs*df/1e6:.3f} MHz), "
          f"topk={probe_scores.min():.4f}")

    raw_site = extract_obs_windows(obs_arrays, quiet_fs, fchans)  # (6,16,1024)

    # ---- Inject every morphology x SNR at that one site ----
    morphs = args.morphologies
    snrs = sorted(args.snr_list)
    amaps = {}       # (morph, snr) -> (6,64)
    snippets = {}    # (morph, snr) -> (96,1024) injected spectrogram
    info = {}        # (morph, snr) -> dict
    for mi, name in enumerate(morphs):
        injector = build_morphology(name, data_cfg, seed=args.seed + 100000 * mi)
        site = injector.sample_site(fchans, INPUT_SHAPE[0])
        for snr in snrs:
            injected, _ = injector.inject(raw_site, site, snr, on_indices=ON_ROWS)
            snip = preprocess_injected(injected, preproc, mode=args.preproc_mode)
            x = torch.from_numpy(snip).float().unsqueeze(0).unsqueeze(0).to(args.device)
            amap = model.anomaly_map(x)[0].cpu().numpy()
            score = topk_scores(amap[None], topk_frac)[0]
            leaking, off_leak, n_on, ratio = leaking_off_rows(amap, off_ceiling)
            amaps[(name, snr)] = amap
            snippets[(name, snr)] = snip
            info[(name, snr)] = dict(score=score, leaking=leaking, off_leak=off_leak,
                                     n_on=n_on, ratio=ratio)

    del obs_arrays

    # Shared colour scale across the whole grid so intensities are comparable
    # cell-to-cell. Anchored on the injected cells only (SNR>0); SNR=0 stays dark.
    hot = np.concatenate([amaps[(m, s)].ravel() for m in morphs for s in snrs if s > 0])
    vmax = float(np.percentile(hot, 99)) if hot.size else 1.0

    # ---- Figure 1: the grid of anomaly maps ----
    nrow, ncol = len(morphs), len(snrs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow),
                             squeeze=False)
    for mi, name in enumerate(morphs):
        for si, snr in enumerate(snrs):
            ax = axes[mi][si]
            amap = amaps[(name, snr)]
            im = ax.imshow(amap, aspect="auto", origin="upper", cmap="inferno",
                           vmin=0, vmax=vmax, interpolation="nearest")
            add_obs_dividers(ax, n_rows=6, n_obs=6, color="white", lw=0.5, alpha=0.4)
            d = info[(name, snr)]
            for r in d["leaking"]:
                ax.add_patch(Rectangle((-0.5, r - 0.5), 64, 1, fill=False,
                                       edgecolor="cyan", lw=2.0))
            tag = "  LEAK" if d["off_leak"] else ""
            ax.set_title(f"SNR {snr:g} | score {d['score']:.3f} | "
                         f"ON/OFF {d['ratio']:.1f}{tag}",
                         fontsize=8, color=("red" if d["off_leak"] else "black"))
            ax.set_yticks(list(ON_ROWS) + list(OFF_ROWS))
            ax.set_yticklabels(["ON", "ON", "ON", "OFF", "OFF", "OFF"], fontsize=6)
            ax.set_xticks([])
            if si == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"UDMA fused anomaly maps (6x64) — one quiet site of {source}, "
        f"off_ceiling={off_ceiling:.3f}\n"
        f"ON rows carry the injection; a hot OFF row is model-generated. "
        f"Cyan box = OFF-leak hit; LEAK = candidate dropped from short list.",
        fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("fused disagreement (map_cob)", fontsize=8)
    out1 = args.out_dir / f"anomaly_map_grid_cad{args.cad_idx:02d}_{source}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out1}")

    # ---- Figure 2: injected spectrogram | map overlay, per morphology at max SNR ----
    snr_hi = snrs[-1]
    fig2, axes2 = plt.subplots(len(morphs), 2, figsize=(9, 2.6 * len(morphs)),
                               squeeze=False)
    for mi, name in enumerate(morphs):
        snip = snippets[(name, snr_hi)]
        amap = amaps[(name, snr_hi)]
        vmn, vmx = np.percentile(snip, [1, 99])
        ax0 = axes2[mi][0]
        ax0.imshow(snip, aspect="auto", origin="upper", cmap="viridis",
                   vmin=vmn, vmax=vmx)
        add_obs_dividers(ax0, n_rows=96, n_obs=6)
        ax0.set_ylabel(name, fontsize=9, fontweight="bold")
        ax0.set_title(f"injected spectrogram (SNR {snr_hi:g})", fontsize=8)
        ax0.set_xticks([]); ax0.set_yticks([])
        overlay_anomaly_map(axes2[mi][1], snip, amap,
                            title=f"map overlay | {'LEAK' if info[(name, snr_hi)]['off_leak'] else 'kept'}")
        axes2[mi][1].set_xticks([]); axes2[mi][1].set_yticks([])
    fig2.suptitle(f"Injected signal vs anomaly map at SNR {snr_hi:g} — {source}",
                  fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = args.out_dir / f"spectrogram_vs_map_cad{args.cad_idx:02d}_{source}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {out2}")

    # ---- Console summary table ----
    print(f"\n{'morphology':<18} {'SNR':>5} {'score':>7} {'n_on':>5} "
          f"{'onoff_r':>9} {'off_leak':>9} {'leaking_rows':>14}")
    for name in morphs:
        for snr in snrs:
            d = info[(name, snr)]
            print(f"{name:<18} {snr:>5g} {d['score']:>7.3f} {d['n_on']:>5d} "
                  f"{d['ratio']:>9.2f} {str(d['off_leak']):>9} "
                  f"{str(d['leaking']):>14}")
    print("\nDone.")


if __name__ == "__main__":
    main()
