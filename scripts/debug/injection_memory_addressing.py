"""Which MemAE memory slot addresses an injected narrowband-drift signal, vs SNR?

Follow-up to ``visualize_memae_memory.py``: the ``--rank bottom`` patch gallery
showed several low-usage slots (e.g. 372, 489, 314, 352, 152, 139, 181, 470,
283 on the 2026-07-07 run) whose real-data crops look like drifting narrowband
lines — i.e. the memory has (rarely) learned narrowband-drift RFI as a "normal"
prototype. This script tests the direct operational consequence: when a
setigen-injected narrowband-drift signal (same injector as
``scripts/inject_recover.py``) is run through the encoder, which slot does its
footprint actually address, and does that change with SNR? If injected signals
route to the same drift-like prototypes at recoverable SNR, that is a concrete
mechanism for why recon-based scoring under-recovers narrowband drift (the
decoder can redraw the injected line via a learned "this is normal" prototype
instead of erroring on it) — reinforcing the rfi_outscores_eti_max_error
finding with a direct causal link through the memory, not just a
correlation between smoothness and low error.

Method per injection site x SNR:
1. Inject (reusing ``scripts.inject_recover``'s exact generator/preprocessing).
2. Locate the signal's footprint in the bottleneck grid from the injected-vs-
   clean preprocessed difference (NOT from the drift-rate geometry — the
   absolute-cadence-timeline drift used by ``inject_on_only_cadence`` makes
   that fragile; diffing the two preprocessed frames is exact and simple).
   Only ``on_indices`` grid rows (h = observation index; the encoder's 16x
   time downsampling maps 1:1 onto GBT's 16-bin-per-obs blocks) can contain
   the signal.
3. Per ON row, take the bottleneck column with the largest pooled diff, and
   only count it as a located footprint if that cell's diff clears the row's
   background variation by a margin (``--footprint_ratio``) — at low SNR the
   injected signal may not be locatable at all, and forcing an argmax onto
   pure noise would fabricate a spurious "hit".
4. Record which slot wins the addressing at each located footprint cell.

Usage:
    python scripts/debug/injection_memory_addressing.py \
        --checkpoint outputs/.../best_model.ckpt \
        --model_config configs/model/memae.yaml \
        --cadence_list data/processed/inject_recovery_cadences.txt \
        --snr_list 3 5 7 10 15 20 30 50 --n_injections 20
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.inject_recover import (
    INPUT_SHAPE, extract_obs_windows, load_model, preprocess_injected,
    preprocess_raw_window, score_snippet,
)
from src.data.synthetic import NarrowbandDriftingGenerator, NarrowbandParams
from src.data.torch_dataset import _load_full_obs
from scripts.debug.visualize_memae_memory import bottleneck_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True,
                    help="Must have memory: true (e.g. configs/model/memae.yaml).")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--cadence_list", type=Path, required=True)
    p.add_argument("--max_cadences", type=int, default=3)
    p.add_argument("--n_injections", type=int, default=20)
    p.add_argument("--n_background_probe", type=int, default=500)
    p.add_argument("--snr_list", type=float, nargs="+",
                    default=[3, 5, 7, 10, 15, 20, 30, 50])
    p.add_argument("--footprint_ratio", type=float, default=3.0,
                    help="Minimum ratio of the candidate footprint cell's pooled diff to the "
                         "median pooled diff elsewhere in the same row, to count the footprint "
                         "as located rather than an arbitrary pick on noise.")
    p.add_argument("--slots_of_interest", type=int, nargs="+",
                    default=[372, 489, 314, 352, 152, 139, 181, 470, 283],
                    help="Drift-like low-usage slots identified visually by "
                         "visualize_memae_memory.py --rank bottom, to check for overlap.")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/injection_memory_addressing")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pooled_diff(diff, grid_hw):
    """Block-average ``diff`` (H_in, W_in) down to the bottleneck grid (gh, gw)."""
    gh, gw = grid_hw
    H_in, W_in = diff.shape
    fh, fw = H_in // gh, W_in // gw
    return diff[:fh * gh, :fw * gw].reshape(gh, fh, gw, fw).mean(axis=(1, 3))


def locate_footprint(clean, injected, on_indices, grid_hw, ratio_thresh):
    """For each ON row, find the bottleneck column with the strongest injected-vs-
    clean pooled difference; keep it only if it clears the row's background by
    ``ratio_thresh``. Returns a list of (h, w) located footprint cells.
    """
    diff = np.abs(injected - clean)
    pooled = pooled_diff(diff, grid_hw)  # (gh, gw)
    located = []
    for h in on_indices:
        row = pooled[h]
        w_star = int(np.argmax(row))
        rest = np.delete(row, w_star)
        bg = np.median(rest) if rest.size else 0.0
        if bg <= 0 or row[w_star] / bg >= ratio_thresh:
            located.append((h, w_star))
    return located


def winning_slots(model, frame, device):
    x = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encoder(x)
        _, att = model.memory(z)
    h_p, w_p = z.shape[-2], z.shape[-1]
    return att.argmax(dim=1).cpu().numpy().reshape(h_p, w_p)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    if not model_cfg.get("memory", False):
        raise ValueError(
            f"{args.model_config} does not have memory: true — this script needs the "
            f"standalone MemAE (Gong et al.), not a plain AE/MAE/VAE."
        )
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame_cfg = data_cfg["frame"]
    fchans = frame_cfg["fchans"]
    downsample_factor = frame_cfg.get("downsample_factor", 1)

    nb_params = NarrowbandParams.from_config(data_cfg)
    total_tchans = INPUT_SHAPE[0]

    print(f"Loading MemAE from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)
    grid_hw = bottleneck_grid(model, INPUT_SHAPE, args.device)
    print(f"Bottleneck grid: {grid_hw}")

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    # slot_by_snr[snr] -> Counter(slot -> count), over all located footprints.
    slot_by_snr = {snr: Counter() for snr in args.snr_list}
    n_located = {snr: 0 for snr in args.snr_list}
    n_total = {snr: 0 for snr in args.snr_list}

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]
        print(f"\nCadence {cad_idx} ({len(obs_paths)} obs)")
        try:
            obs_arrays = [_load_full_obs(p, downsample_factor) for p in obs_paths]
        except OSError as e:
            print(f"  SKIPPING — corrupt file: {e}")
            continue
        nchans = obs_arrays[0].shape[1]

        n_probe = min(args.n_background_probe, nchans - fchans)
        probe_fstarts = rng.choice((nchans - fchans), size=n_probe, replace=False)
        recon_probe = np.array([
            score_snippet(model, preprocess_raw_window(obs_arrays, fs, fchans, preproc),
                          "recon", args.device)
            for fs in probe_fstarts
        ])
        quiet_fstarts = probe_fstarts[recon_probe <= np.median(recon_probe)]
        injection_fstarts = rng.choice(quiet_fstarts,
                                        size=min(args.n_injections, len(quiet_fstarts)),
                                        replace=False)

        for j, fs in enumerate(injection_fstarts):
            site_seed = args.seed + cad_idx * 1000 + j
            gen = NarrowbandDriftingGenerator(nb_params, seed=site_seed)
            drift_rate, start_channel, f_profile, t_profile_builder, _ = \
                gen.sample_cadence_signal_params(fchans, total_tchans)

            obs_windows = extract_obs_windows(obs_arrays, fs, fchans)
            clean_frame = preprocess_injected(obs_windows, preproc)

            for snr in args.snr_list:
                raw_inj, info = gen.inject_on_only_cadence(
                    obs_windows, snr=snr, drift_rate=drift_rate,
                    start_channel=start_channel, f_profile=f_profile,
                    t_profile_builder=t_profile_builder,
                )
                injected_frame = preprocess_injected(raw_inj, preproc)

                located = locate_footprint(clean_frame, injected_frame, info["on_indices"],
                                            grid_hw, args.footprint_ratio)
                n_total[snr] += len(info["on_indices"])
                n_located[snr] += len(located)
                if not located:
                    continue
                winners = winning_slots(model, injected_frame, args.device)
                for h, w in located:
                    slot_by_snr[snr][int(winners[h, w])] += 1

        del obs_arrays
        print(f"  Done cadence {cad_idx}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    interest = set(args.slots_of_interest)

    print(f"\n{'='*78}")
    print(f"{'SNR':>5s}  {'located/total':>14s}  {'top slot (count)':>20s}  "
          f"{'overlap w/ interest set':>26s}")
    print(f"{'-'*5}  {'-'*14}  {'-'*20}  {'-'*26}")
    rows = []
    for snr in args.snr_list:
        counter = slot_by_snr[snr]
        located_frac = f"{n_located[snr]}/{n_total[snr]}"
        if counter:
            top_slot, top_count = counter.most_common(1)[0]
            n_hits_interest = sum(c for s, c in counter.items() if s in interest)
            n_hits_total = sum(counter.values())
            overlap_pct = 100.0 * n_hits_interest / n_hits_total if n_hits_total else 0.0
            print(f"{snr:5.0f}  {located_frac:>14s}  {f'{top_slot} ({top_count})':>20s}  "
                  f"{overlap_pct:24.1f}%")
        else:
            print(f"{snr:5.0f}  {located_frac:>14s}  {'(no located footprints)':>20s}  {'--':>26s}")
        rows.append((snr, counter, n_located[snr], n_total[snr]))

    # Plot: for each SNR, stacked bar of slot hit distribution (top 10 slots + "other"),
    # marking bars that are in the interest set.
    all_snrs = [r[0] for r in rows if sum(r[1].values()) > 0]
    if all_snrs:
        fig, ax = plt.subplots(figsize=(10, 5))
        top_slots_overall = [s for s, _ in Counter(
            {k: v for r in rows for k, v in r[1].items()}).most_common(10)]
        bottoms = np.zeros(len(all_snrs))
        for slot in top_slots_overall:
            heights = np.array([slot_by_snr[snr].get(slot, 0) for snr in all_snrs], dtype=float)
            label = f"slot {slot}" + (" *interest*" if slot in interest else "")
            ax.bar([str(s) for s in all_snrs], heights, bottom=bottoms, label=label)
            bottoms += heights
        ax.set_xlabel("Injection SNR")
        ax.set_ylabel("Located footprint count")
        ax.set_title("Which memory slot addresses the injected signal, vs SNR\n"
                      "(*interest* = drift-like slots found by visualize_memae_memory.py --rank bottom)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        out_path = args.out_dir / "injection_slot_addressing_vs_snr.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved → {out_path}")
    else:
        print("\nNo located footprints at any SNR — nothing to plot. Consider lowering "
              "--footprint_ratio or raising --snr_list.")


if __name__ == "__main__":
    main()
