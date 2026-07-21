"""Qualitative UDMA anomaly-map inspection (Q9 "characterization without bar",
docs/02_udma-architecture.md) — the last, non-gating item of the UDMA
eval checklist (step 6).

Plots the input spectrogram next to the four disagreement maps
(``map_st1``, ``map_st2``, ``map_ss``, fused ``map_cob``) on three examples:

  - ``noise`` : a quiet real site, no injection.
  - ``rfi``   : a real-RFI-rich site (highest hot-fraction in the sampled pool).
  - ``eti``   : a quiet site with a synthetic narrowband drifting signal
                injected ON-only across the cadence (blocks 0,2,4), at
                ``--snr`` (default 25 — the regime where every prior scorer,
                including the λ3 probe, clearly detects).

This is purely visual/diagnostic (no AUC, no pass/fail) — the read is whether
the maps localise on the injected/real line or smear across the whole (6,64)
grid via the teacher's global attention (the Q1 risk: "se le mappe risultano
diffuse -> v2: teacher CNN a receptive field piccolo").

Usage (server):
    PYTHONPATH=/path/to/BL-Exotica-AD python scripts/debug/udma_anomaly_maps.py \\
        --checkpoint outputs/20260705_224358_e8411e9/checkpoints/epoch=029-val_loss=0.6381.ckpt \\
        --model_config configs/model/udma.yaml \\
        --cache /path/to/data/processed/cache_gbt_fine \\
        --out_dir outputs/sweeps/udma_anomaly_maps
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import NarrowbandParams, NarrowbandDriftingGenerator
from scripts.debug.injection_vs_rfi_test import preprocess_raw
from scripts.debug.encode_separation_test import load_model
from src.utils.visualization import overlay_anomaly_map

INPUT_SHAPE = (96, 1024, 1)
MAP_KEYS = ("st1", "st2", "ss", "cob")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/udma.yaml")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n_probe", type=int, default=300, help="Sites sampled to pick noise/RFI/injection candidates")
    p.add_argument("--snr", type=float, default=25.0, help="Injected ETI SNR")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/udma_anomaly_maps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def _maps(model, frame: np.ndarray, device: str) -> dict:
    x = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    components = model.anomaly_map_components(x)
    return {k: v[0].cpu().numpy() for k, v in components.items()}


def main():
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    fchans = data_cfg["frame"]["fchans"]
    total_tchans = INPUT_SHAPE[0]
    nb_params = NarrowbandParams.from_config(data_cfg)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    if model_cfg.get("architecture") != "udma":
        raise SystemExit(f"--model_config must be architecture: udma, got '{model_cfg.get('architecture')}'.")

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device, require_encode=False)

    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    idx_pool = rng.choice(arr.shape[0], size=min(args.n_probe, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[idx_pool])
    del arr

    frames_pre = [preprocess_raw(raw_pool[i], preproc) for i in range(len(raw_pool))]
    hot_fracs = np.array([float((f > 5.0).sum()) / f.size for f in frames_pre])
    order = np.argsort(hot_fracs)
    quiet_idx = order[0]                 # lowest hot-fraction -> "noise"
    rfi_idx = order[-1]                  # highest hot-fraction -> "rfi"
    eti_site_idx = order[1]              # second-quietest, distinct from the noise example

    print(f"  noise site: hot_frac={hot_fracs[quiet_idx]:.4f}   "
          f"rfi site: hot_frac={hot_fracs[rfi_idx]:.4f}   "
          f"eti site: hot_frac={hot_fracs[eti_site_idx]:.4f}")

    examples = {
        "noise": frames_pre[quiet_idx],
        "rfi": frames_pre[rfi_idx],
    }

    gen = NarrowbandDriftingGenerator(nb_params, seed=args.seed)
    drift_rate, start_channel, f_profile, t_profile_builder, meta = \
        gen.sample_cadence_signal_params(fchans, total_tchans)
    eti_raw, inj_info = gen.inject_on_only_cadence(
        raw_pool[eti_site_idx], snr=args.snr, drift_rate=drift_rate, start_channel=start_channel,
        f_profile=f_profile, t_profile_builder=t_profile_builder, on_indices=(0, 2, 4),
    )
    print(f"  injected ETI: snr={args.snr}, drift_rate={drift_rate:.3f} Hz/s, "
          f"start_channel={start_channel}, on_indices={inj_info['on_indices']}")
    examples["eti"] = preprocess_raw(eti_raw, preproc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 6, figsize=(26, 12))
    col_titles = ["Input", "map_st1 (AE gap)", "map_st2 (MemAE gap)", "map_ss (disagreement)",
                  "map_cob (fused)", "map_cob (bilinear overlay)"]
    saved = {}
    for row, (name, frame) in enumerate(examples.items()):
        maps = _maps(model, frame, args.device)
        saved[name] = {"frame": frame, **maps}
        panels = [frame] + [maps[k] for k in MAP_KEYS]
        for col, (img, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            im = ax.imshow(img, aspect="auto", origin="upper", interpolation="nearest")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{name}\ntime bin")
            ax.set_xlabel("freq channel" if col == 0 else "freq patch col")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        overlay_ax = axes[row, 5]
        overlay_anomaly_map(overlay_ax, frame, maps["cob"], origin="upper")
        if row == 0:
            overlay_ax.set_title(col_titles[5])
        overlay_ax.set_xlabel("freq channel")

    fig.suptitle("UDMA anomaly maps — noise / rfi / eti (qualitative, Q1 smearing check)")
    fig.tight_layout()
    fig_path = args.out_dir / f"udma_anomaly_maps_{args.checkpoint.stem}.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    npz_path = args.out_dir / f"udma_anomaly_maps_{args.checkpoint.stem}.npz"
    flat = {f"{name}_{k}": v for name, d in saved.items() for k, v in d.items()}
    np.savez(npz_path, **flat)
    print(f"Saved arrays -> {npz_path}")


if __name__ == "__main__":
    main()
