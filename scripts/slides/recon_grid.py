"""
Slide visualization: 3×3 grid of (input | reconstruction | error) for
clean / naturally-RFI-contaminated / ETI-injected snippets from the real cache.

Produces one PNG per model backbone (AE, ViT-MAE).

Usage:
    python scripts/slides/recon_grid.py \\
        --cache     /path/to/cache_gbt_fine \\
        --ae_checkpoint     outputs/<run>/checkpoints/best.ckpt \\
        --vitmae_checkpoint outputs/<run>/checkpoints/best.ckpt \\
        --snr_eti 25 --drift_rate 0.3 \\
        --out_dir outputs/slides

Snippet selection:
  clean  — quietest snippet found (lowest fraction of pixels > 5 sigma)
  rfi    — most RFI-contaminated snippet (highest hot_frac), no injection
  eti    — second-quietest snippet + injected narrowband drifting signal
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

from src.data.preprocessing import bandpass_correct, core_transform
from src.data.synthetic import NarrowbandDriftingGenerator, NarrowbandParams
from src.models.autoencoder import build_autoencoder


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _load_model(ckpt_path: Path, model_cfg: dict, device: str) -> torch.nn.Module:
    model = build_autoencoder((96, 1024, 1), model_cfg, loss="mse")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    return model.eval().to(device)


# --------------------------------------------------------------------------- #
# Cache I/O and preprocessing
# --------------------------------------------------------------------------- #

def _preprocess(raw_obs: np.ndarray, preproc_cfg: dict) -> np.ndarray:
    """(n_obs, tchans_per_obs, fchans) → preprocessed (tchans, fchans) float32.

    Mirrors CachedDataset.__getitem__: concatenate observations, then apply
    bandpass_correct + core_transform on the full cadence frame.
    """
    frame = np.concatenate(raw_obs, axis=0)   # (tchans, fchans)
    frame = bandpass_correct(frame,
                             method=preproc_cfg.get("bandpass_method", "polynomial"),
                             poly_degree=preproc_cfg.get("poly_degree", 3))
    return core_transform(frame, preproc_cfg.get("mad_epsilon", 1e-6)).astype(np.float32)


def _hot_frac(snippet: np.ndarray, sigma: float = 5.0) -> float:
    """Fraction of pixels exceeding sigma in a preprocessed (tchans, fchans) frame."""
    return float((snippet > sigma).sum()) / snippet.size


def select_snippets(
    cache_path: Path,
    split: str,
    preproc_cfg: dict,
    n_scan: int = 500,
    seed: int = 42,
) -> dict:
    """Scan n_scan random snippets and return raw arrays for three roles.

    Returns:
        {
          "clean":  (n_obs, tchans_per_obs, fchans) raw — lowest hot_frac
          "clean2": same shape — second-lowest hot_frac (used for ETI injection)
          "rfi":    same shape — highest hot_frac
        }
    """
    npy = cache_path / f"{split}.npy"
    arr = np.load(str(npy), mmap_mode="r")
    n_total = arr.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n_scan, n_total), replace=False)

    print(f"  Scanning {len(idx)} snippets from {npy.name} …")
    hot_fracs = []
    for i in idx:
        snippet = _preprocess(arr[i], preproc_cfg)
        hot_fracs.append(_hot_frac(snippet))

    order = np.argsort(hot_fracs)
    clean_i  = int(idx[order[0]])
    clean2_i = int(idx[order[1]])
    rfi_i    = int(idx[order[-1]])

    print(f"  clean  [{clean_i}]  hot_frac={hot_fracs[order[0]]:.4f}")
    print(f"  clean2 [{clean2_i}]  hot_frac={hot_fracs[order[1]]:.4f}  (→ ETI base)")
    print(f"  rfi    [{rfi_i}]  hot_frac={hot_fracs[order[-1]]:.4f}")

    return {
        "clean":  np.array(arr[clean_i]),
        "clean2": np.array(arr[clean2_i]),
        "rfi":    np.array(arr[rfi_i]),
    }


# --------------------------------------------------------------------------- #
# ETI injection on raw concatenated frame
# --------------------------------------------------------------------------- #

def _inject_eti(
    raw_obs: np.ndarray,
    params: NarrowbandParams,
    preproc_cfg: dict,
    snr: float,
    drift_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a narrowband drifting signal, return (raw_injected, snippet_injected).

    Injection is done on the concatenated raw cadence frame so the signal
    drifts continuously across all observations.
    """
    raw_cat = np.concatenate(raw_obs, axis=0).astype(float)   # (tchans, fchans)
    gen = NarrowbandDriftingGenerator(params, seed=seed)
    raw_inj, info = gen.inject_signal(raw_cat, snr=snr, drift_rate=drift_rate)
    print(f"  ETI injection: snr={info['snr']:.1f}  drift={info['drift_rate']:.3f} Hz/s  "
          f"chan={info['start_channel']}  f_profile={info['f_profile']}")
    snippet = bandpass_correct(raw_inj.astype(np.float32),
                               method=preproc_cfg.get("bandpass_method", "polynomial"),
                               poly_degree=preproc_cfg.get("poly_degree", 3))
    snippet = core_transform(snippet, preproc_cfg.get("mad_epsilon", 1e-6)).astype(np.float32)
    return raw_inj, snippet


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def _reconstruct(model: torch.nn.Module, snippet: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(snippet[None, None]).to(device)   # (1,1,H,W)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze().cpu().numpy()


def make_examples(
    model: torch.nn.Module,
    raw_snippets: dict,
    params: NarrowbandParams,
    preproc_cfg: dict,
    device: str,
    snr_eti: float,
    drift_rate: float,
    seed: int,
) -> list:
    """Build (label, snippet, recon, error) triples for [clean, rfi, eti]."""
    examples = []

    for label, role in [("Clean noise", "clean"), ("RFI (real)", "rfi")]:
        snippet = _preprocess(raw_snippets[role], preproc_cfg)
        recon   = _reconstruct(model, snippet, device)
        error   = (snippet - recon) ** 2
        examples.append((label, snippet, recon, error))

    # ETI: inject on raw, preprocess
    _, snippet_eti = _inject_eti(
        raw_snippets["clean2"], params, preproc_cfg,
        snr=snr_eti, drift_rate=drift_rate, seed=seed,
    )
    recon_eti = _reconstruct(model, snippet_eti, device)
    error_eti = (snippet_eti - recon_eti) ** 2
    examples.append(("ETI-injected", snippet_eti, recon_eti, error_eti))

    return examples


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_grid(examples: list, model_name: str, out_path: Path) -> None:
    """3 rows × 3 cols: rows = clean / rfi / eti, cols = input / recon / error."""
    col_labels = ["Input", "Reconstruction", "Reconstruction error"]

    # Shared intensity range across all input and reconstruction panels
    spec_vals = np.concatenate(
        [e[1].ravel() for e in examples] + [e[2].ravel() for e in examples]
    )
    vmin, vmax = np.percentile(spec_vals, [2, 98])

    # Shared error scale: 99th-percentile across all three error maps
    err_vmax = max(float(np.percentile(e[3], 99)) for e in examples)

    fig, axes = plt.subplots(3, 3, figsize=(18, 6))
    fig.suptitle(f"{model_name}  —  input / reconstruction / error",
                 fontsize=13, y=1.01)

    for row, (label, snippet, recon, error) in enumerate(examples):
        mse = float(error.mean())

        axes[row, 0].imshow(snippet, aspect="auto", origin="lower",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(recon, aspect="auto", origin="lower",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        im_err = axes[row, 2].imshow(error, aspect="auto", origin="lower",
                                     cmap="hot", vmin=0, vmax=err_vmax)

        # Row label on left
        axes[row, 0].set_ylabel(label, fontsize=11, labelpad=8)

        # MSE in the top-left corner of the error panel
        axes[row, 2].text(0.02, 0.96, f"MSE = {mse:.4f}",
                          transform=axes[row, 2].transAxes,
                          va="top", fontsize=8, color="white",
                          bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=11, pad=5)

    # Single colorbar for the error column
    cbar = fig.colorbar(im_err, ax=axes[:, 2].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("squared error", fontsize=9)

    # Frequency-axis label on bottom row only
    for col in range(3):
        axes[2, col].set_xlabel("frequency channel", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True,
                   help="Cache directory containing train.npy / val.npy")
    p.add_argument("--split", default="train",
                   help="Which split to sample from (default: train)")
    p.add_argument("--n_scan", type=int, default=500,
                   help="Number of snippets to scan when selecting examples")
    p.add_argument("--ae_checkpoint",     type=Path, default=None)
    p.add_argument("--vitmae_checkpoint", type=Path, default=None)
    p.add_argument("--ae_config",     type=Path, default=ROOT / "configs/model/convae.yaml")
    p.add_argument("--vitmae_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--data_config",   type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--snr_eti",    type=float, default=25.0,
                   help="Injected ETI SNR (setigen frame-integrated)")
    p.add_argument("--drift_rate", type=float, default=0.3,
                   help="ETI drift rate in Hz/s")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/slides")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ae_checkpoint is None and args.vitmae_checkpoint is None:
        raise SystemExit(
            "Provide at least one of --ae_checkpoint / --vitmae_checkpoint."
        )

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc_cfg = data_cfg["preprocessing"]

    # NarrowbandParams for ETI injection — geometry matches gbt_fine product
    raw_cfg = data_cfg["raw"]
    frm_cfg = data_cfg["frame"]
    params = NarrowbandParams(
        df=raw_cfg["df"],
        dt=raw_cfg["dt"],
        fch1=1500.0,          # L-band centre approximation; only affects setigen geometry
        fchans=frm_cfg["fchans"],
        tchans=frm_cfg["tchans"],
        ascending=False,
    )

    print(f"\nSelecting snippets from {args.cache} ({args.split}) …")
    raw_snippets = select_snippets(
        args.cache, args.split, preproc_cfg,
        n_scan=args.n_scan, seed=args.seed,
    )

    jobs = []
    if args.ae_checkpoint:
        with open(args.ae_config) as f:
            jobs.append(("ConvAE", args.ae_checkpoint, yaml.safe_load(f), "ae"))
    if args.vitmae_checkpoint:
        with open(args.vitmae_config) as f:
            jobs.append(("ViT-MAE", args.vitmae_checkpoint, yaml.safe_load(f), "vitmae"))

    for name, ckpt_path, model_cfg, tag in jobs:
        print(f"\n[{name}] loading {ckpt_path} …")
        model = _load_model(ckpt_path, model_cfg, args.device)
        print(f"[{name}] building examples …")
        examples = make_examples(
            model, raw_snippets, params, preproc_cfg, args.device,
            snr_eti=args.snr_eti, drift_rate=args.drift_rate, seed=args.seed,
        )
        plot_grid(examples, name, args.out_dir / f"recon_grid_{tag}.png")


if __name__ == "__main__":
    main()
