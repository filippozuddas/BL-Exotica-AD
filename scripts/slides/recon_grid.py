"""
Slide visualization: 3×3 grid of (input | reconstruction | error) for
clean / naturally-RFI-contaminated / ETI-injected snippets from the real cache.

Produces one PNG per model backbone (AE, ViT-MAE).

ETI injection is ON-only: the signal is injected in observations 0, 2, 4
(the A observations of the ABACAD cadence) with drift continuity preserved
across the full time axis.  OFF observations (1, 3, 5) are left untouched.

Usage:
    python scripts/slides/recon_grid.py \\
        --cache     /path/to/cache_gbt_fine \\
        --ae_checkpoint     outputs/<run>/checkpoints/best.ckpt \\
        --vitmae_checkpoint outputs/<run>/checkpoints/best.ckpt \\
        --snr_eti 20 --drift_rate 0.3 --width_chans 1.0 \\
        --out_dir outputs/slides
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
# Cache I/O and snippet selection
# --------------------------------------------------------------------------- #

def _preprocess(raw_obs: np.ndarray, preproc_cfg: dict) -> np.ndarray:
    """(n_obs, tchans_per_obs, fchans) → preprocessed (tchans, fchans) float32.

    Mirrors CachedDataset.__getitem__: concatenate obs, bandpass + core_transform.
    """
    frame = np.concatenate(raw_obs, axis=0).astype(np.float32)
    frame = bandpass_correct(frame,
                             method=preproc_cfg.get("bandpass_method", "polynomial"),
                             poly_degree=preproc_cfg.get("poly_degree", 3))
    return core_transform(frame, preproc_cfg.get("mad_epsilon", 1e-6)).astype(np.float32)


def _hot_frac(snippet: np.ndarray, sigma: float = 5.0) -> float:
    return float((snippet > sigma).sum()) / snippet.size


def select_snippets(
    cache_path: Path,
    split: str,
    preproc_cfg: dict,
    n_scan: int = 500,
    seed: int = 42,
) -> dict:
    """Return raw arrays for three roles by scanning n_scan random snippets.

    clean  — lowest hot_frac (quietest)
    clean2 — second-lowest (used as ETI injection base)
    rfi    — highest hot_frac (most naturally contaminated)
    """
    npy = cache_path / f"{split}.npy"
    arr = np.load(str(npy), mmap_mode="r")
    n_total = arr.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n_scan, n_total), replace=False)

    print(f"  Scanning {len(idx)} snippets from {npy.name} …")
    hot_fracs = []
    for i in idx:
        hot_fracs.append(_hot_frac(_preprocess(arr[i], preproc_cfg)))

    order = np.argsort(hot_fracs)
    clean_i  = int(idx[order[0]])
    clean2_i = int(idx[order[1]])
    rfi_i    = int(idx[order[-1]])

    print(f"  clean  [{clean_i}]   hot_frac = {hot_fracs[order[0]]:.4f}")
    print(f"  clean2 [{clean2_i}]  hot_frac = {hot_fracs[order[1]]:.4f}  (ETI base)")
    print(f"  rfi    [{rfi_i}]   hot_frac = {hot_fracs[order[-1]]:.4f}")

    return {
        "clean":  np.array(arr[clean_i]),
        "clean2": np.array(arr[clean2_i]),
        "rfi":    np.array(arr[rfi_i]),
    }


# --------------------------------------------------------------------------- #
# ETI injection — ON observations only, narrow Gaussian profile
# --------------------------------------------------------------------------- #

def _inject_narrowband_on_only(
    raw_obs: np.ndarray,
    preproc_cfg: dict,
    raw_cfg: dict,
    snr: float,
    drift_rate: float,
    width_chans: float,
    on_obs: tuple,
    seed: int,
) -> np.ndarray:
    """Inject a drifting narrowband signal in ON observations only.

    raw_obs : (n_obs, tchans_per_obs, fchans) — raw un-preprocessed
    Returns  : preprocessed (tchans, fchans) float32 with signal injected

    The signal drifts continuously (global time index) so the track is coherent
    across ON observations even though OFF observations carry no signal.
    """
    rng  = np.random.default_rng(seed)
    raw  = raw_obs.copy().astype(float)
    n_obs, tchans_per_obs, fchans = raw.shape

    noise_std        = np.median(np.abs(raw - np.median(raw))) * 1.4826
    signal_amplitude = snr * noise_std

    margin     = max(50, int(fchans * 0.10))
    start_chan = float(rng.integers(margin, fchans - margin))

    dt = raw_cfg["dt"]   # s / time-bin
    df = raw_cfg["df"]   # Hz / channel
    drift_chans_per_bin = (drift_rate * dt) / df
    chans = np.arange(fchans, dtype=float)

    for obs_idx in on_obs:
        global_t0 = obs_idx * tchans_per_obs
        for t in range(tchans_per_obs):
            center  = start_chan + (global_t0 + t) * drift_chans_per_bin
            profile = signal_amplitude * np.exp(
                -0.5 * ((chans - center) / width_chans) ** 2
            )
            raw[obs_idx, t] += profile

    # Preprocess after injection
    frame = np.concatenate(raw, axis=0).astype(np.float32)
    frame = bandpass_correct(frame,
                             method=preproc_cfg.get("bandpass_method", "polynomial"),
                             poly_degree=preproc_cfg.get("poly_degree", 3))
    return core_transform(frame, preproc_cfg.get("mad_epsilon", 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def _reconstruct(model: torch.nn.Module, snippet: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(snippet[None, None]).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze().cpu().numpy()


def make_examples(
    model: torch.nn.Module,
    raw_snippets: dict,
    preproc_cfg: dict,
    raw_cfg: dict,
    device: str,
    snr_eti: float,
    drift_rate: float,
    width_chans: float,
    on_obs: tuple,
    seed: int,
) -> list:
    """Return (label, snippet, recon, error) triples for [clean, rfi, eti]."""
    examples = []

    for label, role in [("Clean noise", "clean"), ("RFI (real)", "rfi")]:
        snippet = _preprocess(raw_snippets[role], preproc_cfg)
        recon   = _reconstruct(model, snippet, device)
        examples.append((label, snippet, recon, (snippet - recon) ** 2))

    snippet_eti = _inject_narrowband_on_only(
        raw_snippets["clean2"], preproc_cfg, raw_cfg,
        snr=snr_eti, drift_rate=drift_rate, width_chans=width_chans,
        on_obs=on_obs, seed=seed,
    )
    recon_eti = _reconstruct(model, snippet_eti, device)
    examples.append(("ETI-injected\n(ON obs only)", snippet_eti, recon_eti,
                     (snippet_eti - recon_eti) ** 2))

    return examples


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_grid(examples: list, model_name: str, out_path: Path) -> None:
    col_labels = ["Input", "Reconstruction", "Reconstruction error"]

    spec_vals = np.concatenate(
        [e[1].ravel() for e in examples] + [e[2].ravel() for e in examples]
    )
    vmin, vmax = np.percentile(spec_vals, [2, 98])
    err_vmax   = max(float(np.percentile(e[3], 99)) for e in examples)

    fig, axes = plt.subplots(3, 3, figsize=(18, 6))
    fig.suptitle(f"{model_name}  —  input / reconstruction / error",
                 fontsize=13, y=0.98)

    last_im_err = None
    for row, (label, snippet, recon, error) in enumerate(examples):
        axes[row, 0].imshow(snippet, aspect="auto", origin="lower",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(recon,   aspect="auto", origin="lower",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        last_im_err = axes[row, 2].imshow(error, aspect="auto", origin="lower",
                                          cmap="hot", vmin=0, vmax=err_vmax)

        axes[row, 0].set_ylabel(label, fontsize=10, labelpad=8)
        axes[row, 2].text(
            0.02, 0.96, f"MSE = {error.mean():.4f}",
            transform=axes[row, 2].transAxes,
            va="top", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=11, pad=5)

    for col in range(3):
        axes[2, col].set_xlabel("frequency channel", fontsize=9)

    # Reserve right margin, then place colorbar anchored to the error column
    fig.subplots_adjust(
        left=0.07, right=0.88, top=0.92, bottom=0.10,
        wspace=0.03, hspace=0.04,
    )
    pos_top = axes[0, 2].get_position()
    pos_bot = axes[2, 2].get_position()
    cbar_ax = fig.add_axes([
        pos_top.x1 + 0.01,          # left edge: just right of error column
        pos_bot.y0,                  # bottom: aligned with bottom row
        0.012,                       # width
        pos_top.y1 - pos_bot.y0,     # height: full span of three rows
    ])
    cb = fig.colorbar(last_im_err, cax=cbar_ax)
    cb.set_label("squared error", fontsize=9)

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
                   help="Cache directory (contains train.npy / val.npy)")
    p.add_argument("--split",  default="train")
    p.add_argument("--n_scan", type=int, default=500)
    p.add_argument("--ae_checkpoint",     type=Path, default=None)
    p.add_argument("--vitmae_checkpoint", type=Path, default=None)
    p.add_argument("--ae_config",     type=Path, default=ROOT / "configs/model/convae.yaml")
    p.add_argument("--vitmae_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--data_config",   type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--snr_eti",     type=float, default=20.0)
    p.add_argument("--drift_rate",  type=float, default=0.3,
                   help="ETI drift rate in Hz/s")
    p.add_argument("--width_chans", type=float, default=1.0,
                   help="ETI Gaussian half-width in channels (default: 1.0)")
    p.add_argument("--on_obs",  default="0,2,4",
                   help="Comma-separated ON observation indices (default: 0,2,4)")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/slides")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ae_checkpoint is None and args.vitmae_checkpoint is None:
        raise SystemExit("Provide --ae_checkpoint and/or --vitmae_checkpoint.")

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc_cfg = data_cfg["preprocessing"]
    raw_cfg     = data_cfg["raw"]
    on_obs      = tuple(int(x) for x in args.on_obs.split(","))

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
            model, raw_snippets, preproc_cfg, raw_cfg, args.device,
            snr_eti=args.snr_eti, drift_rate=args.drift_rate,
            width_chans=args.width_chans, on_obs=on_obs, seed=args.seed,
        )
        plot_grid(examples, name, args.out_dir / f"recon_grid_{tag}.png")


if __name__ == "__main__":
    main()
