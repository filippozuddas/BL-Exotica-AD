"""
SSAST-style ViT-MAE: compare the three anomaly scores on injected signals vs RFI.

The empirical test that decides which score works for which product ("best model
wins", CLAUDE.md). For a trained ViT-MAE checkpoint, computes all three scores —
``recon`` (partitioned reconstruction MSE), ``infonce`` (per-patch self-recognition
difficulty), ``embedding`` (encoder features + Isolation Forest fit on the
"normal" training snippets) — across four categories:

  - Quiet:        low-RFI baseline snippets
  - RFI:          high-RFI snippets
  - Inject-ALL:   quiet + synthetic narrowband in all 6 observations
  - Inject-ON:    quiet + synthetic narrowband in ON observations only (realistic ETI)

Success = at least one score separates Inject-ON from Quiet/RFI (ratio >> 1),
ideally better than the reconstruction-only ViT-MAE baseline.

Reuses the injection/preprocess helpers from injection_vs_rfi_test.py.

Usage (run on the server, not the dev machine):
    PYTHONPATH=. python scripts/debug/ssast_scoring_test.py \
        --checkpoint outputs/<run>/checkpoints/best_model.ckpt \
        --cache /path/to/cache_gbt_fine.npz \
        --data_config configs/data/gbt_fine.yaml \
        --model_config configs/model/vit_mae.yaml \
        --n_samples 100 --inject_snr 25 \
        --out_dir outputs/ssast_scoring_test
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

from src.models.autoencoder import build_autoencoder
from src.search.scorer import OneClassScorer
from scripts.debug.injection_vs_rfi_test import (
    preprocess_raw,
    inject_narrowband,
    inject_narrowband_on_only,
)

INPUT_SHAPE = (96, 1024, 1)
METHODS = ["recon", "infonce", "embedding"]


def load_model(checkpoint_path: Path, model_config: dict, device: str):
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def to_tensor(snip: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(snip).float().unsqueeze(0).unsqueeze(0).to(device)


def score_snippet(model, snip: np.ndarray, method: str, occ, device: str) -> float:
    x = to_tensor(snip, device)
    return float(model.anomaly_score(x, method=method, occ=occ).item())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True, help="NPZ cache path")
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--inject_snr", type=float, default=25.0)
    p.add_argument("--occ_estimator", default="isolation_forest",
                   choices=["isolation_forest", "ocsvm"])
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/ssast_scoring_test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, args.device)

    print(f"Loading NPZ: {args.cache}")
    archive = np.load(str(args.cache), mmap_mode="r")
    arr = archive[args.split]
    indices = rng.choice(arr.shape[0], size=min(args.n_samples, arr.shape[0]), replace=False)
    raw_snippets = np.array(arr[indices])
    del arr, archive
    print(f"  Raw snippets: {raw_snippets.shape}")

    # Preprocess + categorise by RFI content (hot-pixel fraction).
    preprocessed, hot_fracs = [], []
    for i in range(len(raw_snippets)):
        snip = preprocess_raw(raw_snippets[i], preproc)
        preprocessed.append(snip)
        hot_fracs.append(float((snip > 5.0).sum()) / snip.size)
    preprocessed = np.array(preprocessed)
    hot_fracs = np.array(hot_fracs)

    q25, q75 = np.percentile(hot_fracs, [25, 75])
    quiet_idx = np.where(hot_fracs <= q25)[0]
    rfi_idx = np.where(hot_fracs >= q75)[0]
    print(f"  Quiet: {len(quiet_idx)}   RFI: {len(rfi_idx)}")

    # Fit the one-class scorer on ALL sampled snippets (the "normal" training set).
    print(f"Fitting OneClassScorer ({args.occ_estimator}) on encoder embeddings...")
    with torch.no_grad():
        embs = [model.encode(to_tensor(s, args.device)).cpu().numpy() for s in preprocessed]
    occ = OneClassScorer(args.occ_estimator).fit_embeddings(np.concatenate(embs, axis=0))
    occ_path = args.out_dir / "oneclass.joblib"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    occ.save(occ_path)
    print(f"  Saved → {occ_path}")

    # Build the four snippet categories.
    n_inject = min(15, len(quiet_idx))
    categories = {
        "Quiet": [preprocessed[i] for i in quiet_idx],
        "RFI": [preprocessed[i] for i in rfi_idx],
        "Inject-ALL": [preprocess_raw(inject_narrowband(raw_snippets[i], snr=args.inject_snr,
                                                        seed=args.seed + j), preproc)
                       for j, i in enumerate(quiet_idx[:n_inject])],
        "Inject-ON": [preprocess_raw(inject_narrowband_on_only(raw_snippets[i], snr=args.inject_snr,
                                                               seed=args.seed + j), preproc)
                      for j, i in enumerate(quiet_idx[:n_inject])],
    }

    # Score every category with every method.
    results = {m: {} for m in METHODS}
    for method in METHODS:
        for cat, snips in categories.items():
            vals = np.array([score_snippet(model, s, method, occ, args.device) for s in snips])
            results[method][cat] = vals

    # Report.
    print("\n" + "=" * 64)
    print("ANOMALY SCORE BY METHOD AND CATEGORY (mean ± std)")
    print("=" * 64)
    for method in METHODS:
        print(f"\n  [{method}]")
        ref_quiet = results[method]["Quiet"].mean()
        ref_rfi = results[method]["RFI"].mean()
        for cat in categories:
            v = results[method][cat]
            print(f"    {cat:<12s}  {v.mean():12.4f} ± {v.std():8.4f}")
        on = results[method]["Inject-ON"].mean()
        print(f"    -- Inject-ON / Quiet: {on / ref_quiet:.3f}x   "
              f"Inject-ON / RFI: {on / ref_rfi:.3f}x")

    # Plot: grouped bars per method.
    fig, axes = plt.subplots(1, len(METHODS), figsize=(6 * len(METHODS), 5))
    for ax, method in zip(axes, METHODS):
        cats = list(categories)
        means = [results[method][c].mean() for c in cats]
        stds = [results[method][c].std() for c in cats]
        ax.bar(cats, means, yerr=stds, capsize=4,
               color=["steelblue", "orange", "crimson", "darkred"],
               alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(f"score: {method}")
        ax.set_ylabel("anomaly score")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"SSAST ViT-MAE anomaly scores (inject SNR={args.inject_snr})")
    plt.tight_layout()
    out = args.out_dir / "score_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved → {out}\nDone.")


if __name__ == "__main__":
    main()
