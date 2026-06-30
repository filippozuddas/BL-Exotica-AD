"""SSIM overfit test — verifica se ViT-MAE + SSIM impara a ricostruire la struttura RFI.

Domanda: un modello trainato con SSIM loss su una piccola batch di snippet reali
riesce a ricostruire gli snippet RFI strutturati meglio degli snippet di rumore puro?

Se sì → SSIM previene il collasso alla media e il modello impara struttura locale.
Se no → anche SSIM collassa, problema più profondo.

Metrica: dopo ogni eval_every epoche, misura per ogni gruppo:
  1 - SSIM(snippet, ricostruzione)  (più basso = ricostruzione migliore)

Un modello che funziona mostra: SSIM-loss(noise) < SSIM-loss(RFI),
cioè ricostruisce meglio i pattern RFI strutturati che ha visto in training.

Usage (server):
    PYTHONPATH=/content/filippo/BL-Exotica-AD \\
    python scripts/debug/ssim_overfit_test.py \\
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine \\
        --model_config configs/model/vit_mae_ssim.yaml \\
        --n_noise 50 --n_rfi 50 \\
        --epochs 300 --eval_every 25 \\
        --device cuda
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
from src.data.preprocessing import bandpass_correct, core_transform
from src.models.losses import ssim_loss


def preprocess(raw: np.ndarray, preproc: dict) -> np.ndarray:
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    result = np.concatenate(raw, axis=0)
    result = bandpass_correct(result, method=method, poly_degree=poly_degree)
    return core_transform(result, mad_epsilon).astype(np.float32)


def hot_frac(snippet: np.ndarray, thr: float = 5.0) -> float:
    return float((snippet > thr).sum()) / snippet.size


@torch.no_grad()
def eval_ssim_loss(model, batch: torch.Tensor) -> float:
    """Mean 1-SSIM reconstruction loss on a batch (no masking = forward inference)."""
    recon = model.forward(batch)
    return float(ssim_loss(batch, recon).mean().item())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae_ssim.yaml")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--n_noise", type=int, default=50, help="Noise snippets (low hot_frac)")
    p.add_argument("--n_rfi", type=int, default=50, help="RFI snippets (high hot_frac)")
    p.add_argument("--sample_pool", type=int, default=2000, help="Snippets to scan for selection")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/ssim_overfit")
    p.add_argument("--n_examples", type=int, default=4, help="Examples to visualise per group")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    # ── load snippets from cache ──────────────────────────────────────────────
    npy_path = args.cache / f"{args.split}.npy"
    print(f"Scanning cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    pool_idx = rng.choice(arr.shape[0], size=min(args.sample_pool, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[pool_idx])
    del arr
    print(f"  Pool shape: {raw_pool.shape}")

    print("Preprocessing + computing hot_frac...")
    preprocessed, hot_fracs = [], []
    for raw in raw_pool:
        pp = preprocess(raw, preproc)
        preprocessed.append(pp)
        hot_fracs.append(hot_frac(pp))
    preprocessed = np.array(preprocessed)
    hot_fracs = np.array(hot_fracs)

    order = np.argsort(hot_fracs)
    noise_idx = order[:args.n_noise]
    rfi_idx = order[-args.n_rfi:]

    noise_hf = hot_fracs[noise_idx]
    rfi_hf = hot_fracs[rfi_idx]
    print(f"  Noise  (bottom {args.n_noise}): hot_frac {noise_hf.min():.2e}–{noise_hf.max():.2e}")
    print(f"  RFI    (top    {args.n_rfi}):   hot_frac {rfi_hf.min():.2e}–{rfi_hf.max():.2e}")

    def to_tensor(idx):
        return torch.from_numpy(preprocessed[idx]).float().unsqueeze(1).to(args.device)

    noise_batch = to_tensor(noise_idx)
    rfi_batch = to_tensor(rfi_idx)
    # train on the union of noise + RFI (realistic: model sees both as "normal")
    all_snippets = torch.cat([noise_batch, rfi_batch], dim=0)

    # ── build model ───────────────────────────────────────────────────────────
    input_shape = (preprocessed.shape[1], preprocessed.shape[2], 1)  # (96, 1024, 1)
    print(f"\nBuilding model  input_shape={input_shape}  loss_mode={model_cfg.get('loss_mode')}")
    model = build_autoencoder(input_shape, model_cfg, loss="mse", learning_rate=args.lr)
    model = model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(all_snippets)
    bs = args.batch_size

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining {args.epochs} epochs on {n} snippets (batch_size={bs})\n")
    print(f"{'Epoch':>6}  {'train_loss':>12}  {'SSIM↑err noise':>16}  {'SSIM↑err RFI':>14}  "
          f"{'Δ(RFI-noise)':>13}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=args.device)
        epoch_losses = []
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            x = all_snippets[idx]
            optimizer.zero_grad()
            loss = model.compute_loss(x)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_loss = float(np.mean(epoch_losses))
                err_noise = eval_ssim_loss(model, noise_batch)
                err_rfi = eval_ssim_loss(model, rfi_batch)
                delta = err_rfi - err_noise
                # delta < 0 → model reconstructs RFI BETTER than noise (good)
                # delta > 0 → model reconstructs noise better (collapsed to mean)
                flag = " ← RFI better" if delta < -0.005 else (" ← COLLAPSED" if delta > 0.01 else "")
                print(f"{epoch:6d}  {train_loss:12.6f}  {err_noise:16.4f}  {err_rfi:14.4f}  "
                      f"{delta:+13.4f}{flag}")

    # ── final verdict ─────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_noise = eval_ssim_loss(model, noise_batch)
        final_rfi = eval_ssim_loss(model, rfi_batch)

    print(f"\n{'='*75}")
    print("VERDICT")
    print(f"  Final SSIM reconstruction error — noise: {final_noise:.4f}  RFI: {final_rfi:.4f}")
    if final_rfi < final_noise - 0.01:
        print("  PASS: model reconstructs RFI better than noise → SSIM prevents collapse,")
        print("        structural learning is happening. Proceed to full training.")
    elif final_rfi > final_noise + 0.01:
        print("  FAIL (collapsed): model reconstructs noise better than RFI → SSIM loss")
        print("        still collapses to noise mean. Structural learning not happening.")
    else:
        print("  AMBIGUOUS: RFI and noise reconstruction quality are similar.")
        print("        Increase n_rfi, epochs, or check hot_frac threshold.")
    print(f"{'='*75}")

    # ── reconstruction examples ───────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    k = min(args.n_examples, len(noise_idx), len(rfi_idx))

    for group_name, batch, hf_vals in [("noise", noise_batch[:k], hot_fracs[noise_idx[:k]]),
                                        ("rfi",   rfi_batch[:k],   hot_fracs[rfi_idx[:k]])]:
        with torch.no_grad():
            recon = model.forward(batch)
        orig  = batch.cpu().numpy()[:, 0]    # (k, 96, 1024)
        rec   = recon.cpu().numpy()[:, 0]
        diff  = np.abs(orig - rec)

        # clip display range to ±5 MAD units for noise legibility
        vmin, vmax = -3.0, 10.0

        fig, axes = plt.subplots(k, 3, figsize=(18, 3.5 * k))
        if k == 1:
            axes = [axes]
        fig.suptitle(f"Group: {group_name}   (columns: original | reconstruction | |error|)",
                     fontsize=11)

        for i in range(k):
            err_i = float(ssim_loss(batch[i:i+1], recon[i:i+1]).item())
            titles = [
                f"Original  hot_frac={hf_vals[i]:.2e}",
                f"Reconstruction  SSIM-err={err_i:.3f}",
                "Absolute error",
            ]
            for j, (ax, data, title) in enumerate(zip(axes[i], [orig[i], rec[i], diff[i]], titles)):
                vm = (vmin, vmax) if j < 2 else (0, diff[i].max())
                im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                               vmin=vm[0], vmax=vm[1])
                ax.set_title(title, fontsize=8)
                ax.set_xlabel("freq channel")
                ax.set_ylabel("time bin")
                plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        plt.tight_layout()
        out_path = args.out_dir / f"recon_{group_name}.png"
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
