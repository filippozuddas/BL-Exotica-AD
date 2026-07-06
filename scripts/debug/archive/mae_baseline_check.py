"""Measure the masked-patch MSE floor for trivial predictors on the val set.

Compares two trivial baselines against CNN-MAE's val_loss, using the same
mask geometry (patch_size=(4,4), mask_ratio=0.5 -- convae_mae_lowmask) and
the same val split (seed=42) as configs/training/srt_real.yaml:

  (a) predict 0 everywhere
  (b) predict the per-sample mean of the VISIBLE (unmasked) pixels

If MAE's val_loss is close to or above these numbers, the model isn't using
visible context to fill masked patches -- a genuine architecture-level
plateau. If it's clearly below both, the model IS extracting context and the
plateau is more likely "needs more epochs / lower LR".

Run on the server (needs real data):
    PYTHONPATH=. python scripts/debug/mae_baseline_check.py
"""

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.torch_dataset import build_datasets
from src.models.autoencoder import MAE

data_cfg = yaml.safe_load(open(ROOT / "configs/data/gbt_fine.yaml"))

file_list_path = ROOT / data_cfg["dataset"]["file_list"]
file_list = [p.strip() for p in file_list_path.read_text().splitlines() if p.strip()]

_, val_ds = build_datasets(file_list, data_cfg, val_fraction=0.15, seed=42)
print(f"Val snippets: {len(val_ds)}")

val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4)

# Same mask geometry as the running experiment (convae_mae_lowmask.yaml).
mae = MAE(encoder=None, decoder=None, loss_fn=None, patch_size=(4, 4), mask_ratio=0.5)

zero_num = zero_den = 0.0
mean_num = mean_den = 0.0

with torch.no_grad():
    for x in val_loader:
        mask = mae._make_mask(x)  # 1 = masked, 0 = visible
        visible = 1.0 - mask

        # (a) predict 0 everywhere
        zero_num += ((x ** 2) * mask).sum().item()
        zero_den += mask.sum().item()

        # (b) predict per-sample mean of visible pixels
        n_visible = visible.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        mean_visible = (x * visible).sum(dim=(1, 2, 3), keepdim=True) / n_visible
        mean_num += (((x - mean_visible) ** 2) * mask).sum().item()
        mean_den += mask.sum().item()

baseline_zero = zero_num / zero_den
baseline_mean = mean_num / mean_den

print(f"\nMasked-patch MSE, predict 0 everywhere:    {baseline_zero:.4f}")
print(f"Masked-patch MSE, predict per-sample mean: {baseline_mean:.4f}")
print("\nCompare to CNN-MAE val_loss (best so far: 2.2134, run 20260615_091556_da44aab, ep25).")
