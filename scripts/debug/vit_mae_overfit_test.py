import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.autoencoder import build_autoencoder
from src.models.losses import _masked_mse
from src.models.vit_mae import _sample_random_ids, unpatchify

torch.manual_seed(0)

vit_cfg = yaml.safe_load(open(ROOT / "configs/model/vit_mae.yaml"))

input_shape = (96, 1024, 1)  # matches the actual training shape (6-obs cadence × 1024 fchans)

# Synthetic batch roughly matching post-preprocessing stats (mean~0, var~2.2)
x = torch.randn(8, 1, 96, 1024) * (2.2 ** 0.5)

print("=== ViT-MAE overfit on single batch, FIXED mask ===")
vit_mae = build_autoencoder(input_shape, vit_cfg, loss="mse", learning_rate=1e-3)
b, n = x.shape[0], vit_mae.num_patches
len_keep = n - int(vit_mae.mask_ratio * n)
ids_keep, ids_restore = _sample_random_ids(b, n, len_keep, device=x.device)

mask = torch.ones(b, n, device=x.device, dtype=x.dtype)
mask[:, :len_keep] = 0
mask = mask.gather(1, ids_restore)
nh, nw = vit_mae.grid_size
ph, pw = vit_mae.patch_size
pixel_mask = mask.view(b, 1, nh, nw).repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)

opt = torch.optim.Adam(vit_mae.parameters(), lr=1e-3)
for i in range(300):
    opt.zero_grad()
    pred_patches = vit_mae._decode_from_keep(x, ids_keep, ids_restore)
    pred = unpatchify(pred_patches, vit_mae.patch_size, (b, *vit_mae.input_shape))
    loss = _masked_mse(x, pred, pixel_mask)
    loss.backward()
    opt.step()
    if i % 50 == 0 or i == 299:
        print(f"step {i:3d}: loss={loss.item():.4f}")

print("\n=== ViT-MAE overfit on single batch, RANDOM mask each step ===")
vit_mae2 = build_autoencoder(input_shape, vit_cfg, loss="mse", learning_rate=1e-3)
opt = torch.optim.Adam(vit_mae2.parameters(), lr=1e-3)
for i in range(300):
    opt.zero_grad()
    loss = vit_mae2.compute_loss(x)
    loss.backward()
    opt.step()
    if i % 50 == 0 or i == 299:
        print(f"step {i:3d}: loss={loss.item():.4f}")

print(f"\nTarget variance (proxy for 'predict mean' floor): {x.var().item():.4f}")
