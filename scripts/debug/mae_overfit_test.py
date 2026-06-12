import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.autoencoder import build_autoencoder

torch.manual_seed(0)

mae_cfg = yaml.safe_load(open(ROOT / "configs/model/convae_mae.yaml"))
ae_cfg = yaml.safe_load(open(ROOT / "configs/model/convae.yaml"))

input_shape = (16, 1024, 1)

# Synthetic batch roughly matching post-preprocessing stats (mean~0, var~2.2)
x = torch.randn(8, 1, 16, 1024) * (2.2 ** 0.5)

print("=== AE overfit on single batch ===")
ae = build_autoencoder(input_shape, ae_cfg, loss="mse", learning_rate=1e-3)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
for i in range(300):
    opt.zero_grad()
    loss = ae.compute_loss(x)
    loss.backward()
    opt.step()
    if i % 50 == 0 or i == 299:
        print(f"step {i:3d}: loss={loss.item():.4f}")

print("\n=== MAE overfit on single batch, FIXED mask ===")
mae = build_autoencoder(input_shape, mae_cfg, loss="mse", learning_rate=1e-3)
fixed_mask = mae._make_mask(x)
opt = torch.optim.Adam(mae.parameters(), lr=1e-3)
for i in range(300):
    opt.zero_grad()
    recon = mae(x * (1.0 - fixed_mask))
    loss = mae._masked_mse(x, recon, fixed_mask)
    loss.backward()
    opt.step()
    if i % 50 == 0 or i == 299:
        print(f"step {i:3d}: loss={loss.item():.4f}")

print("\n=== MAE overfit on single batch, RANDOM mask each step ===")
mae2 = build_autoencoder(input_shape, mae_cfg, loss="mse", learning_rate=1e-3)
opt = torch.optim.Adam(mae2.parameters(), lr=1e-3)
for i in range(300):
    opt.zero_grad()
    loss = mae2.compute_loss(x)
    loss.backward()
    opt.step()
    if i % 50 == 0 or i == 299:
        print(f"step {i:3d}: loss={loss.item():.4f}")

print(f"\nTarget variance (proxy for 'predict mean' floor): {x.var().item():.4f}")
