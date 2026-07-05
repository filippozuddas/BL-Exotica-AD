"""One-shot CPU smoke test for the UDMA backbone (checklist step 4,
docs/2026-07-05_udma_design_spec.md).

Random input only — no real dataset — but DOES need the pinned ViT-MAE
checkpoint (``build_udma`` loads it via a strict ``state_dict``), so this is a
SERVER task like ``fit_udma_teacher_norm.py``, despite the design spec's
"(dev machine)" note for step 4 (written before the checkpoint's location was
pinned down to the server). Pure construction/shape/contract checks — it does
NOT validate learned behaviour (that needs ``fit_udma_teacher_norm.py`` +
training + the eval harness, checklist steps 3/5/6).

Checks, in order:
  1. ``build_autoencoder(..., architecture: udma)`` constructs without error —
     exercises the strict ``state_dict`` load of the pinned ViT checkpoint.
  2. ``TeacherViT(x) -> (B, 128, 6, 64)``; ``ViTMAE.encode_tokens(x)`` still
     equals ``encode_tokens_at(x, -1)`` after the ``udma.py`` refactor
     (regression check on the existing dist384/Mahalanobis embedding path).
  3. ``FeatureStudent`` forward shapes match the teacher's grid, for both the
     plain and memory-augmented student.
  4. ``compute_loss(x)`` returns ``(scalar, {st1,st2,ss,entropy,st_sum})``,
     all finite, ``st_sum == st1 + st2``.
  5. ``anomaly_score(x, method='recon'|'topk'|'max')`` -> ``(B,)``, all finite.
  6. ``model(x)`` raises ``NotImplementedError`` (no pixel decoder, by design).

Usage (server, torch + the pinned checkpoint required):
    PYTHONPATH=/content/filippo/BL-Exotica-AD python scripts/debug/smoke_test_udma.py \\
        --model_config configs/model/udma.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import build_autoencoder

INPUT_SHAPE = (96, 1024, 1)
BATCH = 4


def _check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"Smoke test failed: {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_config", type=Path, default=Path("configs/model/udma.yaml"))
    args = p.parse_args()

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    if model_config.get("architecture") != "udma":
        raise SystemExit(f"--model_config must be architecture: udma, got '{model_config.get('architecture')}'.")

    print("1. Building UDMA (loads pinned ViT-MAE checkpoint)...")
    model = build_autoencoder(INPUT_SHAPE, model_config, loss="mse")
    model.eval()
    print("   built OK.")

    torch.manual_seed(0)
    x = torch.randn(BATCH, 1, INPUT_SHAPE[0], INPUT_SHAPE[1])

    print("2. Teacher token grid + encode_tokens/_at(-1) regression check...")
    target = model.teacher(x)
    _check(
        "teacher(x) shape == (B, channels, nh, nw)",
        tuple(target.shape) == (BATCH, model.teacher.channels, *model.teacher.grid_size),
    )
    tok_default = model.teacher.vit.encode_tokens(x)
    tok_final = model.teacher.vit.encode_tokens_at(x, -1)
    _check("encode_tokens(x) == encode_tokens_at(x, -1)", torch.allclose(tok_default, tok_final))

    print("3. Student forward shapes...")
    s_ae = model.student_ae(x)
    s_mem, att = model.student_mem(x)
    _check("student_ae(x) shape matches teacher grid", tuple(s_ae.shape) == tuple(target.shape))
    _check("student_mem(x) shape matches teacher grid", tuple(s_mem.shape) == tuple(target.shape))
    nh, nw = model.teacher.grid_size
    _check(
        "student_mem attention shape (B*nh*nw, mem_slots)",
        att.shape[0] == BATCH * nh * nw,
    )

    print("4. compute_loss contract...")
    total, components = model.compute_loss(x)
    _check("total is a finite scalar", total.dim() == 0 and bool(torch.isfinite(total)))
    expected_keys = {"st1", "st2", "ss", "entropy", "st_sum"}
    _check(f"components has keys {expected_keys}", expected_keys.issubset(components.keys()))
    _check("all components finite", all(bool(torch.isfinite(v)) for v in components.values()))
    _check("st_sum == st1 + st2", bool(torch.allclose(components["st_sum"], components["st1"] + components["st2"])))

    print("5. anomaly_score contract...")
    for method in ("recon", "topk", "max"):
        score = model.anomaly_score(x, method=method)
        _check(f"anomaly_score(method='{method}') shape (B,)", tuple(score.shape) == (BATCH,))
        _check(f"anomaly_score(method='{method}') finite", bool(torch.isfinite(score).all()))

    print("6. forward() raises (no pixel decoder)...")
    try:
        model(x)
        _check("model(x) raises NotImplementedError", False)
    except NotImplementedError:
        print("  [PASS] model(x) raises NotImplementedError")

    print(
        "\nAll smoke checks passed. UDMA construction/shapes/contracts are sound "
        "(random input only — does not validate learned behaviour)."
    )


if __name__ == "__main__":
    main()
