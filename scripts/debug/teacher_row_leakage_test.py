"""ON->OFF row-leakage probe for the out-of-domain teacher P (ResNet-18),
per depth stage — the cheap, training-free experiment that decides whether the
Q1 "small-RF teacher" mitigation (docs/2026-07-14_paper_alignment_plan.md, Fase 2)
can work at all, BEFORE spending a distillation + student-retrain run.

Motivation (2026-07-16): the distilled CNN teacher (read at ResNet layer3)
produces DIFFUSE UDMA anomaly maps — an ON-only signal (Voyager-1) bleeds into
the OFF rows, collapsing on_off_contrast from ~32 (old ViT-MAE teacher) to ~2.3
and breaking every ON/OFF short-list filter. The mechanism is specifically
CROSS-OBSERVATION leakage: the (96,1024) input stacks 6 observations of 16 time
bins each, so for the teacher's feature at an OFF row to move when a signal is
injected only into ON observations, the teacher's VERTICAL receptive field must
cross the 16-bin observation boundary. layer3's vertical RF (~211 px) spans the
whole input; earlier stages have smaller RF:

    stage           grid      vertical RF   spans (obs of 16px)
    stem(conv1+mp)  (24,256)  ~11 px        < 1  -> no cross-obs leak (in theory)
    layer1          (24,256)  ~43 px        ~2.7 -> partial
    layer2          (12,128)  ~99 px        ~6   -> whole cadence
    layer3 (now)    (6,64)    ~211 px       all  -> maximal

This probe measures the leakage DIRECTLY on P's features (no students, no
distillation): inject an ON-only line into a quiet frame, take the teacher
feature displacement ||P(x+s) - P(x)|| per (row, col) cell, and compare the
response in ON feature-rows vs OFF feature-rows. A localizing teacher keeps the
displacement in ON rows (high ON/OFF ratio); a leaky one spreads it (ratio ->1).

The test is clean because (a) inject_narrowband_on_only renders the signal ONLY
into obs 0/2/4 (OFF frames byte-identical) and (b) preprocess_raw normalizes
PER OBSERVATION (gbt_fine_normalization_bug fix), so an ON injection cannot shift
OFF pixels via renormalization — any OFF-row feature motion is pure RF leakage.

Feature rows map to observations by the stage's vertical downsampling: obs i
occupies feature rows [i*(nh/6) : (i+1)*(nh/6)); ON = obs {0,2,4}. No column
window / morphology assumption is used (row max over the full frequency axis).

Decision: if an early stage keeps the ON/OFF ratio high (localized), the RF is
the lever -> the small-RF teacher path is worth a training run (distill from
that stage + pool to (6,64)). If even the smallest useful stage stays diffuse
(ratio ->1), the leakage is not RF-driven (likely the domain-matching of the old
ViT-MAE teacher) and the ImageNet-P route is a dead end -> keep the domain-matched
teacher as a reported design result, or pursue a shallow anisotropic teacher.

Cost: ~ (n_frames * len(snr_list) * len(stages)) frozen ResNet forwards — minutes
on GPU, NO training.

``--architecture vit_mae`` runs the SAME differential metric on the domain-matched
ViT-MAE teacher (read at its deployed block, ``--layer 3``) as the apples-to-apples
LOCALIZING reference: one "stage" (global attention, no RF sweep). Use it to put a
number on the localization gap that the ResNet stages leave open (2026-07-16: RF
route refuted, ResNet ratio ~2-3 at every depth — see udma_teacher_rf_leakage_refuted).

Usage (server, not dev machine):
    # out-of-domain P (ResNet-18), RF sweep — default
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/teacher_row_leakage_test.py \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine_exotica

    # domain-matched ViT-MAE teacher (localizing reference), read at block3
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture vit_mae --layer 3 \
        --checkpoint outputs/training/20260709_205938_4b5660c/checkpoints/epoch=094-val_loss=2.1740.ckpt \
        --model_config configs/model/vit_mae.yaml \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine_exotica
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.debug.injection_vs_rfi_test import preprocess_raw, inject_narrowband_on_only

INPUT_SHAPE = (96, 1024, 1)
INPUT_HW = (96, 1024)
N_OBS = 6
ON_OBS = (0, 2, 4)

# Stage -> approximate theoretical vertical receptive field (px), for the
# printout only (helps read the ratio trend against the 16-px observation
# boundary). Values are ResNet-18 theoretical max RF on a (96,1024) input.
STAGE_RF = {"stem": 11, "layer1": 43, "layer2": 99, "layer3": 211}
STAGES = ("stem", "layer1", "layer2", "layer3")


class ResNetStages(nn.Module):
    """Frozen ImageNet ResNet-18 exposing features at each depth stage, so the
    same probe runs across the receptive-field sweep. Input (B,1,96,1024) is
    replicated to 3 channels with NO ImageNet mean/std renorm — same deliberate
    simplification as ``ResNetTeacher`` (the gate validated it)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
        self.layer1 = b.layer1
        self.layer2 = b.layer2
        self.layer3 = b.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ResNetStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) at the requested stage."""
        h = self.stem(x.repeat(1, 3, 1, 1))
        if stage == "stem":
            return h
        h = self.layer1(h)
        if stage == "layer1":
            return h
        h = self.layer2(h)
        if stage == "layer2":
            return h
        h = self.layer3(h)
        if stage == "layer3":
            return h
        raise ValueError(f"unknown stage {stage!r}")


class ViTMAEStages(nn.Module):
    """Domain-matched ViT-MAE teacher, exposing the SAME ``features(x, stage) ->
    (B,C,nh,nw)`` interface as :class:`ResNetStages` so it plugs into
    ``stage_displacement`` unchanged. A ViT has no receptive-field sweep (global
    attention), so it contributes ONE pseudo-stage read at the deployed
    transformer block (``layer``, 1-indexed; -1 = final). Token grid comes from
    the model config's ``patch_size`` — (6,64) for the 0000.fil teacher, rows
    mapping 1:1 to the 6 observations. Reads tokens exactly as
    ``teacher_sensitivity_test.encode_tokens_layer`` (patch_embed + pos_embed,
    blocks 1..k, final norm only when k == depth)."""

    def __init__(self, checkpoint, model_cfg, device, layer=-1):
        super().__init__()
        from scripts.debug.encode_separation_test import load_model
        self.model = load_model(checkpoint, model_cfg, device, require_encode=False)
        if not (hasattr(self.model, "patch_embed") and hasattr(self.model, "encoder")
                and hasattr(self.model.encoder, "layers")):
            raise SystemExit("--architecture vit_mae requires a ViT-MAE checkpoint "
                             "(architecture: vit_mae in --model_config).")
        self.layer = layer
        ph, pw = model_cfg["patch_size"]
        self.grid_size = (INPUT_HW[0] // ph, INPUT_HW[1] // pw)
        self.stage_name = "final" if layer == -1 else f"block{layer}"
        self.eval()

    def train(self, mode: bool = True) -> "ViTMAEStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str = None) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) token-feature grid at ``self.layer``."""
        m = self.model
        tok = m.patch_embed(x) + m.pos_embed
        n_layers = len(m.encoder.layers)
        k = n_layers if self.layer in (-1, n_layers) else self.layer
        if not 0 <= k <= n_layers:
            raise SystemExit(f"--layer must be in 0..{n_layers} or -1, got {self.layer}")
        for j, blk in enumerate(m.encoder.layers, start=1):
            if j > k:
                break
            tok = blk(tok)
        if k == n_layers and getattr(m.encoder, "norm", None) is not None:
            tok = m.encoder.norm(tok)
        b, n, d = tok.shape
        nh, nw = self.grid_size
        return tok.reshape(b, nh, nw, d).permute(0, 3, 1, 2)


def row_on_off_masks(nh: int) -> tuple:
    """Boolean (nh,) masks labelling each feature row ON or OFF by which of the
    6 observations it belongs to (ON = obs 0/2/4)."""
    rows_per_obs = nh // N_OBS
    if rows_per_obs == 0:
        raise ValueError(f"grid rows {nh} < {N_OBS} observations")
    obs_of_row = np.arange(nh) // rows_per_obs
    on_mask = np.isin(obs_of_row, ON_OBS)
    return on_mask, ~on_mask


@torch.no_grad()
def stage_displacement(model, f_clean, f_inj, stage, device, batch=32):
    """Per-frame ON-row and OFF-row response to the injection at one stage.

    Returns (on_resp, off_resp): each (N,) = per-frame mean over that group's
    feature rows of the row's MAX displacement over the frequency axis (mirrors
    ``full_row_hits``'s per-row max, no column window)."""
    on_resp, off_resp = [], []
    for i in range(0, len(f_clean), batch):
        xc = torch.from_numpy(f_clean[i:i + batch]).float().unsqueeze(1).to(device)
        xi = torch.from_numpy(f_inj[i:i + batch]).float().unsqueeze(1).to(device)
        fc = model.features(xc, stage)
        fi = model.features(xi, stage)
        disp = (fi - fc).norm(dim=1)                 # (b, nh, nw)
        row_max = disp.max(dim=2).values.cpu().numpy()  # (b, nh)
        nh = row_max.shape[1]
        on_mask, off_mask = row_on_off_masks(nh)
        on_resp.append(row_max[:, on_mask].mean(axis=1))
        off_resp.append(row_max[:, off_mask].mean(axis=1))
    return np.concatenate(on_resp), np.concatenate(off_resp)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", choices=["resnet18", "vit_mae"], default="resnet18",
                   help="resnet18: out-of-domain P, RF sweep across stem/layer1/2/3 (default). "
                        "vit_mae: domain-matched teacher, single stage at --layer (localizing "
                        "reference) — requires --checkpoint/--model_config.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Required for --architecture vit_mae; ignored for resnet18.")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml",
                   help="ViT-MAE model config (patch_size -> token grid). Used only for vit_mae.")
    p.add_argument("--layer", type=int, default=3,
                   help="ViT-MAE transformer block to read (1-indexed; -1 = final). Default 3 = "
                        "the deployed teacher_layer in configs/model/udma.yaml. Ignored for resnet18.")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--n_frames", type=int, default=200, help="Quiet injection sites.")
    p.add_argument("--snr_list", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES,
                   help="ResNet depth stages to sweep (resnet18 only).")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/teacher_row_leakage")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]

    if args.architecture == "vit_mae":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture vit_mae.")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        print(f"Loading domain-matched ViT-MAE teacher from {args.checkpoint} "
              f"(block {args.layer})")
        model = ViTMAEStages(args.checkpoint, model_cfg, args.device, layer=args.layer)
        stages = [model.stage_name]
        rf_label = {model.stage_name: "global"}
        print(f"  Token grid: {model.grid_size} — global attention, no RF sweep")
    else:
        print("Loading ResNet-18 (ImageNet, frozen) — teacher P, all stages")
        model = ResNetStages().to(args.device)
        stages = args.stages
        rf_label = {s: f"{STAGE_RF[s]}px" for s in stages}

    # ---- quiet injection sites (lowest hot-fraction), same selection as the gate ----
    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    pool_idx = rng.choice(arr.shape[0], size=min(args.n_frames * 4, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[pool_idx])
    del arr

    pre_pool = np.array([preprocess_raw(raw_pool[i], preproc) for i in range(len(raw_pool))])
    hot = np.array([(f > 5.0).mean() for f in pre_pool])
    quiet_sel = np.argsort(hot)[:args.n_frames]
    raw_quiet = raw_pool[quiet_sel]
    f_quiet = pre_pool[quiet_sel]
    print(f"  Quiet frames: {len(quiet_sel)}")

    results = {}  # (stage, snr) -> (on_mean, off_mean, ratio)
    for snr in args.snr_list:
        f_inj = np.array([
            preprocess_raw(
                inject_narrowband_on_only(raw_quiet[i], snr=snr,
                                          drift_rate=args.drift_rate, seed=args.seed + i),
                preproc)
            for i in range(len(raw_quiet))
        ])
        for stage in stages:
            on_resp, off_resp = stage_displacement(model, f_quiet, f_inj, stage, args.device)
            on_m, off_m = float(on_resp.mean()), float(off_resp.mean())
            ratio = on_m / (off_m + 1e-12)
            results[(stage, snr)] = (on_m, off_m, ratio)

    # ---- report ----
    print(f"\n{'='*72}")
    print("ON->OFF ROW LEAKAGE  (ON-only injection; ON/OFF = feature-row response)")
    print("higher ON/OFF ratio = better localization; ratio -> 1 = diffuse/leaky")
    print(f"{'='*72}")
    for snr in args.snr_list:
        print(f"\n  SNR = {snr:g}")
        print(f"    {'stage':>8s}  {'vert.RF':>7s}  {'ON_resp':>9s}  {'OFF_resp':>9s}  {'ON/OFF':>7s}")
        for stage in stages:
            on_m, off_m, ratio = results[(stage, snr)]
            print(f"    {stage:>8s}  {rf_label[stage]:>7s}  {on_m:9.4f}  {off_m:9.4f}  {ratio:7.2f}")

    # Headline: ratio trend across stages at the highest SNR (clearest signal).
    ref_snr = max(args.snr_list)
    print(f"\n{'='*72}\nVERDICT (ratio trend @ SNR {ref_snr:g}, deepest->shallowest)\n{'='*72}")
    ratios = {s: results[(s, ref_snr)][2] for s in stages}
    for stage in stages:
        tag = ("LOCALIZED" if ratios[stage] >= 3.0
               else "partial" if ratios[stage] >= 1.5 else "DIFFUSE")
        print(f"  {stage:>8s} (RF~{rf_label[stage]}): ON/OFF = {ratios[stage]:.2f}  [{tag}]")
    if args.architecture == "vit_mae":
        print("\n  Read: this is the domain-matched teacher's localizing ON/OFF ratio at its\n"
              "  deployed block — the apples-to-apples reference for the ResNet stages (which\n"
              "  stayed ~2-3, RF-invariant, 2026-07-16). A markedly higher ratio here confirms\n"
              "  localization is a domain-matching property, not a receptive-field one.")
    else:
        print("\n  Read: if a shallow stage recovers a high ratio while layer3 is ~1, the RF is\n"
              "  the lever and the small-RF teacher is worth a distill+retrain run. If every\n"
              "  stage stays near 1, the ImageNet-P leakage is not RF-driven -> that route is a\n"
              "  dead end (keep the domain-matched teacher / try a shallow anisotropic teacher).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_dir / f"teacher_row_leakage_{args.architecture}.npz"
    np.savez(
        out_npz,
        architecture=args.architecture,
        stages=np.array(stages),
        snr_list=np.array(args.snr_list),
        on_resp=np.array([[results[(s, snr)][0] for s in stages] for snr in args.snr_list]),
        off_resp=np.array([[results[(s, snr)][1] for s in stages] for snr in args.snr_list]),
        ratio=np.array([[results[(s, snr)][2] for s in stages] for snr in args.snr_list]),
    )
    print(f"\nSaved -> {out_npz}")


if __name__ == "__main__":
    main()
