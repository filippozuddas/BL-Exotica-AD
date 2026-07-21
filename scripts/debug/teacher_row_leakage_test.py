"""ON->OFF row-leakage probe for a UDMA teacher's feature grid.

Measures, without any training, whether a candidate teacher LOCALIZES an
ON-only signal or bleeds it into the OFF rows. Inject a narrowband line into
obs {0,2,4} of a quiet frame, take the feature displacement
``||P(x+s) - P(x)||`` per (row, col) cell, and compare ON feature-rows against
OFF feature-rows. A localizing teacher keeps the displacement in ON rows (high
ratio); a leaky one spreads it (ratio -> 1).

The measurement is clean because ``inject_narrowband_on_only`` leaves OFF
frames byte-identical and ``preprocess_raw`` normalizes per observation, so any
OFF-row motion is pure receptive-field leakage.

``--architecture`` selects the target under test, each isolating one variable:

    resnet18 (default)  out-of-domain CNN, swept across stem/layer1/layer2/
                        layer3 -- tests the receptive-field hypothesis
    vit_mae             domain-matched teacher, the localizing reference
    udma_student        a trained student's own trunk -- tests architecture
                        vs. training target
    convnextv2_fcmae    ImageNet domain, reconstruction pretraining -- tests
                        objective vs. domain (needs ``pip install timm``)

Cost: n_frames * len(snr_list) * len(stages) frozen forwards. No training.

Outcome (2026-07-16): all three hypotheses refuted; only domain-matched
training localizes. Full record in ``docs/03_teacher-localization.md``.

Usage (run on the data host; --cache must point at a preprocessed cache):

    # out-of-domain CNN, receptive-field sweep -- default
    python scripts/debug/teacher_row_leakage_test.py --cache <cache_dir>

    # domain-matched ViT-MAE teacher, read at block 3
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture vit_mae --layer 3 \
        --checkpoint <vit_mae.ckpt> \
        --model_config configs/model/vit_mae.yaml --cache <cache_dir>

    # a trained UDMA student's trunk
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture udma_student --student ae \
        --checkpoint <udma.ckpt> \
        --model_config configs/model/udma.yaml --cache <cache_dir>

    # ConvNeXt V2 / FCMAE control
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture convnextv2_fcmae --cache <cache_dir>
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


CONVNEXT_STAGES = ("stem", "stage0", "stage1", "stage2")
# Downsampling factor labels only (not pixel RF — ConvNeXt's 7x7 depthwise
# kernels stacked over variable block counts [3,3,9,3] make a precise
# theoretical pixel RF non-trivial to state correctly; the /N factor is what
# actually matters for reading this alongside ResNetStages, since stage2's
# grid (6,64) at /16 lines up with ResNet's layer3, the direct comparison
# point). stage3 (/32) is excluded: its (3,32) grid has fewer rows than the
# 6 cadence observations, incompatible with row_on_off_masks.
CONVNEXT_RF = {"stem": "/4", "stage0": "/4", "stage1": "/8", "stage2": "/16"}


class ConvNeXtV2Stages(nn.Module):
    """Frozen ConvNeXt V2 (``convnextv2_tiny.fcmae``, timm) exposing features at
    each depth stage — the domain-vs-objective isolating control
    (2026-07-16): SAME domain (ImageNet) and SAME family (CNN) as
    :class:`ResNetStages`, but pretrained via masked-patch RECONSTRUCTION
    (FCMAE, self-supervised, no classification head) instead of
    classification. If this localizes where ResNet-18 doesn't, the deciding
    variable is the pretraining OBJECTIVE, not the domain; if it stays
    diffuse like ResNet-18, domain-matching is confirmed necessary regardless
    of objective. Input (B,1,96,1024) replicated to 3 channels, no ImageNet
    mean/std renorm — same simplification as :class:`ResNetStages`, and
    fully convolutional (no positional embeddings to reconcile with a
    non-224x224 input, unlike a ViT-MAE-on-ImageNet alternative)."""

    def __init__(self, model_name: str = "convnextv2_tiny.fcmae"):
        super().__init__()
        import timm
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.stem = m.stem
        self.stages_list = m.stages
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ConvNeXtV2Stages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) at the requested stage."""
        h = self.stem(x.repeat(1, 3, 1, 1))
        if stage == "stem":
            return h
        h = self.stages_list[0](h)
        if stage == "stage0":
            return h
        h = self.stages_list[1](h)
        if stage == "stage1":
            return h
        h = self.stages_list[2](h)
        if stage == "stage2":
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


class UDMAStudentStages(nn.Module):
    """A trained UDMA student's own output feature grid, exposing the SAME
    ``features(x, stage) -> (B,C,nh,nw)`` interface as :class:`ResNetStages` /
    :class:`ViTMAEStages`. One pseudo-stage (the student's final projection
    head output, ``(B, teacher.channels, 6, 64)``) — no depth sweep, since the
    question here isn't receptive field, it's "does the SAME student
    architecture localize when trained against a domain-matched target vs an
    out-of-domain one" (2026-07-16, see udma_teacher_rf_leakage_refuted's
    confound note). Loads the full UDMA model (teacher + both students) via
    ``build_autoencoder`` from ``model_config`` (``architecture: udma``,
    either ``teacher.type: vit_mae`` or ``cnn_distilled``) and a Lightning
    checkpoint, exactly like ``encode_separation_test.load_model``, then reads
    out only the requested student's raw output (``student_ae`` or
    ``student_mem``'s projection head, ignoring the teacher's own features and
    the disagreement maps entirely — this measures the STUDENT's learned
    localization, not the teacher's)."""

    def __init__(self, checkpoint, model_cfg, device, student="ae"):
        super().__init__()
        from src.models import build_autoencoder
        model = build_autoencoder(INPUT_SHAPE, model_cfg, loss="mse")
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
        state = {k.replace("model.", "", 1): v
                 for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state)
        model.eval().to(device)
        if not hasattr(model, "student_ae"):
            raise SystemExit("--architecture udma_student requires a UDMA checkpoint "
                             "(architecture: udma in --model_config).")
        self.model = model
        self.student = {"ae": model.student_ae, "mem": model.student_mem}[student]
        self.grid_size = model.teacher.grid_size
        self.stage_name = f"student_{student}"
        self.eval()

    def train(self, mode: bool = True) -> "UDMAStudentStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str = None) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) student output feature grid."""
        out = self.student(x)
        if isinstance(out, tuple):  # memory student returns (out, attention)
            out = out[0]
        return out


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
    p.add_argument("--architecture",
                   choices=["resnet18", "convnextv2_fcmae", "vit_mae", "udma_student"],
                   default="resnet18",
                   help="resnet18: out-of-domain P, RF sweep across stem/layer1/2/3 (default). "
                        "convnextv2_fcmae: SAME domain+family as resnet18, but masked-"
                        "reconstruction pretraining instead of classification — isolates "
                        "objective from domain (needs `pip install timm` on the runner). "
                        "vit_mae: domain-matched teacher, single stage at --layer (localizing "
                        "reference) — requires --checkpoint/--model_config. udma_student: a "
                        "trained UDMA student's own output grid — requires --checkpoint/"
                        "--model_config (architecture: udma) and --student.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Required for --architecture vit_mae/udma_student; ignored otherwise.")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml",
                   help="Model config: vit_mae.yaml (patch_size -> token grid) for --architecture "
                        "vit_mae, or udma.yaml/udma_cnn_teacher.yaml for udma_student.")
    p.add_argument("--layer", type=int, default=3,
                   help="ViT-MAE transformer block to read (1-indexed; -1 = final). Default 3 = "
                        "the deployed teacher_layer in configs/model/udma.yaml. vit_mae only.")
    p.add_argument("--student", choices=["ae", "mem"], default="ae",
                   help="Which UDMA student to read (udma_student only).")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--n_frames", type=int, default=200, help="Quiet injection sites.")
    p.add_argument("--snr_list", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES,
                   help="ResNet depth stages to sweep (resnet18 only).")
    p.add_argument("--convnext_stages", nargs="+", default=list(CONVNEXT_STAGES),
                   choices=CONVNEXT_STAGES,
                   help="ConvNeXt V2 depth stages to sweep (convnextv2_fcmae only).")
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
    elif args.architecture == "udma_student":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture udma_student.")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        teacher_type = model_cfg.get("teacher", {}).get("type", "vit_mae")
        print(f"Loading UDMA student '{args.student}' from {args.checkpoint} "
              f"(teacher.type={teacher_type})")
        model = UDMAStudentStages(args.checkpoint, model_cfg, args.device, student=args.student)
        stages = [model.stage_name]
        rf_label = {model.stage_name: f"teacher={teacher_type}"}
        print(f"  Student output grid: {model.grid_size}")
    elif args.architecture == "convnextv2_fcmae":
        print("Loading ConvNeXt V2 (ImageNet, FCMAE self-supervised, frozen) — "
              "domain-vs-objective isolating control")
        model = ConvNeXtV2Stages().to(args.device)
        stages = args.convnext_stages
        rf_label = {s: CONVNEXT_RF[s] for s in stages}
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
    elif args.architecture == "udma_student":
        print("\n  Read: this is the SAME student CNN architecture (build_encoder trunk), only\n"
              "  the training TARGET differs by checkpoint (vit_mae domain-matched vs\n"
              "  cnn_distilled out-of-domain). Compare this ratio against the SAME test run on\n"
              "  the other checkpoint: if the vit_mae-target student localizes and the\n"
              "  cnn_distilled-target student doesn't, that's a second, architecture-independent\n"
              "  confirmation the TARGET's domain (not the CNN trunk) is the lever — corroborates\n"
              "  the teacher-level finding without any new training.")
    elif args.architecture == "convnextv2_fcmae":
        print("\n  Read: this is the domain-vs-objective isolating control (2026-07-16) — SAME\n"
              "  domain (ImageNet) and family (CNN) as resnet18, but pretrained via masked-patch\n"
              "  RECONSTRUCTION (FCMAE) instead of classification. Compare stage2 (/16 grid,\n"
              "  matches resnet18's layer3) directly against resnet18 layer3's ratio (~2.17):\n"
              "  if this localizes markedly better, the pretraining OBJECTIVE is the lever, not\n"
              "  domain -> a CNN teacher distilled from THIS backbone could be paper-faithful\n"
              "  (out-of-domain, generic) AND localizing, no audio/domain-closer backbone hunt\n"
              "  needed. If it stays diffuse like resnet18, domain-matching is confirmed\n"
              "  necessary regardless of pretraining objective.")
    else:
        print("\n  Read: if a shallow stage recovers a high ratio while layer3 is ~1, the RF is\n"
              "  the lever and the small-RF teacher is worth a distill+retrain run. If every\n"
              "  stage stays near 1, the ImageNet-P leakage is not RF-driven -> that route is a\n"
              "  dead end (keep the domain-matched teacher / try a shallow anisotropic teacher).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tag = args.architecture
    if args.architecture == "udma_student":
        out_tag = f"udma_student_{args.student}_{args.checkpoint.parents[1].name}"
    out_npz = args.out_dir / f"teacher_row_leakage_{out_tag}.npz"
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
