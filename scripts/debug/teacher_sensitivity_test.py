"""Teacher-fitness pre-flight for the UDMA build — is the frozen ViT-MAE a
usable teacher? (docs/2026-07-05_udma_design_spec.md, Q1 gate; run BEFORE
implementing/training anything.)

For the UDMA student-teacher mechanism, "good feature extractor" does NOT mean
"good ETI/RFI classifier" — discrimination comes from the students failing on
unseen content. The teacher needs exactly two measurable properties, plus a
mechanism preview, each with a PRE-REGISTERED gate:

  T1. NO COLLAPSE / RANK. Token features must vary across inputs with enough
      effective dimensionality to be a non-trivial regression target.
      Gate G1: rel-std >= 0.05 (same convention as the encoder collapse check)
      AND participation-ratio rank >= 16 (of embed_dim).

  T2. RESPONSIVENESS (paired, token-level). Inject a line into a quiet frame
      and measure the per-token feature displacement ||T(x+s) - T(x)|| on the
      SAME background: tokens crossed by the line must move more than tokens
      that aren't. If T barely responds, students predict it trivially and the
      S-T gap is zero regardless of training -> UDMA dead with this teacher.
      Gate G2: displacement AUC (affected vs unaffected tokens) >= 0.80 at
      SNR 20 (where recon scoring already detects, so the tokens MUST carry it).

  T3. MECHANISM PREVIEW (linear student, closed-form — no training). Fit a
      ridge regression from raw patch pixels to teacher tokens on normal data
      (quiet+RFI), then measure the prediction residual on (a) held-out normal
      tokens — predictability baseline and known-RFI FP preview — and (b) line
      tokens. This is a weak lower bound of the real conv students: if even a
      LINEAR student shows the residual gap, the mechanism is alive.
      Gate G3a: token residual AUC (affected vs held-out normal) >= 0.70 at SNR 20.
      Gate G3b: frame-level preview (topk of the residual map, injected-into-quiet
      vs REAL RFI frames, pooled SNRs, energy caliper-matched) AUC >= 0.60.

Decision branches printed at the end:
  - G2 fail                -> teacher blind to lines: try --layer 3/4 (mid-layer
                              features), another checkpoint, else spec v2 (CNN
                              teacher / ImageNet backbone).
  - G2 pass, G3 fail       -> teacher responsive but too predictable (features
                              too smooth): try a mid layer; else v2.
  - all pass               -> teacher fit; proceed with the UDMA build.

Cost: ~1.5k teacher forwards + one 257x257 solve — minutes on GPU, no training.

Usage (server, not dev machine):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/content/filippo/BL-Exotica-AD \
    python scripts/debug/teacher_sensitivity_test.py \
        --checkpoint outputs/training/20260624_084754_057f87c/checkpoints/epoch=006-val_loss=2.1715.ckpt \
        --cache /content/nvme_esterno/filippo/BL-Exotica-AD/data/processed/cache_gbt_fine
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.debug.injection_vs_rfi_test import preprocess_raw, inject_narrowband_on_only
from scripts.debug.encode_separation_test import (
    load_model, frame_energy, frame_stats, morphology_matched_energy_recon, cohens_d,
)

INPUT_SHAPE = (96, 1024, 1)

# |delta-pixel| thresholds (standardized units) splitting tokens into
# affected / unaffected by the injection. Per-frame median/MAD renormalisation
# shifts ALL pixels by O(1e-2) when a line is added, so the unaffected band
# must sit above that drift but far below a line pixel (O(1) even at SNR 5).
AFFECTED_THR = 0.5
UNAFFECTED_THR = 0.05


class _TeacherCNNGateAdapter:
    """Minimal ``patch_embed``/``pos_embed``/``encoder.layers`` wrapper so
    ``encode_tokens_layer()`` runs on a distilled :class:`TeacherCNN`
    unmodified — same trick as ``ResNetTeacher``: T has no transformer blocks
    of its own, ``patch_embed`` does the whole conv stack and ``encoder.layers``
    is empty. Uses T's RAW (pre-Norm) output — ``fit_udma_teacher_norm.py``
    hasn't run yet at gate time, ``mu``/``sigma`` are still identity."""

    def __init__(self, teacher_cnn):
        from scripts.debug.resnet_teacher import _EmptyEncoder
        self._t = teacher_cnn
        self.grid_size = teacher_cnn.grid_size
        self.pos_embed = 0.0
        self.encoder = _EmptyEncoder()

    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._t._raw_tokens(x)  # (B, C, nh, nw)
        b, c, nh, nw = raw.shape
        return raw.permute(0, 2, 3, 1).reshape(b, nh * nw, c)


@torch.no_grad()
def encode_tokens_layer(model, frames: np.ndarray, device: str,
                        layer: int = -1, batch: int = 64) -> np.ndarray:
    """(N, H, W) preprocessed -> (N, num_patches, D) token features taken from
    transformer block ``layer`` (1-indexed; -1 = final block incl. terminal
    norm; 0 = patch+positional embedding, before any block)."""
    n_layers = len(model.encoder.layers)
    k = n_layers if layer in (-1, n_layers) else layer
    if not 0 <= k <= n_layers:
        raise SystemExit(f"--layer must be in 0..{n_layers} or -1, got {layer}")
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).float().unsqueeze(1).to(device)
        tok = model.patch_embed(x) + model.pos_embed
        for j, blk in enumerate(model.encoder.layers, start=1):
            if j > k:
                break
            tok = blk(tok)
        if k == n_layers and getattr(model.encoder, "norm", None) is not None:
            tok = model.encoder.norm(tok)
        out.append(tok.cpu().numpy())
    return np.concatenate(out, axis=0)


def patch_pixels(frames: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """(N, H, W) -> (N, nh*nw, ph*pw), row-major patch order — bit-identical to
    ``vit_mae.patchify`` for C=1, so patch k aligns with token k."""
    n, h, w = frames.shape
    nh, nw = h // ph, w // pw
    x = frames.reshape(n, nh, ph, nw, pw)
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(n, nh * nw, ph * pw)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", choices=["vit_mae", "resnet18", "cnn_distilled"], default="vit_mae",
                   help="Teacher candidate under test. 'resnet18' gates the paper-faithful "
                        "out-of-domain P (docs/2026-07-14_paper_alignment_plan.md, D6/D7) — "
                        "ignores --checkpoint/--model_config, no local weights needed "
                        "(auto-downloads ImageNet weights via torchvision on first use). "
                        "'cnn_distilled' gates T after distillation (5.2, scripts/distill_teacher.py) "
                        "— --checkpoint is T's trunk-only checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Required for --architecture vit_mae/cnn_distilled; ignored for resnet18.")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--n_frames", type=int, default=200,
                   help="Frames per class (quiet / RFI).")
    p.add_argument("--snr_list", type=float, nargs="+", default=[5, 7, 10, 15, 20, 25, 40])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--layer", type=int, default=-1,
                   help="Transformer block to read tokens from (1-indexed; -1 = final). "
                        "Mid layers (3-4 of 6) are the fallback if the final layer is "
                        "unresponsive or too predictable.")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--frame_topk", type=int, default=8,
                   help="Tokens kept in the frame-level preview score (~2% of 384).")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/teacher_sensitivity")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise SystemExit("scikit-learn required for this diagnostic.")

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]

    if args.architecture == "resnet18":
        from scripts.debug.resnet_teacher import ResNetTeacher
        print("Loading ResNet-18 (ImageNet, frozen) as teacher candidate P")
        model = ResNetTeacher().to(args.device)
        ph = INPUT_SHAPE[0] // model.grid_size[0]
        pw = INPUT_SHAPE[1] // model.grid_size[1]
        ckpt_tag = "resnet18_imagenet"
    elif args.architecture == "cnn_distilled":
        from src.models.udma import _load_teacher_cnn
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture cnn_distilled "
                             "(T's trunk-only checkpoint from scripts/distill_teacher.py).")
        print(f"Loading distilled CNN teacher T from {args.checkpoint}")
        # Reuse build_udma's own loader (not a re-implementation): single
        # source of truth for the checkpoint schema/fallback rules.
        t_model = _load_teacher_cnn({"checkpoint": str(args.checkpoint)}, INPUT_SHAPE).to(args.device)
        model = _TeacherCNNGateAdapter(t_model)
        ph = INPUT_SHAPE[0] // model.grid_size[0]
        pw = INPUT_SHAPE[1] // model.grid_size[1]
        ckpt_tag = args.checkpoint.stem
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture vit_mae.")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        print(f"Loading teacher candidate from {args.checkpoint}")
        model = load_model(args.checkpoint, model_cfg, args.device, require_encode=False)
        if not (hasattr(model, "patch_embed") and hasattr(model, "encoder")
                and hasattr(model.encoder, "layers")):
            raise SystemExit("Teacher test requires the ViT-MAE backbone "
                             "(architecture: vit_mae in --model_config).")
        ph, pw = model_cfg["patch_size"]
        ckpt_tag = args.checkpoint.stem

    nh, nw = INPUT_SHAPE[0] // ph, INPUT_SHAPE[1] // pw
    n_tok = nh * nw
    layer_tag = "final" if args.layer == -1 else f"block{args.layer}"
    print(f"  Token grid: ({nh}, {nw}) = {n_tok} tokens, layer = {layer_tag}")

    # ---- data: quiet (injection sites) + RFI (negatives / normal mix) ----
    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    pool_idx = rng.choice(arr.shape[0], size=min(args.n_frames * 4, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[pool_idx])
    del arr

    pre_pool = np.array([preprocess_raw(raw_pool[i], preproc) for i in range(len(raw_pool))])
    hot = np.array([(f > 5.0).mean() for f in pre_pool])
    order = np.argsort(hot)
    quiet_sel = order[:args.n_frames]
    rfi_sel = order[-args.n_frames:]
    raw_quiet, f_quiet = raw_pool[quiet_sel], pre_pool[quiet_sel]
    f_rfi = pre_pool[rfi_sel]
    print(f"  Quiet: {len(quiet_sel)}  RFI: {len(rfi_sel)}")

    print("Encoding teacher tokens (clean frames)...")
    tok_quiet = encode_tokens_layer(model, f_quiet, args.device, args.layer)  # (Nq, n_tok, D)
    tok_rfi = encode_tokens_layer(model, f_rfi, args.device, args.layer)
    D = tok_quiet.shape[2]

    # ================= T1: collapse check (+ informative ranks) =================
    # AMENDED 2026-07-05 (see spec Q1): the original ">= 16 pooled PR-rank" gate
    # was mis-specified — pooled token covariance is dominated by positional-
    # embedding structure that is CONSTANT per position, i.e. free to predict
    # for a student, so it must not count against the teacher. G1 now gates on
    # collapse only (rel-std, dead dims); ranks are reported as informative,
    # with the per-position-CENTERED rank as the content-dimensionality figure
    # (hard warning only if it is degenerate, < 4).
    print(f"\n{'='*64}\nT1. COLLAPSE CHECK  (quiet tokens, layer={layer_tag})\n{'='*64}")
    flat = tok_quiet.reshape(-1, D)
    sub = flat[rng.choice(len(flat), size=min(20000, len(flat)), replace=False)]
    sd = sub.std(axis=0)
    mu_norm = float(np.linalg.norm(sub.mean(axis=0)))
    rel_std = float(sd.mean()) / (mu_norm / np.sqrt(D) + 1e-12)
    dead = int((sd < 1e-4).sum())

    def pr_rank_of(rows: np.ndarray) -> float:
        ev = np.linalg.eigvalsh(np.cov((rows - rows.mean(0)).T))
        ev = np.clip(ev, 0, None)
        return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12))

    pr_pooled = pr_rank_of(sub)
    cflat = (tok_quiet - tok_quiet.mean(axis=0, keepdims=True)).reshape(-1, D)
    csub = cflat[rng.choice(len(cflat), size=min(20000, len(cflat)), replace=False)]
    pr_content = pr_rank_of(csub)
    print(f"  mean per-dim std      : {sd.mean():.5f}  (dead dims: {dead}/{D})")
    print(f"  rel. variation        : {rel_std:.4f}   (gate: >= 0.05)")
    print(f"  PR-rank pooled        : {pr_pooled:.1f} / {D}   (informative — inflated/deflated by pos-embed)")
    print(f"  PR-rank content       : {pr_content:.1f} / {D}   (per-position-centered; warn if < 4)")
    g1 = rel_std >= 0.05 and dead <= 0.1 * D
    if pr_content < 4:
        print("  WARNING: content rank < 4 — near-degenerate regression target; "
              "consider PCA-projected target / channel weighting (spec Q4 lever).")
    print(f"  G1 (collapse): {'PASS' if g1 else 'FAIL'}")

    # ================= T2: paired token displacement =================
    print(f"\n{'='*64}\nT2. RESPONSIVENESS — paired token displacement ||T(x+s) − T(x)||\n{'='*64}")
    print(f"  affected = token with max|Δpixel| > {AFFECTED_THR}; "
          f"unaffected < {UNAFFECTED_THR}; in-between excluded.")
    hit_col = f"top{args.frame_topk}-hit"
    print(f"\n  {'SNR':>5s}  {'n_aff/frame':>11s}  {'AUC':>6s}  {'cohen_d':>8s}  {hit_col:>9s}")
    inj_frames_by_snr, aff_by_snr, t2_auc = {}, {}, {}
    g2_ref_snr = 20.0
    for snr in args.snr_list:
        inj = np.array([preprocess_raw(
            inject_narrowband_on_only(raw_quiet[i], snr=snr, drift_rate=args.drift_rate,
                                      seed=args.seed + i), preproc)
            for i in range(len(raw_quiet))])
        inj_frames_by_snr[snr] = inj
        delta = np.abs(inj - f_quiet)
        pmax = patch_pixels(delta, ph, pw).max(axis=2)          # (Nq, n_tok)
        affected = pmax > AFFECTED_THR
        unaffected = pmax < UNAFFECTED_THR
        aff_by_snr[snr] = affected
        tok_inj = encode_tokens_layer(model, inj, args.device, args.layer)
        disp = np.linalg.norm(tok_inj - tok_quiet, axis=2)      # (Nq, n_tok)
        d_aff, d_un = disp[affected], disp[unaffected]
        if len(d_aff) < 10:
            print(f"  {snr:5.0f}  (too few affected tokens — skipped)")
            continue
        y = np.concatenate([np.ones(len(d_aff)), np.zeros(len(d_un))])
        auc = float(roc_auc_score(y, np.concatenate([d_aff, d_un])))
        t2_auc[snr] = auc
        topk_ids = np.argsort(disp, axis=1)[:, -args.frame_topk:]
        hit = float(np.mean([affected[i, topk_ids[i]].any() for i in range(len(disp))]) * 100)
        print(f"  {snr:5.0f}  {affected.sum(1).mean():11.1f}  {auc:6.3f}  "
              f"{cohens_d(d_aff, d_un):8.2f}  {hit:8.1f}%")
    g2 = t2_auc.get(g2_ref_snr, 0.0) >= 0.80
    print(f"\n  G2 (displacement AUC >= 0.80 @ SNR {g2_ref_snr:.0f}): "
          f"{'PASS' if g2 else 'FAIL'}  ({t2_auc.get(g2_ref_snr, float('nan')):.3f})")

    # ================= T3: linear-student mechanism preview =================
    print(f"\n{'='*64}\nT3. LINEAR-STUDENT PREVIEW — ridge patch-pixels -> teacher token\n{'='*64}")
    n_fit_q = int(0.7 * len(f_quiet))
    n_fit_r = int(0.7 * len(f_rfi))
    fit_frames = np.concatenate([f_quiet[:n_fit_q], f_rfi[:n_fit_r]])
    fit_tok = np.concatenate([tok_quiet[:n_fit_q], tok_rfi[:n_fit_r]])
    ho_q_slice = slice(n_fit_q, len(f_quiet))
    ho_r_slice = slice(n_fit_r, len(f_rfi))

    Xf = patch_pixels(fit_frames, ph, pw).reshape(-1, ph * pw)
    Yf = fit_tok.reshape(-1, D)
    x_mu, x_sd = Xf.mean(0), Xf.std(0) + 1e-6
    y_mu, y_sd = Yf.mean(0), Yf.std(0) + 1e-6   # == the UDMA Norm(T) statistics
    Xf = (Xf - x_mu) / x_sd
    Yf = (Yf - y_mu) / y_sd
    Xf = np.concatenate([Xf, np.ones((len(Xf), 1))], axis=1)
    A = Xf.T @ Xf + args.ridge_alpha * np.eye(Xf.shape[1])
    W = np.linalg.solve(A, Xf.T @ Yf)
    print(f"  ridge fit on {len(Xf)} normal tokens "
          f"({n_fit_q} quiet + {n_fit_r} RFI frames), alpha={args.ridge_alpha}")

    def residual_map(frames, tokens):
        """(N,H,W), (N,n_tok,D) -> (N, n_tok) mean squared prediction residual
        in Norm(T) space — exactly the quantity a UDMA student map measures."""
        X = patch_pixels(frames, ph, pw).reshape(-1, ph * pw)
        X = (X - x_mu) / x_sd
        X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        Y = (tokens.reshape(-1, D) - y_mu) / y_sd
        r = ((Y - X @ W) ** 2).mean(axis=1)
        return r.reshape(len(frames), -1)

    r_ho_q = residual_map(f_quiet[ho_q_slice], tok_quiet[ho_q_slice])
    r_ho_r = residual_map(f_rfi[ho_r_slice], tok_rfi[ho_r_slice])
    print(f"  held-out residual median: quiet {np.median(r_ho_q):.4f} | "
          f"RFI {np.median(r_ho_r):.4f}  (ratio {np.median(r_ho_r)/np.median(r_ho_q):.2f}x "
          f"— known-RFI FP preview; the conv students should shrink this)")

    normal_pool = np.concatenate([r_ho_q.ravel(), r_ho_r.ravel()])
    print(f"\n  {'SNR':>5s}  {'tokAUC':>7s}  {'frameAUC(vs real RFI)':>22s}")
    neg_frame = np.sort(r_ho_r, axis=1)[:, -args.frame_topk:].mean(1)
    en_rfi_ho = frame_energy(f_rfi[ho_r_slice])
    st_rfi_ho = frame_stats(f_rfi[ho_r_slice])
    t3_tok_auc, pos_pool, en_pool, st_pool = {}, [], [], []
    for snr in args.snr_list:
        inj_ho = inj_frames_by_snr[snr][ho_q_slice]
        tok_inj_ho = encode_tokens_layer(model, inj_ho, args.device, args.layer)
        r_inj = residual_map(inj_ho, tok_inj_ho)
        aff = aff_by_snr[snr][ho_q_slice]
        if aff.sum() < 10:
            continue
        y = np.concatenate([np.ones(int(aff.sum())), np.zeros(len(normal_pool))])
        s = np.concatenate([r_inj[aff], normal_pool])
        tok_auc = float(roc_auc_score(y, s))
        t3_tok_auc[snr] = tok_auc
        pos_frame = np.sort(r_inj, axis=1)[:, -args.frame_topk:].mean(1)
        yf = np.concatenate([np.ones(len(pos_frame)), np.zeros(len(neg_frame))])
        fr_auc = float(roc_auc_score(yf, np.concatenate([pos_frame, neg_frame])))
        pos_pool.append(pos_frame)
        en_pool.append(frame_energy(inj_ho))
        st_pool.append(frame_stats(inj_ho))
        print(f"  {snr:5.0f}  {tok_auc:7.3f}  {fr_auc:22.3f}")

    g3a = t3_tok_auc.get(g2_ref_snr, 0.0) >= 0.70
    print(f"\n  G3a (token residual AUC >= 0.70 @ SNR {g2_ref_snr:.0f}): "
          f"{'PASS' if g3a else 'FAIL'}  ({t3_tok_auc.get(g2_ref_snr, float('nan')):.3f})")

    mm = morphology_matched_energy_recon(
        np.concatenate(pos_pool), np.concatenate(en_pool), np.concatenate(st_pool),
        neg_frame, en_rfi_ho, st_rfi_ho, seed=args.seed) if pos_pool else None
    g3b = False
    if mm is None or "error" in mm:
        print(f"  G3b matched-energy preview: SKIPPED ({mm['error'] if mm else 'no pooled positives'})")
    else:
        print(f"  G3b matched-energy preview ({mm['n_per_class']}/class, caliper {mm['caliper']}): "
              f"energy {mm['energy_only']:.3f} | trivial {mm['trivial']:.3f} | "
              f"linear-student {mm['recon']:.3f}")
        g3b = mm["recon"] >= 0.60
        print(f"  G3b (>= 0.60): {'PASS' if g3b else 'FAIL'}")

    # ================= verdict =================
    print(f"\n{'='*64}\nVERDICT (teacher = {ckpt_tag}, layer = {layer_tag})\n{'='*64}")
    for name, ok in [("G1 collapse", g1), ("G2 responsiveness", g2),
                     ("G3a token residual", g3a), ("G3b frame preview", g3b)]:
        print(f"  {name:22s}: {'PASS' if ok else 'FAIL'}")
    if not g2:
        print("  -> teacher NOT responsive to lines at token level: rerun with --layer 3/4 "
              "and/or another checkpoint; if none passes G2, the ViT-MAE teacher is unfit "
              "-> spec v2 (small-RF CNN teacher / ImageNet backbone).")
    elif not (g3a and g3b):
        print("  -> teacher responsive but too PREDICTABLE for a linear student. This is "
              "not necessarily fatal (conv students are stronger than ridge), but derisk "
              "first: rerun with --layer 3 and --layer 4; prefer the layer with the best "
              "G3 without losing G2. If no layer clears G3, fall back to spec v2 teacher.")
    elif not g1:
        print("  -> direct mechanism tests PASS but the teacher features look collapsed "
              "(G1) — contradictory readout; inspect T1 numbers before proceeding.")
    else:
        print("  -> teacher FIT: proceed with the UDMA build (spec Q1 confirmed).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{ckpt_tag}_{layer_tag}".replace("=", "").replace(".", "p")
    out_npz = args.out_dir / f"teacher_fitness_{tag}.npz"
    np.savez(out_npz,
             rel_std=rel_std, pr_rank_pooled=pr_pooled, pr_rank_content=pr_content,
             snr_list=np.array(sorted(t2_auc)),
             t2_auc=np.array([t2_auc[s] for s in sorted(t2_auc)]),
             t3_tok_auc=np.array([t3_tok_auc.get(s, np.nan) for s in sorted(t2_auc)]),
             matched=np.array([mm.get("energy_only", np.nan), mm.get("trivial", np.nan),
                               mm.get("recon", np.nan)]) if mm and "error" not in mm
                             else np.array([np.nan] * 3),
             gates=np.array([g1, g2, g3a, g3b]))
    print(f"\nSaved → {out_npz}")


if __name__ == "__main__":
    main()
