# BL-Exotica-AD

Unsupervised anomaly detection pipeline for searching technosignatures in Green Bank Telescope (GBT) observations of sources from the [Breakthrough Listen Exotica Catalog](https://arxiv.org/abs/2006.11304) (Lacki et al. 2020).

**Author:** Filippo Simone Zuddas (Uni. Cagliari)
**Mentors:** Maura Pilia, Vishal Gajjar
**Program:** Breakthrough Listen Summer Internship 2026

## Motivation

Traditional SETI searches rely on matched-filter algorithms like [turboSETI](https://github.com/UCBerkeleySETI/turbo_seti), which are optimized for narrowband drifting signals. However, a genuine technosignature could have any morphology — wideband, pulsed, modulated, or transient. Searching for every possible signal type individually is impractical.

This project extends the autoencoder-based approach of [Ma et al. (2023)](https://arxiv.org/abs/2301.12670) to search for a **broader range of technosignature-like signals**. Instead of using a supervised classifier (ContrastiveVAE + Random Forest) tied to known signal classes, we use an **unsupervised autoencoder** trained on real noise and RFI. Signals with anomalous morphology produce high reconstruction error and are flagged as candidates — no prior signal model required.

The search targets "exotica" sources: rare, extreme, or unusual astrophysical objects that expand SETI target lists beyond traditional nearby-star surveys.

## Method

A convolutional autoencoder learns the distribution of normal radio telescope data (noise + RFI). At inference, each spectrogram snippet is reconstructed by the model; the **reconstruction error** serves as the anomaly score. Snippets that the model cannot faithfully reconstruct are flagged as candidates for further inspection.

### Model Backbones

Four backbone architectures are available, selected by config:

| Backbone | Config | Description |
|----------|--------|-------------|
| **Autoencoder** | `convae.yaml` | Deterministic CNN AE. Full reconstruction error. Comparison baseline. |
| **CNN-MAE** | `convae_mae.yaml` | CNN Masked Autoencoder. Masks ~75% of patches during training, loss on masked positions only. Addresses the "too-good reconstruction" failure mode on locally-regular signals. |
| **VAE** | `convae.yaml` + `variational: true` | Variational AE. Reconstruction + beta*KL. Optional ablation variant. |
| **ViT-MAE** | `vit_mae.yaml` | Vision Transformer Masked Autoencoder (SSAST-style). Patch tokenization + Transformer encoder, token masking during training, partitioned reconstruction at inference. |

The best-performing backbone per data product is selected empirically via injection-recovery testing. No classifier is trained — all backbones are scored purely by reconstruction error.

## Pipeline Phases

1. **Unsupervised training** — the autoencoder learns the noise/RFI distribution from real telescope data
2. **Injection-recovery test** — synthetic signals (narrowband, wideband, broadband transient) are injected at varying SNR to characterize detection sensitivity
3. **Search** — the pipeline is applied to Exotica catalog observations; high-reconstruction-error frames are flagged as candidates and later cross-referenced with turboSETI output

## GBT Data Products

HDF5 filterbank files (`.h5`) at three resolution levels, each targeting a different signal class:

| Suffix | Freq. resolution | Time resolution | Target signal class |
|--------|-----------------|-----------------|---------------------|
| `0000.fil` | ~3 Hz/chan | ~18 s/bin | Narrowband drifting |
| `0001.fil` | ~364 kHz/chan | ~349 us/bin | Broadband transients |
| `0002.fil` | ~2.86 kHz/chan | ~1 s/bin | Wideband / modulated |

Observations follow an ABACAD cadence pattern (3 ON-source + 3 OFF-source), each ~308 s in duration.

## Installation

### With conda (recommended)

```bash
conda env create -f environment.yml
conda activate bl-exotica-ad
pip install -e .
```

On a CUDA-equipped machine, install the appropriate PyTorch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### With pip only

```bash
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, PyTorch >= 2.2, PyTorch Lightning >= 2.2

Key dependencies: `setigen` (synthetic signal generation), `blimpy` (filterbank I/O), `h5py`, `astropy`, `scikit-learn`, `pytorch-msssim`.

## Usage

### Training

```bash
# Default training (synthetic data, CNN autoencoder)
python scripts/train.py --config configs/training/default.yaml

# Override data and model configs
python scripts/train.py --config configs/training/default.yaml \
                        --data   configs/data/gbt_fine.yaml \
                        --model  configs/model/convae_mae.yaml
```

Outputs (checkpoints, logs, reconstruction snapshots) are saved to `outputs/<run_id>/`.

### Inference

Run the trained model on real cadences to score each frequency window:

```bash
python scripts/inference.py \
    --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
    --cadence_list data/processed/inference_cadences.txt \
    --out_dir outputs/inference/run_name \
    --num_workers 32
```

Each cadence gets its own output folder with per-candidate plots (original | reconstruction | error map).

### Injection-Recovery

Evaluate detection sensitivity by injecting synthetic signals into clean cadences:

```bash
python scripts/inject_recover.py \
    --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
    --cadence_list data/processed/inject_recovery_cadences.txt \
    --inference_csv outputs/inference/<run>/inference_scores.csv \
    --out_dir outputs/inject_recovery/run_name
```

### CLI Commands

After `pip install -e .`, the following commands are available system-wide:

| Command | Description |
|---------|-------------|
| `bl-train` | Training entry point |
| `bl-inference` | Run inference on real cadences |
| `bl-inject` | Injection-recovery sensitivity test |
| `bl-preprocess` | Build preprocessed snippet cache |
| `bl-manifest` | Build cadence manifest from raw data |

Each accepts the same arguments as the corresponding `python scripts/...` invocation. For example:

```bash
bl-train --config configs/training/default.yaml
bl-inference --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
             --cadence_list data/processed/inference_cadences.txt \
             --out_dir outputs/inference/run_name
```

## Repository Layout

```
BL-Exotica-AD/
├── configs/
│   ├── data/                 # per-product data configs
│   │   ├── gbt_fine.yaml         # 0000.fil (narrowband, ~3 Hz)
│   │   ├── gbt_moderate.yaml     # 0002.fil (wideband, ~2.86 kHz)
│   │   ├── gbt_high_time.yaml    # 0001.fil (broadband, ~364 kHz)
│   │   ├── synthetic.yaml        # setigen-generated training data
│   │   └── synthetic_fine.yaml
│   ├── model/                # architecture hyperparameters
│   │   ├── convae.yaml           # CNN AE / VAE
│   │   ├── convae_mae.yaml       # CNN Masked Autoencoder
│   │   └── vit_mae.yaml          # Vision Transformer MAE
│   ├── training/             # optimizer, scheduler, hardware
│   └── search/               # sliding-window, threshold settings
├── data/
│   ├── raw/                  # .h5 file lists / symlinks
│   ├── synthetic/            # setigen-generated data
│   └── processed/            # cached normalized snippets (gitignored)
├── src/                      # installable library
│   ├── data/
│   │   ├── loader.py             # blimpy Waterfall reader
│   │   ├── synthetic.py          # setigen wrappers for signal injection
│   │   ├── preprocessing.py      # bandpass correction + log1p + median/MAD
│   │   └── torch_dataset.py      # SpectrogramDataset, sliding-window snippets
│   ├── models/
│   │   ├── autoencoder.py        # AE / MAE / VAE + build_autoencoder()
│   │   ├── encoder.py            # CNN encoder blocks
│   │   ├── decoder.py            # transposed CNN decoder blocks
│   │   ├── losses.py             # MSE, SSIM, perceptual loss variants
│   │   └── vit_mae.py            # ViT-MAE (Transformer backbone)
│   ├── training/
│   │   ├── trainer.py            # LightningModule + Trainer setup
│   │   └── callbacks.py          # LR scheduling, early stopping, snapshots
│   ├── search/
│   │   ├── scorer.py             # sliding-window reconstruction-error scoring
│   │   └── candidates.py         # peak detection, deduplication
│   └── utils/
│       ├── config.py             # YAML loading + schema validation
│       ├── logging.py            # structured logging
│       └── visualization.py      # spectrogram plots, error maps
├── scripts/                  # CLI entry points
│   ├── train.py
│   ├── inference.py
│   ├── inject_recover.py
│   ├── search.py
│   ├── preprocess_cache.py
│   └── build_cadence_manifest.py
├── tests/
├── notebooks/                # exploration only
├── outputs/                  # checkpoints, logs, results (gitignored)
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## Design Choices

- **Framework:** PyTorch + PyTorch Lightning. Models are plain `nn.Module`s; a thin `LightningModule` wrapper handles optimization, checkpointing, and multi-GPU training. Hardware selection (CPU / 1-GPU / 2-GPU, mixed precision) is config-driven with zero code branching.
- **Preprocessing:** two-stage, applied online per snippet. Stage 1: bandpass correction (polynomial fit to per-channel temporal median, with iterative sigma-clipping to exclude RFI). Stage 2: `log1p` compression + robust median/MAD standardization.
- **Reproducibility:** seed applied via Lightning `seed_everything`; every run directory is tagged with a timestamp and git hash.

## References

1. Ma et al. 2023 — *"A Deep-learning Search for Technosignatures from 820 Nearby Stars"* — [arXiv:2301.12670](https://arxiv.org/abs/2301.12670)
2. Lacki et al. 2020 — *"One of Everything: The Breakthrough Listen Exotica Catalog"* — [arXiv:2006.11304](https://arxiv.org/abs/2006.11304)
3. Gajjar et al. 2022 — *"Searching for Broadband Pulsed Beacons from 1883 Stars Using Neural Networks"* — [ApJ 932 81](https://iopscience.iop.org/article/10.3847/1538-4357/ac6dd5/meta)
4. He et al. 2022 — *"Masked Autoencoders Are Scalable Vision Learners"* — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
5. Gong et al. 2022 — *"SSAST: Self-Supervised Audio Spectrogram Transformer"* — [arXiv:2110.09784](https://arxiv.org/abs/2110.09784)
