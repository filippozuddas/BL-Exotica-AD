"""Spectrogram plots, reconstruction error maps, and candidate stamps."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def add_obs_dividers(ax, n_rows: int, n_obs: int = 6, color: str = "white",
                      lw: float = 0.8, alpha: float = 0.7):
    """Draw horizontal lines marking the boundaries between the ``n_obs``
    stacked observations of a cadence in a (time, freq) waterfall plot.

    ``n_rows`` (e.g. ``frame.tchans``) is the concatenated ABACAD cadence —
    ``n_rows // n_obs`` time bins per observation (see ``configs/data/gbt_fine.yaml``:
    96 = 6 obs x 16 bins). No-op if ``n_rows`` doesn't divide evenly by
    ``n_obs`` (e.g. a single-observation plot).
    """
    if n_rows % n_obs != 0:
        return
    bins_per_obs = n_rows // n_obs
    for row in range(bins_per_obs, n_rows, bins_per_obs):
        ax.axhline(row - 0.5, color=color, lw=lw, alpha=alpha, ls="-")


def upsample_map_bilinear(amap: np.ndarray, target_shape) -> np.ndarray:
    """Bilinearly upsample a native (nh,nw) patch-grid map to ``target_shape`` pixels.

    Visualization only — never feed the result back into scoring. The model's
    top-k/mean pooling must always run on the native grid; interpolation here
    just makes the low-resolution grid legible against the full-res spectrogram.
    """
    zh = target_shape[0] / amap.shape[0]
    zw = target_shape[1] / amap.shape[1]
    return zoom(amap, (zh, zw), order=1)


def overlay_anomaly_map(ax, base_img: np.ndarray, amap: np.ndarray, cmap: str = "inferno",
                         alpha: float = 0.45, title: str = None, origin: str = "upper"):
    """Plot ``base_img`` in grayscale with the bilinearly-upsampled ``amap`` overlaid.

    ``amap`` is the native (nh,nw) patch-grid anomaly map (e.g. UDMA's fused
    map_cob); it is upsampled to ``base_img.shape`` purely for this overlay.
    ``origin`` must match the convention used for ``base_img`` elsewhere in the
    same figure, or the overlay will be flipped relative to its neighbors.
    """
    up = upsample_map_bilinear(amap, base_img.shape)
    vmin, vmax = np.percentile(base_img, [1, 99])
    ax.imshow(base_img, aspect="auto", origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    im = ax.imshow(up, aspect="auto", origin=origin, cmap=cmap, alpha=alpha)
    if title:
        ax.set_title(title)
    return im


def plot_candidate(original, reconstruction, score, sigma, method, cad_idx,
                    target, f_start, df, anomaly_map=None, n_obs=6):
    """Build (but don't save) the original|reconstruction|error figure for one
    candidate; caller decides whether to write it to PNG, a per-cadence PDF,
    or both.

    ``reconstruction``/error panels for pixel-decoder backbones; if
    ``reconstruction`` is None (UDMA, no pixel decoder), ``anomaly_map`` — its
    native (nh,nw) disagreement grid — is shown instead (see
    ``UDMA.anomaly_map`` / ``scripts/debug/udma_anomaly_maps.py``).

    Horizontal lines mark the ``n_obs`` observation boundaries (ABACAD
    cadence stacking) on every panel drawn at ``original``'s native time
    resolution — the two full-resolution panels (``original``,
    ``reconstruction``/residual) plus the bilinear anomaly-map overlay.
    """
    n_rows = original.shape[0]
    vmin, vmax = np.percentile(original, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(original, aspect="auto", origin="upper",
                          vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Original")
    axes[0].set_ylabel("Time bin")
    axes[0].set_xlabel("Freq channel")
    add_obs_dividers(axes[0], n_rows, n_obs)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    if reconstruction is not None:
        im1 = axes[1].imshow(reconstruction, aspect="auto", origin="upper",
                              vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1].set_title("Reconstruction")
        axes[1].set_xlabel("Freq channel")
        add_obs_dividers(axes[1], n_rows, n_obs)
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        error = np.abs(original - reconstruction)
        im2 = axes[2].imshow(error, aspect="auto", origin="upper", cmap="hot")
        axes[2].set_title("Residual |orig - recon|")
        axes[2].set_xlabel("Freq channel")
        add_obs_dividers(axes[2], n_rows, n_obs)
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        im1 = axes[1].imshow(anomaly_map, aspect="auto", origin="upper", cmap="viridis")
        axes[1].set_title("anomaly_map (UDMA, native (nh,nw) grid)")
        axes[1].set_xlabel("Freq patch col")
        axes[1].set_ylabel("Time patch row")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        overlay_anomaly_map(axes[2], original, anomaly_map,
                             title="anomaly_map (bilinear overlay)")
        add_obs_dividers(axes[2], n_rows, n_obs)

    f_center_mhz = f_start * df / 1e6
    score_line = f"{method} score={score:.4f}"
    if sigma is not None:
        score_line += f" ({sigma:.1f}s)"
    fig.suptitle(
        f"Candidate: cad={cad_idx} ({target})  f_start={f_start}  "
        f"f~{f_center_mhz:.4f} MHz\n"
        f"{score_line}",
        fontsize=11,
    )
    plt.tight_layout()
    return fig
