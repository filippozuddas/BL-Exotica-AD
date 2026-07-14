"""Spectrogram plots, reconstruction error maps, and candidate stamps."""

import numpy as np
from scipy.ndimage import zoom


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
