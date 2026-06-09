import numpy as np
from numpy.polynomial.polynomial import Polynomial


def bandpass_correct(
    frame: np.ndarray,
    method: str = "polynomial",
    poly_degree: int = 3,
) -> np.ndarray:
    """
    Remove instrumental bandpass from a (ntime, fchans) power frame.

    Estimates the smooth spectral baseline B(f) from the per-channel temporal
    median, then divides the frame by it. Two methods are supported:

    - 'polynomial': fits a Chebyshev-stable polynomial to M(f) after 3
      iterations of 3-sigma clipping to exclude RFI-bright channels. More
      robust for products with few time bins (e.g. 0000.fil, ntime=16) where
      a per-channel temporal median is noisy.
    - 'median': divides directly by M(f). Faster and sufficient for products
      with many time bins (e.g. 0001.fil, ntime=884736).

    Args:
        frame: (ntime, fchans) positive float32, raw power after downsampling.
        method: 'polynomial' or 'median'.
        poly_degree: degree of the polynomial fit (only used when method='polynomial').

    Returns:
        frame / B(f) as float32, same shape. Values are ratios (unitless).
    """
    f64 = frame.astype(np.float64)
    nchans = f64.shape[1]

    M = np.median(f64, axis=0)  # (fchans,)

    if method == "polynomial":
        x = np.linspace(-1.0, 1.0, nchans)  # normalised domain for stability

        mask = np.ones(nchans, dtype=bool)
        p = Polynomial.fit(x, M, deg=poly_degree)
        for _ in range(3):
            residuals = M - p(x)
            sigma = np.std(residuals[mask])
            if sigma == 0.0:
                break
            mask = np.abs(residuals) <= 3.0 * sigma
            if mask.sum() < poly_degree + 1:
                break
            p = Polynomial.fit(x[mask], M[mask], deg=poly_degree)

        B = p(x)
    else:
        B = M

    # Floor at a small fraction of the peak to avoid division by zero or
    # near-zero values at band edges where the fit may dip.
    B = np.maximum(B, B.max() * 1e-4 + 1e-10)

    return (f64 / B[np.newaxis, :]).astype(np.float32)


def core_transform(frame: np.ndarray, mad_epsilon: float = 1e-6) -> np.ndarray:
    """
    log1p compression followed by robust median/MAD standardisation.

    Applied per-snippet after bandpass_correct. The median/MAD standardisation
    makes the reconstruction error interpretable as "sigma unexplained by the
    noise model", and is robust to residual RFI spikes that would dominate a
    min-max or mean/std normalisation.

    Args:
        frame: (ntime, fchans) float32, output of bandpass_correct.
        mad_epsilon: additive floor on the MAD to prevent division by zero in
            spectrally quiet snippets.

    Returns:
        Standardised float32 array, same shape.
    """
    f = np.log1p(np.clip(frame, 0.0, None).astype(np.float64))

    median = np.median(f)
    mad = np.median(np.abs(f - median)) + mad_epsilon

    return ((f - median) / mad).astype(np.float32)
