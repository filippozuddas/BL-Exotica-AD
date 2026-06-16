import numpy as np
from pathlib import Path
from typing import List, Union

import blimpy


def load_cadence(
    file_paths: List[Union[str, Path]],
    downsample_factor: int = 1,
) -> List[np.ndarray]:
    """
    Load a cadence (group of observations) and return a list of (ntime, nchans) arrays.

    Files are sorted by MJD-seconds extracted from the filename (field index 3
    when splitting by '_', e.g. 'blc01_guppi_59378_47976_...' → 47976) so that
    out-of-order entries in the manifest are corrected automatically.

    Args:
        file_paths: paths to the observation files belonging to one cadence.
        downsample_factor: passed through to load_observation().

    Returns:
        List of float32 arrays, one per file, each shaped (ntime, nchans // downsample_factor),
        sorted chronologically by MJD-seconds.
    """
    sorted_paths = sorted(
        [Path(p) for p in file_paths],
        key=lambda p: int(p.name.split("_")[3]),
    )
    return [load_observation(p, downsample_factor) for p in sorted_paths]


def load_observation(
    file_path: Union[str, Path],
    downsample_factor: int = 1,
) -> np.ndarray:
    """
    Load a filterbank / HDF5 observation file and return a (ntime, nchans) array.

    Handles both .fil and .h5 formats via blimpy. The polarisation dimension is
    always squeezed: blimpy typically returns (ntime, nifs, nchans); we keep
    only the first IF.

    Frequency-axis downsampling is applied by average-pooling with the given
    factor before returning. This reduces memory by factor^2 relative to holding
    full-resolution data and lets the Dataset work entirely in the downsampled
    channel space (stride and fchans parameters are in downsampled units).

    Args:
        file_path: path to a .h5 or .fil filterbank file.
        downsample_factor: average-pool along the frequency axis by this factor.
            Must evenly divide nchans; excess channels are truncated.

    Returns:
        float32 array of shape (ntime, nchans // downsample_factor).
    """
    wf = blimpy.Waterfall(str(file_path), load_data=True)
    data = wf.data  # (ntime,) or (ntime, nifs, nchans) depending on blimpy version

    if data.ndim == 3:
        data = data[:, 0, :]  # (ntime, nchans)
    elif data.ndim == 2:
        pass  # already (ntime, nchans)
    else:
        raise ValueError(f"Unexpected data shape from blimpy: {data.shape}")

    data = data.astype(np.float32)

    if downsample_factor > 1:
        ntime, nchans = data.shape
        nchans_out = nchans // downsample_factor
        data = data[:, : nchans_out * downsample_factor]
        data = data.reshape(ntime, nchans_out, downsample_factor).mean(axis=-1)

    return data
