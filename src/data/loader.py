import numpy as np
from pathlib import Path
from typing import Union

import blimpy


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
