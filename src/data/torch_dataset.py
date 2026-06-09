import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Union

from .loader import load_observation
from .preprocessing import bandpass_correct, core_transform


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset serving (1, ntime, fchans) tensors from pre-loaded
    spectrogram observations.

    Each __getitem__ extracts a frequency window of width `fchans` from one
    observation, applies bandpass correction and log1p/MAD normalisation, and
    returns a float32 NCHW tensor ready for the autoencoder.

    Observations are stored in memory as (ntime, nchans_ds) arrays (already
    downsampled). The snippet index is built once at __init__ by sliding a
    window of width `fchans` along the frequency axis with the given stride.
    """

    def __init__(
        self,
        observations: List[np.ndarray],
        fchans: int,
        stride: int,
        cfg_preproc: Dict[str, Any],
    ):
        """
        Args:
            observations: list of (ntime, nchans_ds) float32 arrays.
            fchans: snippet width in downsampled channels.
            stride: step between consecutive snippet start positions (channels).
            cfg_preproc: preprocessing config with keys:
                bandpass_method ('polynomial' | 'median'),
                poly_degree (int),
                mad_epsilon (float).
        """
        self.observations = observations
        self.fchans = fchans
        self.cfg_preproc = cfg_preproc

        self.snippet_index: List[Tuple[int, int]] = []
        for obs_idx, obs in enumerate(observations):
            nchans = obs.shape[1]
            f_start = 0
            while f_start + fchans <= nchans:
                self.snippet_index.append((obs_idx, f_start))
                f_start += stride

    def __len__(self) -> int:
        return len(self.snippet_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        obs_idx, f_start = self.snippet_index[idx]
        # .copy() so preprocessing operates on its own buffer, not a view
        frame = self.observations[obs_idx][:, f_start : f_start + self.fchans].copy()

        frame = bandpass_correct(
            frame,
            method=self.cfg_preproc.get("bandpass_method", "polynomial"),
            poly_degree=self.cfg_preproc.get("poly_degree", 3),
        )
        frame = core_transform(frame, self.cfg_preproc.get("mad_epsilon", 1e-6))

        return torch.from_numpy(frame).float().unsqueeze(0)  # (1, ntime, fchans)


def build_datasets(
    file_list: List[Union[str, Path]],
    cfg_data: Dict[str, Any],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[SpectrogramDataset, SpectrogramDataset]:
    """
    Load observations from disk and build train/val SpectrogramDatasets.

    Observations are split at the file level (not snippet level) to ensure
    no noise realisations appear in both train and val sets.

    Args:
        file_list: paths to .h5 / .fil observation files.
        cfg_data: merged data config dict containing 'frame' and
            'preprocessing' sub-dicts (see configs/data/gbt_fine.yaml).
        val_fraction: fraction of observations reserved for validation.
        seed: RNG seed for the observation shuffle.

    Returns:
        (train_dataset, val_dataset) pair of SpectrogramDatasets.
    """
    rng = np.random.default_rng(seed)
    paths = [Path(p) for p in file_list]
    rng.shuffle(paths)

    n_val = max(1, int(len(paths) * val_fraction))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]

    frame_cfg = cfg_data["frame"]
    downsample_factor = frame_cfg["downsample_factor"]
    fchans = frame_cfg["fchans"]
    stride_train = frame_cfg["stride_train"]
    stride_infer = frame_cfg["stride_infer"]
    cfg_preproc = cfg_data["preprocessing"]

    train_obs = [load_observation(p, downsample_factor) for p in train_paths]
    val_obs = [load_observation(p, downsample_factor) for p in val_paths]

    train_ds = SpectrogramDataset(train_obs, fchans, stride_train, cfg_preproc)
    val_ds = SpectrogramDataset(val_obs, fchans, stride_infer, cfg_preproc)

    return train_ds, val_ds
