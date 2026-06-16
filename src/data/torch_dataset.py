import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Union

from .loader import load_cadence
from .preprocessing import bandpass_correct, core_transform


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset serving (1, ntime, fchans) tensors from pre-loaded cadences.

    A cadence is a list of per-observation (ntime_obs, nchans_ds) arrays. Each
    __getitem__ slides a frequency window of width `fchans` across the cadence,
    normalises each observation independently (bandpass + log1p/MAD), concatenates
    them along the time axis, and returns a float32 NCHW tensor.

    For single-observation products (e.g. 0001.fil) pass cadences where each
    inner list contains exactly one array — the logic is identical.

    Cadences are stored in memory as List[List[np.ndarray]] (already downsampled).
    The snippet index is built once at __init__ by sliding a window of width
    `fchans` along the frequency axis of each cadence with the given stride.
    """

    def __init__(
        self,
        cadences: List[List[np.ndarray]],
        fchans: int,
        stride: int,
        cfg_preproc: Dict[str, Any],
    ):
        """
        Args:
            cadences: list of cadences; each cadence is a list of (ntime_obs, nchans_ds)
                float32 arrays sorted chronologically.
            fchans: snippet width in downsampled channels.
            stride: step between consecutive snippet start positions (channels).
            cfg_preproc: preprocessing config with keys:
                bandpass_method ('polynomial' | 'median'),
                poly_degree (int),
                mad_epsilon (float).
        """
        self.cadences = cadences
        self.fchans = fchans
        self.cfg_preproc = cfg_preproc

        self.snippet_index: List[Tuple[int, int]] = []
        for cad_idx, cadence in enumerate(cadences):
            nchans = cadence[0].shape[1]
            f_start = 0
            while f_start + fchans <= nchans:
                self.snippet_index.append((cad_idx, f_start))
                f_start += stride

    def __len__(self) -> int:
        return len(self.snippet_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        cad_idx, f_start = self.snippet_index[idx]
        method = self.cfg_preproc.get("bandpass_method", "polynomial")
        poly_degree = self.cfg_preproc.get("poly_degree", 3)
        mad_epsilon = self.cfg_preproc.get("mad_epsilon", 1e-6)

        sub_frames = []
        for obs_arr in self.cadences[cad_idx]:
            # .copy() so preprocessing operates on its own buffer, not a view
            frame = obs_arr[:, f_start : f_start + self.fchans].copy()
            frame = bandpass_correct(frame, method=method, poly_degree=poly_degree)
            frame = core_transform(frame, mad_epsilon)
            sub_frames.append(frame)

        result = np.concatenate(sub_frames, axis=0)  # (total_time, fchans)
        return torch.from_numpy(result).float().unsqueeze(0)  # (1, total_time, fchans)


def build_datasets(
    cadence_list: List[List[Union[str, Path]]],
    cfg_data: Dict[str, Any],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[SpectrogramDataset, SpectrogramDataset]:
    """
    Load cadences from disk and build train/val SpectrogramDatasets.

    Cadences are split at the cadence level (not snippet level) to ensure
    no noise realisations appear in both train and val sets.

    Args:
        cadence_list: list of cadences; each cadence is a list of paths to
            observation files (typically 6 for 0000.fil, 1 for 0001.fil).
        cfg_data: merged data config dict containing 'frame' and
            'preprocessing' sub-dicts (see configs/data/gbt_fine.yaml).
        val_fraction: fraction of cadences reserved for validation.
        seed: RNG seed for the cadence shuffle.

    Returns:
        (train_dataset, val_dataset) pair of SpectrogramDatasets.
    """
    rng = np.random.default_rng(seed)
    cadences = [[Path(p) for p in group] for group in cadence_list]
    rng.shuffle(cadences)

    n_val = max(1, int(len(cadences) * val_fraction))
    val_paths = cadences[:n_val]
    train_paths = cadences[n_val:]

    frame_cfg = cfg_data["frame"]
    downsample_factor = frame_cfg["downsample_factor"]
    fchans = frame_cfg["fchans"]
    stride_train = frame_cfg["stride_train"]
    stride_infer = frame_cfg["stride_infer"]
    cfg_preproc = cfg_data["preprocessing"]

    train_obs = [load_cadence(group, downsample_factor) for group in train_paths]
    val_obs = [load_cadence(group, downsample_factor) for group in val_paths]

    train_ds = SpectrogramDataset(train_obs, fchans, stride_train, cfg_preproc)
    val_ds = SpectrogramDataset(val_obs, fchans, stride_infer, cfg_preproc)

    return train_ds, val_ds
