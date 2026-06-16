import numpy as np
import torch
import h5py
try:
    import hdf5plugin  # noqa: F401 — registers bitshuffle/LZ4 filters for BL HDF5 files
except ImportError:
    pass
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Union

from .preprocessing import bandpass_correct, core_transform


def _read_nchans(path: Path, downsample_factor: int = 1) -> int:
    """Read channel count from file header without loading any data."""
    if path.suffix == '.h5':
        with h5py.File(str(path), 'r') as f:
            return int(f['data'].shape[2]) // downsample_factor
    else:
        import blimpy
        wf = blimpy.Waterfall(str(path), load_data=False)
        return int(wf.header['nchans']) // downsample_factor


def _load_channel_window(
    path: Path,
    f_start: int,
    fchans: int,
    downsample_factor: int = 1,
) -> np.ndarray:
    """
    Load a (ntime, fchans) window from a filterbank file using partial I/O.

    For .h5 files uses h5py hyperslab selection — reads only the requested
    channel range from disk (O(ntime * fchans), not O(ntime * nchans_total)).
    For .fil files falls back to loading the full file.

    Args:
        path: path to the .h5 or .fil file.
        f_start: start channel index in downsampled space.
        fchans: number of channels to read (downsampled).
        downsample_factor: average-pool along frequency by this factor.

    Returns:
        float32 array of shape (ntime, fchans).
    """
    raw_start = f_start * downsample_factor
    raw_fchans = fchans * downsample_factor

    if path.suffix == '.h5':
        with h5py.File(str(path), 'r') as f:
            data = f['data'][:, 0, raw_start : raw_start + raw_fchans]
        data = np.asarray(data, dtype=np.float32)
    else:
        import blimpy
        wf = blimpy.Waterfall(str(path), load_data=True)
        raw = wf.data
        if raw.ndim == 3:
            raw = raw[:, 0, :]
        data = raw[:, raw_start : raw_start + raw_fchans].astype(np.float32)

    if downsample_factor > 1:
        ntime = data.shape[0]
        data = data.reshape(ntime, fchans, downsample_factor).mean(axis=-1)

    return data


class SpectrogramDataset(Dataset):
    """
    Lazy PyTorch Dataset serving (1, tchans, fchans) tensors from cadence files.

    No observation data is loaded at init — only file headers are read to build
    the snippet index. Each __getitem__ loads only the requested `fchans`-wide
    frequency window from disk via HDF5 partial I/O, normalises each observation
    independently (bandpass_correct + core_transform), concatenates along the
    time axis, truncates to exactly `tchans` rows, and returns a float32 NCHW
    tensor.

    The truncation to `tchans` is the contract: if a cadence file set produces
    more rows than `tchans` (e.g. a 7-obs cadence or obs with ntime>16), the
    excess rows are silently dropped. Callers must ensure cadences have at least
    `tchans` total time bins; cadences that fall short are skipped during init.

    RAM cost: O(n_cadences) for the path index — effectively zero regardless of
    dataset size. I/O cost per snippet: n_obs × ntime × fchans × 4 bytes
    (≈ 384 KB for 0000.fil with 6 obs × 16 bins × 1024 chans).

    For single-observation products (e.g. 0001.fil) pass cadences where each
    inner list contains exactly one path — the logic is identical.
    """

    def __init__(
        self,
        cadence_paths: List[List[Path]],
        tchans: int,
        fchans: int,
        stride: int,
        cfg_preproc: Dict[str, Any],
        downsample_factor: int = 1,
    ):
        """
        Args:
            cadence_paths: list of cadences; each cadence is a list of Paths to
                observation files sorted chronologically.
            tchans: expected number of time bins in the output (after
                concatenating all obs in a cadence). Cadences with fewer total
                time bins are skipped; those with more are truncated.
            fchans: snippet width in (downsampled) channels.
            stride: step between consecutive snippet start positions.
            cfg_preproc: preprocessing config (bandpass_method, poly_degree,
                mad_epsilon).
            downsample_factor: frequency-axis average-pooling factor applied
                during loading.
        """
        self.cadence_paths = cadence_paths
        self.tchans = tchans
        self.fchans = fchans
        self.cfg_preproc = cfg_preproc
        self.downsample_factor = downsample_factor

        # Build snippet index by reading nchans from the header of the first
        # file in each cadence (all files in a cadence share the same nchans).
        self.snippet_index: List[Tuple[int, int]] = []
        for cad_idx, cadence in enumerate(cadence_paths):
            nchans = _read_nchans(cadence[0], downsample_factor)
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
        for path in self.cadence_paths[cad_idx]:
            frame = _load_channel_window(path, f_start, self.fchans, self.downsample_factor)
            frame = bandpass_correct(frame, method=method, poly_degree=poly_degree)
            frame = core_transform(frame, mad_epsilon)
            sub_frames.append(frame)

        result = np.concatenate(sub_frames, axis=0)  # (total_time, fchans)
        result = result[: self.tchans, :]             # enforce fixed height
        return torch.from_numpy(result).float().unsqueeze(0)  # (1, tchans, fchans)


class CachedDataset(Dataset):
    """
    In-RAM dataset backed by a preprocessed NPZ cache (see scripts/preprocess_cache.py).

    Loads the entire split into a numpy array at init. Raw snippets have shape
    (N, n_obs, tchans_per_obs, fchans). __getitem__ applies bandpass_correct +
    core_transform per observation (fast — no I/O), concatenates along the time
    axis, and returns a (1, tchans, fchans) float32 tensor.

    Storing RAW data means preprocessing hyperparameters can be changed without
    re-extracting the cache (same approach as RST background_extractor.py).

    RAM cost: N × n_obs × tchans_per_obs × fchans × 4 bytes
    (≈78 GB for 200k × 6 × 16 × 1024 float32).
    """

    def __init__(self, npz_path: Path, split: str, cfg_preproc: Dict[str, Any]):
        print(f"Loading {split} cache from {npz_path} into RAM...")
        archive = np.load(str(npz_path))
        self.data = archive[split]          # (N, n_obs, tchans_per_obs, fchans)
        self.cfg_preproc = cfg_preproc
        print(f"  {split}: {self.data.shape[0]} snippets  "
              f"shape={self.data.shape[1:]}  "
              f"{self.data.nbytes / 1e9:.1f} GB in RAM")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        method = self.cfg_preproc.get("bandpass_method", "polynomial")
        poly_degree = self.cfg_preproc.get("poly_degree", 3)
        mad_epsilon = self.cfg_preproc.get("mad_epsilon", 1e-6)

        raw = self.data[idx]   # (n_obs, tchans_per_obs, fchans)
        sub_frames = []
        for obs in raw:
            frame = bandpass_correct(obs, method=method, poly_degree=poly_degree)
            frame = core_transform(frame, mad_epsilon)
            sub_frames.append(frame)

        result = np.concatenate(sub_frames, axis=0)   # (tchans, fchans)
        return torch.from_numpy(result).float().unsqueeze(0)


def build_datasets(
    cadence_list: List[List[Union[str, Path]]],
    cfg_data: Dict[str, Any],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[SpectrogramDataset, SpectrogramDataset]:
    """
    Build train/val SpectrogramDatasets from a cadence list (lazy, no data loaded).

    Cadences are split at the cadence level for statistical independence between
    train and val sets.

    Args:
        cadence_list: list of cadences; each cadence is a list of paths to
            observation files (6 for 0000.fil, 1 for 0001.fil).
        cfg_data: merged data config dict containing 'frame' and
            'preprocessing' sub-dicts.
        val_fraction: fraction of cadences reserved for validation.
        seed: RNG seed for the cadence shuffle.

    Returns:
        (train_dataset, val_dataset) pair of SpectrogramDatasets (lazy) or
        CachedDatasets (if cfg_data["dataset"]["cache_file"] exists on disk).
    """
    cache_file = cfg_data.get("dataset", {}).get("cache_file")
    if cache_file and Path(cache_file).exists():
        cfg_preproc = cfg_data["preprocessing"]
        cache_path = Path(cache_file)
        train_ds = CachedDataset(cache_path, "train", cfg_preproc)
        val_ds   = CachedDataset(cache_path, "val",   cfg_preproc)
        return train_ds, val_ds

    rng = np.random.default_rng(seed)
    cadences = [[Path(p) for p in group] for group in cadence_list]
    rng.shuffle(cadences)

    n_val = max(1, int(len(cadences) * val_fraction))
    n_val = min(n_val, len(cadences) - 1)  # guarantee at least 1 cadence in train
    if n_val == 0:
        # single cadence: val and train share the same data
        val_paths = cadences
        train_paths = cadences
    else:
        val_paths = cadences[:n_val]
        train_paths = cadences[n_val:]

    frame_cfg = cfg_data["frame"]
    tchans = frame_cfg["tchans"]
    fchans = frame_cfg["fchans"]
    stride_train = frame_cfg["stride_train"]
    stride_infer = frame_cfg["stride_infer"]
    downsample_factor = frame_cfg.get("downsample_factor", 1)
    cfg_preproc = cfg_data["preprocessing"]

    train_ds = SpectrogramDataset(train_paths, tchans, fchans, stride_train, cfg_preproc, downsample_factor)
    val_ds = SpectrogramDataset(val_paths, tchans, fchans, stride_train, cfg_preproc, downsample_factor)

    return train_ds, val_ds
