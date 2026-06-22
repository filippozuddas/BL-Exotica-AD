import numpy as np
import torch
import h5py
try:
    import hdf5plugin  # noqa: F401 — registers bitshuffle/LZ4 filters for BL HDF5 files
except ImportError:
    pass
from pathlib import Path
from torch.utils.data import Dataset, Subset
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


def _load_full_obs(path: Path, downsample_factor: int = 1) -> np.ndarray:
    """Load a full observation into a (ntime, nchans_downsampled) float32 array.

    Called once per file at Dataset init to populate the in-RAM observation
    cache, eliminating all per-snippet file I/O during training.
    """
    if path.suffix == '.h5':
        with h5py.File(str(path), 'r') as f:
            data = np.asarray(f['data'][:, 0, :], dtype=np.float32)
    else:
        import blimpy
        wf = blimpy.Waterfall(str(path), load_data=True)
        raw = wf.data
        if raw.ndim == 3:
            raw = raw[:, 0, :]
        data = raw.astype(np.float32)

    if downsample_factor > 1:
        ntime, nchans = data.shape
        nchans_ds = nchans // downsample_factor
        data = data[:, : nchans_ds * downsample_factor].reshape(
            ntime, nchans_ds, downsample_factor
        ).mean(axis=-1)

    return data


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
    PyTorch Dataset serving (1, tchans, fchans) tensors from cadence files.

    All observation files are loaded fully into RAM at init (one float32 array
    per file). __getitem__ slices the requested frequency window directly from
    RAM — zero disk I/O during training, keeping GPU utilisation high.

    RAM cost: n_unique_obs × ntime × nchans × 4 bytes
    (≈ 4.3 GB per 0000.fil observation at 16 bins × 67M chans).

    The truncation to `tchans` is the contract: if a cadence file set produces
    more rows than `tchans` (e.g. a 7-obs cadence or obs with ntime>16), the
    excess rows are silently dropped. Callers must ensure cadences have at least
    `tchans` total time bins; cadences that fall short are skipped during init.

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

        # Load all observation files into RAM once — eliminates per-snippet HDF5
        # I/O that would otherwise starve the GPU (each __getitem__ would open
        # and close n_obs files, keeping GPU utilisation at ~0%).
        all_paths = sorted(
            {path for cadence in cadence_paths for path in cadence},
            key=str,
        )
        print(f"Loading {len(all_paths)} observation files into RAM...")
        self._obs_cache: Dict[Path, np.ndarray] = {}
        for path in all_paths:
            self._obs_cache[path] = _load_full_obs(path, downsample_factor)
            gb = self._obs_cache[path].nbytes / 1e9
            print(f"  {path.name}  {self._obs_cache[path].shape}  {gb:.2f} GB")

        # Validate cadences: discard any where an observation has unexpected ntime.
        # A malformed obs would corrupt the cadence structure (ON/OFF ordering),
        # so the whole cadence is dropped rather than silently truncated.
        valid_cadences = []
        for cadence in cadence_paths:
            n_obs = len(cadence)
            if self.tchans % n_obs != 0:
                names = [p.name for p in cadence]
                print(f"  WARNING: skipping cadence — tchans={self.tchans} not divisible "
                      f"by n_obs={n_obs}: {names}")
                continue
            expected_ntime = self.tchans // n_obs
            bad = [p for p in cadence if self._obs_cache[p].shape[0] != expected_ntime]
            if bad:
                print(f"  WARNING: skipping cadence — unexpected ntime "
                      f"(expected {expected_ntime}): {[p.name for p in bad]}")
                continue
            valid_cadences.append(cadence)
        n_dropped = len(cadence_paths) - len(valid_cadences)
        if n_dropped:
            print(f"  Dropped {n_dropped}/{len(cadence_paths)} cadences with bad obs shapes.")
        self.cadence_paths = valid_cadences

        # Build snippet index using nchans from the cached arrays.
        self.snippet_index: List[Tuple[int, int]] = []
        for cad_idx, cadence in enumerate(self.cadence_paths):
            nchans = self._obs_cache[cadence[0]].shape[1]
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

        sub_frames = [
            self._obs_cache[path][:, f_start : f_start + self.fchans]
            for path in self.cadence_paths[cad_idx]
        ]
        result = np.concatenate(sub_frames, axis=0)          # (total_time, fchans)
        result = result[: self.tchans, :]                    # enforce fixed height
        result = bandpass_correct(result, method=method, poly_degree=poly_degree)
        result = core_transform(result, mad_epsilon)
        return torch.from_numpy(result).float().unsqueeze(0)  # (1, tchans, fchans)


class CachedDataset(Dataset):
    """
    Memory-mapped dataset backed by per-split .npy files (see scripts/preprocess_cache.py).

    Each split is a single .npy file inside a cache directory, loaded with
    ``mmap_mode='r'`` so the OS pages data in on demand and forked DataLoader
    workers share the same physical pages (no RAM duplication).

    Raw snippets have shape (N, n_obs, tchans_per_obs, fchans). __getitem__
    applies bandpass_correct + core_transform per observation, concatenates
    along the time axis, and returns a (1, tchans, fchans) float32 tensor.

    Storing RAW data means preprocessing hyperparameters can be changed without
    re-extracting the cache.
    """

    def __init__(self, cache_dir: Path, split: str, cfg_preproc: Dict[str, Any]):
        npy_path = cache_dir / f"{split}.npy"
        print(f"Memory-mapping {split} cache from {npy_path}...")
        self.data = np.load(str(npy_path), mmap_mode="r")  # (N, n_obs, tchans_per_obs, fchans)
        self.cfg_preproc = cfg_preproc
        print(f"  {split}: {self.data.shape[0]} snippets  "
              f"shape={self.data.shape[1:]}  "
              f"~{self.data.nbytes / 1e9:.1f} GB on disk (mmap, not in RAM)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        method = self.cfg_preproc.get("bandpass_method", "polynomial")
        poly_degree = self.cfg_preproc.get("poly_degree", 3)
        mad_epsilon = self.cfg_preproc.get("mad_epsilon", 1e-6)

        raw = self.data[idx]                          # (n_obs, tchans_per_obs, fchans)
        result = np.concatenate(raw, axis=0)          # (tchans, fchans)
        result = bandpass_correct(result, method=method, poly_degree=poly_degree)
        result = core_transform(result, mad_epsilon)
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
        CachedDatasets (if cfg_data["dataset"]["cache_dir"] exists on disk).
    """
    cache_dir = cfg_data.get("dataset", {}).get("cache_dir")
    if cache_dir and Path(cache_dir).is_dir():
        cfg_preproc = cfg_data["preprocessing"]
        cache_path = Path(cache_dir)
        train_ds = CachedDataset(cache_path, "train", cfg_preproc)
        val_ds   = CachedDataset(cache_path, "val",   cfg_preproc)
        return train_ds, val_ds

    rng = np.random.default_rng(seed)
    cadences = [[Path(p) for p in group] for group in cadence_list]
    rng.shuffle(cadences)

    frame_cfg = cfg_data["frame"]
    tchans = frame_cfg["tchans"]
    fchans = frame_cfg["fchans"]
    stride_train = frame_cfg["stride_train"]
    downsample_factor = frame_cfg.get("downsample_factor", 1)
    cfg_preproc = cfg_data["preprocessing"]

    n_val = max(1, int(len(cadences) * val_fraction))
    n_val = min(n_val, len(cadences) - 1)  # guarantee at least 1 cadence in train

    if n_val == 0:
        # Single cadence: split at the snippet level so train/val are disjoint.
        full_ds = SpectrogramDataset(cadences, tchans, fchans, stride_train, cfg_preproc, downsample_factor)
        n_total = len(full_ds)
        n_val_snip = max(1, int(n_total * val_fraction))
        indices = rng.permutation(n_total).tolist()
        train_ds = Subset(full_ds, indices[n_val_snip:])
        val_ds   = Subset(full_ds, indices[:n_val_snip])
    else:
        val_paths   = cadences[:n_val]
        train_paths = cadences[n_val:]
        train_ds = SpectrogramDataset(train_paths, tchans, fchans, stride_train, cfg_preproc, downsample_factor)
        val_ds   = SpectrogramDataset(val_paths,   tchans, fchans, stride_train, cfg_preproc, downsample_factor)

    return train_ds, val_ds
