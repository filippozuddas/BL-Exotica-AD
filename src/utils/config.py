"""YAML config loading with nested merge support."""

import yaml
from pathlib import Path
from typing import Union


def load_config(training_yaml: Union[str, Path]) -> dict:
    """Load a training config and resolve the referenced data/model configs.

    The training YAML contains top-level ``data:`` and ``model:`` keys whose
    values are paths (relative to the repo root) pointing to the corresponding
    sub-configs. This function loads all three files and returns a single dict:

        {
            "data":     <contents of data YAML>,
            "model":    <contents of model YAML>,
            "training": {...},
            "hardware": {...},
            "output_dir": "...",
        }

    Args:
        training_yaml: path to a ``configs/training/*.yaml`` file.

    Returns:
        Merged config dict.
    """
    training_yaml = Path(training_yaml)
    with open(training_yaml) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to repo root (two levels up from configs/training/)
    root = training_yaml.parent.parent.parent

    data_path = root / cfg.pop("data")
    model_path = root / cfg.pop("model")

    with open(data_path) as f:
        cfg["data"] = yaml.safe_load(f)
    with open(model_path) as f:
        cfg["model"] = yaml.safe_load(f)

    return cfg
