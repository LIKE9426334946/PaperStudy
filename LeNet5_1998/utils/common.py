"""Common experiment utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML experiment configuration."""

    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"experiment", "data", "model", "loss", "training"}
    if not isinstance(config, dict) or not required.issubset(config):
        missing = sorted(required - set(config or {}))
        raise ValueError(f"Configuration is missing sections: {missing}.")
    return config


def resolve_project_paths(
    config: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Resolve all configured artifact paths from the paper directory."""

    root = Path(project_root).resolve()
    config["data"]["root"] = str(root / config["data"]["root"])
    experiment = config["experiment"]
    experiment["output_dir"] = str(root / experiment["output_dir"])
    experiment["checkpoint_dir"] = str(root / experiment["checkpoint_dir"])
    return config


def choose_device(requested: str) -> torch.device:
    """Select CUDA, MPS or CPU, unless an explicit device was requested."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible(seed: int) -> None:
    """Seed Python and PyTorch and request deterministic kernels."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def learning_rate_for_epoch(
    schedule: list[list[float]],
    epoch: int,
) -> float:
    """Return the last scheduled learning rate whose start epoch is active."""

    ordered = sorted((int(start), float(rate)) for start, rate in schedule)
    if not ordered or ordered[0][0] != 0:
        raise ValueError("Learning-rate schedule must start at epoch zero.")
    rate = ordered[0][1]
    for start, candidate in ordered:
        if epoch < start:
            break
        rate = candidate
    return rate
