"""Checkpoint loading and saving with friendly errors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from flow_painter.io.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove a leading ``module.`` prefix from DataParallel checkpoints.

    Args:
        state_dict: Raw PyTorch state dictionary.

    Returns:
        State dictionary compatible with a non-DataParallel module.
    """
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint.

    Args:
        path: Checkpoint path.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        Loaded checkpoint object.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded.
    """
    checkpoint_path = resolve_path(path)
    try:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        return torch.load(checkpoint_path, map_location=map_location)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc


def save_checkpoint(state_dict: dict[str, Any], path: str | Path) -> Path:
    """Save a PyTorch state dictionary.

    Args:
        state_dict: Model state dictionary to save.
        path: Output checkpoint path.

    Returns:
        Absolute saved checkpoint path.

    Raises:
        RuntimeError: If saving fails.
    """
    checkpoint_path = resolve_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(state_dict, checkpoint_path)
        LOGGER.info("Saved checkpoint to %s", checkpoint_path)
        return checkpoint_path
    except Exception as exc:
        raise RuntimeError(f"Failed to save checkpoint {checkpoint_path}: {exc}") from exc

