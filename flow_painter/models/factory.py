"""Model construction helpers."""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from flow_painter.config.defaults import default_base_model_checkpoint
from flow_painter.io.checkpoints import load_checkpoint, normalize_state_dict
from flow_painter.io.paths import project_root, resolve_path

LOGGER = logging.getLogger(__name__)


def register_legacy_core() -> None:
    """Make the legacy numerical core importable.

    Returns:
        None.
    """
    core_dir = project_root() / "core"
    core_path = str(core_dir)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)


def build_flow_model(args: Namespace, device: torch.device) -> nn.Module:
    """Instantiate the optical-flow model without altering its math.

    Args:
        args: Parsed CLI/config namespace.
        device: Device where the main model should run.

    Returns:
        Flow model placed on ``device``.
    """
    register_legacy_core()
    from flowpainter import FlowPainter

    if not hasattr(args, "base_model_ckpt") or args.base_model_ckpt is None:
        args.base_model_ckpt = str(default_base_model_checkpoint())
    else:
        args.base_model_ckpt = str(resolve_path(args.base_model_ckpt))

    model = FlowDiffuser(args).to(device)
    LOGGER.info("Initialized FlowPainter backbone on %s", device)
    return model


def load_model_weights(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Load model weights into an existing module.

    Args:
        model: Target model.
        checkpoint_path: Checkpoint path.
        device: Device used for checkpoint mapping.

    Returns:
        Model with loaded weights.
    """
    state_dict = normalize_state_dict(load_checkpoint(checkpoint_path, map_location=device))
    if any(key.startswith("trans.tb.") for key in state_dict):
        state_dict = {
            key.replace("trans.tb.", "trans.feature_mixer.", 1) if key.startswith("trans.tb.") else key: value
            for key, value in state_dict.items()
        }
        LOGGER.info("Mapped legacy checkpoint keys from trans.tb.* to trans.feature_mixer.*")
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded model weights from %s", resolve_path(checkpoint_path))
    return model


def wrap_data_parallel(model: nn.Module, gpu_ids: list[int]) -> nn.DataParallel:
    """Wrap a model with ``torch.nn.DataParallel``.

    Args:
        model: Model to wrap.
        gpu_ids: CUDA device ids.

    Returns:
        DataParallel wrapper.
    """
    return nn.DataParallel(model, device_ids=gpu_ids)
