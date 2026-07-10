"""Model statistics and optional FLOPs profiling."""

from __future__ import annotations

import logging

import torch

from flow_painter.models.factory import register_legacy_core

try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

register_legacy_core()
import datasets
from utils.utils import InputPadder

LOGGER = logging.getLogger(__name__)


class FlowModelProfileWrapper(torch.nn.Module):
    """Wrap the model so profilers call the same test-time forward path."""

    def __init__(self, model: torch.nn.Module, iters: int) -> None:
        """Initialize wrapper.

        Args:
            model: Optical-flow model.
            iters: Recurrent refinement iterations.
        """
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        """Run test-mode inference.

        Args:
            image1: First RGB image tensor shaped ``[B, 3, H, W]``.
            image2: Second RGB image tensor shaped ``[B, 3, H, W]``.

        Returns:
            Model forward outputs.
        """
        return self.model(image1, image2, iters=self.iters, test_mode=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters.

    Args:
        model: PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_count(count: int | float) -> str:
    """Format a large count using K/M/G suffixes.

    Args:
        count: Numeric count.

    Returns:
        Human-readable string.
    """
    if count >= 1e9:
        return f"{count / 1e9:.3f}G"
    if count >= 1e6:
        return f"{count / 1e6:.3f}M"
    if count >= 1e3:
        return f"{count / 1e3:.3f}K"
    return str(count)


def report_model_stats(model: torch.nn.Module, dataset_name: str, device: torch.device, iters: int) -> None:
    """Log parameter counts and optional FLOPs.

    Args:
        model: Optical-flow model.
        dataset_name: Dataset name used to choose a representative input.
        device: Profiling device.
        iters: Recurrent refinement iterations.

    Returns:
        None.
    """
    base_model = getattr(model, "base_model", None)
    total_params = count_parameters(model)
    base_params = count_parameters(base_model) if base_model is not None else 0
    LOGGER.info("Total model params: %d (%s)", total_params, format_count(total_params))
    if base_model is not None:
        LOGGER.info("Base model params: %d (%s)", base_params, format_count(base_params))
    else:
        LOGGER.info("Base model params unavailable")

    if thop_profile is None:
        LOGGER.info("FLOPs skipped; install thop to enable profiling")
        return

    try:
        if dataset_name == "sintel":
            sample_dataset = datasets.MpiSintel(split="training", dstype="clean")
        elif dataset_name == "kitti":
            sample_dataset = datasets.KITTI(split="training")
        else:
            sample_dataset = datasets.FlyingChairs(split="validation")

        sample = sample_dataset[0]
        image1, image2 = sample[0], sample[1]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape, mode="kitti" if dataset_name == "kitti" else "sintel")
        image1, image2 = padder.pad(image1, image2)
        profiled_model = FlowModelProfileWrapper(model, iters).to(device)
        profiled_model.eval()

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_flops, _ = thop_profile(profiled_model, inputs=(image1, image2), verbose=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        LOGGER.info("Total model FLOPs/image: %.0f (%s)", total_flops, format_count(total_flops))

        if base_model is not None:
            base_flops, _ = thop_profile(base_model, inputs=(image1, image2), verbose=False)
            LOGGER.info("Base model FLOPs/image: %.0f (%s)", base_flops, format_count(base_flops))
    except Exception as exc:
        LOGGER.warning("FLOPs profiling failed: %r", exc)

