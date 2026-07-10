"""High-level inference pipeline for FlowPainter."""

from __future__ import annotations

import logging
from argparse import Namespace

import torch

from flow_painter.models.factory import build_flow_model, load_model_weights

LOGGER = logging.getLogger(__name__)


def _resolve_device(device_name: str | torch.device | None, fallback: torch.device) -> torch.device:
    """Resolve a requested device with a clear CUDA fallback."""
    if device_name is None:
        return fallback

    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Requested CUDA device %s is unavailable; falling back to %s", requested, fallback)
        return fallback
    return requested


class FlowInferenceEngine:
    """Load a model once and expose benchmark-oriented inference methods."""

    def __init__(self, args: Namespace, device: torch.device | None = None) -> None:
        """Initialize the inference engine.

        Args:
            args: Evaluation configuration namespace.
            device: Optional inference device. Defaults to CUDA when available.
        """
        self.args = args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_flow_model(args, self.device)
        if args.model is not None:
            load_model_weights(self.model, args.model, self.device)

        base_model_device = _resolve_device(getattr(args, "base_model_device", None), self.device)
        if hasattr(self.model, "set_base_model_device"):
            try:
                self.model.set_base_model_device(base_model_device)
            except RuntimeError as exc:
                if base_model_device.type != "cuda":
                    raise
                LOGGER.warning("Failed to place base model on %s; falling back to cpu: %s", base_model_device, exc)
                self.model.set_base_model_device("cpu")
        self.model.eval()
        LOGGER.info("Main model device: %s; base model device: %s", self.device, getattr(self.model, "base_model_device", "n/a"))

    @torch.no_grad()
    def predict(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int,
        flow_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict optical flow for one image pair.

        Args:
            image1: First RGB image tensor shaped ``[B, 3, H, W]``.
            image2: Second RGB image tensor shaped ``[B, 3, H, W]``.
            iters: Recurrent refinement iterations.
            flow_init: Optional warm-start flow tensor shaped ``[B, 2, H/8, W/8]``.

        Returns:
            Tuple of low-resolution and full-resolution predicted flow tensors.
        """
        return self.model(
            image1.to(self.device),
            image2.to(self.device),
            iters=iters,
            flow_init=flow_init,
            test_mode=True,
        )
