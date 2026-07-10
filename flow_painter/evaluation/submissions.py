"""Benchmark submission writers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from flow_painter.io.paths import resolve_path
from flow_painter.models.factory import register_legacy_core

register_legacy_core()
import datasets
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate

LOGGER = logging.getLogger(__name__)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def create_sintel_submission(
    model: torch.nn.Module,
    device: torch.device,
    iters: int = 32,
    warm_start: bool = False,
    output_path: str | Path = "sintel_submission",
) -> None:
    """Create Sintel leaderboard submission files.

    Args:
        model: Optical-flow model.
        device: Inference device.
        iters: Recurrent refinement iterations.
        warm_start: Whether to propagate low-resolution flow between frames.
        output_path: Output directory for ``.flo`` files.

    Returns:
        None.
    """
    model.eval()
    root = resolve_path(output_path)

    for render_pass in ["clean", "final"]:
        test_dataset = datasets.MpiSintel(split="test", aug_params=None, dstype=render_pass)
        if len(test_dataset) == 0:
            raise RuntimeError(
                f"No Sintel {render_pass} test image pairs found. "
                "Set --sintel_root or FLOWPAINTER_SINTEL_ROOT to the Sintel dataset root."
            )
        LOGGER.info("Creating Sintel %s submission for %d image pairs", render_pass, len(test_dataset))
        flow_prev, sequence_prev = None, None

        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if (sequence != sequence_prev) or (render_pass == "final" and sequence in ["market_4"]) or render_pass == "clean":
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(device), image2[None].to(device))
            flow_low, flow_prediction = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_prediction[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].to(device)

            output_dir = root / render_pass / sequence
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_utils.writeFlow(str(output_dir / f"frame{frame + 1:04d}.flo"), flow)
            sequence_prev = sequence
            LOGGER.info("Wrote Sintel %s %s frame %04d", render_pass, sequence, frame + 1)


@torch.no_grad()
def create_kitti_submission(
    model: torch.nn.Module,
    device: torch.device,
    iters: int = 24,
    output_path: str | Path = "kitti_submission",
) -> None:
    """Create KITTI leaderboard submission PNG files.

    Args:
        model: Optical-flow model.
        device: Inference device.
        iters: Recurrent refinement iterations.
        output_path: Output directory for KITTI flow PNG files.

    Returns:
        None.
    """
    model.eval()
    root = resolve_path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    test_dataset = datasets.KITTI(split="testing", aug_params=None)
    if len(test_dataset) == 0:
        raise RuntimeError(
            "No KITTI testing image pairs found. "
            "Set --kitti_root or FLOWPAINTER_KITTI_ROOT to a root containing testing/image_2/*_10.png and *_11.png."
        )
    LOGGER.info("Creating KITTI submission for %d image pairs", len(test_dataset))

    runtimes = []
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode="kitti")
        image1, image2 = padder.pad(image1[None].to(device), image2[None].to(device))

        _sync_device(device)
        start_time = time.time()
        _, flow_prediction = model(image1, image2, iters=iters, test_mode=True)
        _sync_device(device)
        elapsed = time.time() - start_time
        runtimes.append(elapsed)
        LOGGER.info("KITTI submission image %d/%d runtime=%.6fs", test_id + 1, len(test_dataset), elapsed)

        flow = padder.unpad(flow_prediction[0]).permute(1, 2, 0).cpu().numpy()
        frame_utils.writeFlowKITTI(str(root / frame_id), flow)

    if runtimes:
        LOGGER.info("KITTI submission average runtime: %.6fs (%d images)", np.mean(runtimes), len(runtimes))
