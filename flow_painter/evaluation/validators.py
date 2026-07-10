"""Dataset validators for optical-flow benchmarks."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from flow_painter.models.factory import register_legacy_core

register_legacy_core()
import datasets
from utils.utils import InputPadder

LOGGER = logging.getLogger(__name__)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def validate_chairs(model: torch.nn.Module, device: torch.device, iters: int = 24) -> dict[str, float]:
    """Evaluate on FlyingChairs validation split.

    Args:
        model: Optical-flow model.
        device: Evaluation device.
        iters: Recurrent refinement iterations.

    Returns:
        Dictionary with the ``chairs`` EPE metric.
    """
    model.eval()
    epe_list = []
    runtimes = []
    val_dataset = datasets.FlyingChairs(split="validation")

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        _sync_device(device)
        start_time = time.time()
        _, flow_prediction = model(image1, image2, iters=iters, test_mode=True)
        _sync_device(device)
        elapsed = time.time() - start_time
        runtimes.append(elapsed)
        LOGGER.info("chairs image %d/%d runtime=%.6fs", val_id + 1, len(val_dataset), elapsed)

        epe = torch.sum((flow_prediction[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = float(np.mean(np.concatenate(epe_list)))
    LOGGER.info("Validation Chairs EPE: %.6f", epe)
    if runtimes:
        LOGGER.info("Validation Chairs average runtime: %.6fs (%d images)", np.mean(runtimes), len(runtimes))
    return {"chairs": epe}


@torch.no_grad()
def validate_sintel(model: torch.nn.Module, device: torch.device, iters: int = 32) -> dict[str, float]:
    """Evaluate on Sintel clean and final training splits.

    Args:
        model: Optical-flow model.
        device: Evaluation device.
        iters: Recurrent refinement iterations.

    Returns:
        Dictionary with ``clean`` and ``final`` EPE metrics.
    """
    model.eval()
    results = {}

    for render_pass in ["clean", "final"]:
        val_dataset = datasets.MpiSintel(split="training", dstype=render_pass)
        epe_list = []
        runtimes = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].to(device)
            image2 = image2[None].to(device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _sync_device(device)
            start_time = time.time()
            _, flow_prediction = model(image1, image2, iters=iters, test_mode=True)
            _sync_device(device)
            flow = padder.unpad(flow_prediction[0]).cpu()
            elapsed = time.time() - start_time
            runtimes.append(elapsed)
            LOGGER.info("%s image %d/%d runtime=%.6fs", render_pass, val_id + 1, len(val_dataset), elapsed)

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = float(np.mean(epe_all))
        px1 = float(np.mean(epe_all < 1))
        px3 = float(np.mean(epe_all < 3))
        px5 = float(np.mean(epe_all < 5))
        LOGGER.info(
            "Validation %s EPE: %.6f, 1px: %.6f, 3px: %.6f, 5px: %.6f",
            render_pass,
            epe,
            px1,
            px3,
            px5,
        )
        if runtimes:
            LOGGER.info("Validation %s average runtime: %.6fs (%d images)", render_pass, np.mean(runtimes), len(runtimes))
        results[render_pass] = float(np.mean(epe_list))

    return results


@torch.no_grad()
def validate_kitti(model: torch.nn.Module, device: torch.device, iters: int = 24) -> dict[str, float]:
    """Evaluate on KITTI-2015 training split.

    Args:
        model: Optical-flow model.
        device: Evaluation device.
        iters: Recurrent refinement iterations.

    Returns:
        Dictionary with ``kitti-epe`` and ``kitti-f1`` metrics.
    """
    model.eval()
    val_dataset = datasets.KITTI(split="training")

    out_list, epe_list = [], []
    runtimes = []
    invalid_frames = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape, mode="kitti")
        image1, image2 = padder.pad(image1, image2)

        _sync_device(device)
        start_time = time.time()
        _, flow_prediction = model(image1, image2, iters=iters, test_mode=True)
        _sync_device(device)
        elapsed = time.time() - start_time
        runtimes.append(elapsed)
        LOGGER.info("kitti image %d/%d runtime=%.6fs", val_id + 1, len(val_dataset), elapsed)

        flow = padder.unpad(flow_prediction[0]).cpu()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        magnitude = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        magnitude = magnitude.view(-1)
        valid = valid_gt.view(-1) >= 0.5

        frame_info = val_dataset.extra_info[val_id][0] if val_dataset.extra_info else str(val_id)
        valid_count = int(valid.sum().item())
        if torch.isnan(flow).any() or torch.isinf(flow).any() or torch.isnan(epe).any() or torch.isinf(epe).any() or valid_count == 0:
            invalid_frames.append(frame_info)
            LOGGER.warning("KITTI frame %s has invalid flow/epe values or empty valid mask", frame_info)

        out = ((epe > 3.0) & ((epe / magnitude) > 0.05)).float()
        if valid_count == 0:
            continue

        valid_epe = epe[valid]
        valid_epe_mean = valid_epe.mean().item()
        if not np.isfinite(valid_epe_mean):
            invalid_frames.append(frame_info)
            LOGGER.warning("KITTI frame %s produced non-finite valid EPE", frame_info)

        epe_list.append(valid_epe_mean)
        out_list.append(out[valid].cpu().numpy())

    epe = float(np.mean(np.array(epe_list)))
    f1 = float(100 * np.mean(np.concatenate(out_list)))

    if invalid_frames:
        unique_invalid_frames = list(dict.fromkeys(invalid_frames))
        LOGGER.warning("KITTI invalid frames (%d): %s", len(unique_invalid_frames), ", ".join(unique_invalid_frames))
    else:
        LOGGER.info("No NaN/Inf detected during KITTI validation")

    LOGGER.info("Validation KITTI: epe=%.6f f1=%.6f", epe, f1)
    if runtimes:
        LOGGER.info("Validation KITTI average runtime: %.6fs (%d images)", np.mean(runtimes), len(runtimes))
    return {"kitti-epe": epe, "kitti-f1": f1}

