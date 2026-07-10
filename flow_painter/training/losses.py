"""Training losses for optical-flow supervision."""

from __future__ import annotations

import torch

MAX_FLOW = 400


def sequence_loss(
    flow_predictions: list[torch.Tensor],
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = 0.8,
    max_flow: float = MAX_FLOW,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the weighted multi-step optical-flow loss.

    Args:
        flow_predictions: Predicted flow sequence, each tensor shaped ``[B, 2, H, W]``.
        flow_gt: Ground-truth flow tensor shaped ``[B, 2, H, W]``.
        valid: Valid-pixel mask shaped ``[B, H, W]``.
        gamma: Exponential weight decay over prediction steps.
        max_flow: Maximum flow magnitude used to filter outliers.

    Returns:
        A tuple containing the scalar training loss and metric dictionary.
    """
    num_predictions = len(flow_predictions)
    flow_loss = 0.0

    magnitude = torch.sum(flow_gt**2, dim=1).sqrt()
    valid_mask = (valid >= 0.5) & (magnitude < max_flow)

    for index, prediction in enumerate(flow_predictions):
        step_weight = gamma ** (num_predictions - index - 1)
        step_error = (prediction - flow_gt).abs()
        flow_loss += step_weight * (valid_mask[:, None] * step_error).mean()

    endpoint_error = torch.sum((flow_predictions[-1] - flow_gt) ** 2, dim=1).sqrt()
    endpoint_error = endpoint_error.view(-1)[valid_mask.view(-1)]

    metrics = {
        "epe": endpoint_error.mean().item(),
        "1px": (endpoint_error < 1).float().mean().item(),
        "3px": (endpoint_error < 3).float().mean().item(),
        "5px": (endpoint_error < 5).float().mean().item(),
    }
    return flow_loss, metrics

