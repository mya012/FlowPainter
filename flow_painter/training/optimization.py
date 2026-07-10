"""Optimizer and scheduler construction."""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn


def build_optimizer(args: Namespace, model: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create the AdamW optimizer and OneCycleLR scheduler.

    Args:
        args: Training configuration namespace.
        model: Trainable model.

    Returns:
        Optimizer and scheduler pair.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=args.epsilon,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler

