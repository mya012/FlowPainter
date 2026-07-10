"""High-level training pipeline."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from flow_painter.config.defaults import default_output_dir
from flow_painter.evaluation.validators import validate_chairs, validate_kitti, validate_sintel
from flow_painter.io.checkpoints import load_checkpoint, save_checkpoint
from flow_painter.io.paths import ensure_dir, resolve_path
from flow_painter.models.factory import build_flow_model, register_legacy_core, wrap_data_parallel
from flow_painter.training.losses import sequence_loss
from flow_painter.training.monitoring import TrainingMonitor
from flow_painter.training.optimization import build_optimizer

register_legacy_core()
import datasets

LOGGER = logging.getLogger(__name__)


class FlowTrainer:
    """Own the full training lifecycle for the optical-flow model."""

    def __init__(self, args: Namespace) -> None:
        """Initialize the trainer.

        Args:
            args: Training configuration namespace.
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = ensure_dir(args.checkpoint_dir)
        self.log_dir = resolve_path(args.log_dir or default_output_dir(args.name))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.enabled = False

        model = build_flow_model(args, self.device)
        self.model = wrap_data_parallel(model, args.gpus)
        LOGGER.info("Parameter count: %d", self.count_parameters(self.model))

        if args.restore_ckpt is not None:
            checkpoint = load_checkpoint(args.restore_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            LOGGER.info("Restored training checkpoint from %s", resolve_path(args.restore_ckpt))

        self.model.cuda()
        self.model.train()
        if args.stage != "chairs":
            self.model.module.freeze_bn()

        self.train_loader = datasets.fetch_dataloader(args)
        self.optimizer, self.scheduler = build_optimizer(args, self.model)
        self.scaler = GradScaler(enabled=args.mixed_precision)
        self.monitor = TrainingMonitor(str(self.log_dir), self.scheduler, args.summary_freq)

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable model parameters.

        Args:
            model: Model module.

        Returns:
            Number of trainable parameters.
        """
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def fit(self) -> Path:
        """Run the training loop.

        Returns:
            Path to the final saved checkpoint.
        """
        total_steps = 0
        accumulation_counter = 0
        should_keep_training = True

        while should_keep_training:
            for data_blob in self.train_loader:
                if accumulation_counter == 0:
                    self.optimizer.zero_grad()

                image1, image2, flow, valid = [tensor.cuda() for tensor in data_blob]
                if self.args.add_noise:
                    noise_std = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + noise_std * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + noise_std * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = self.model(image1, image2, iters=self.args.iters, flow_gt=flow)
                loss, metrics = sequence_loss(flow_predictions, flow, valid, self.args.gamma)
                loss = loss / self.args.accumulation_steps
                self.scaler.scale(loss).backward()

                accumulation_counter += 1
                if accumulation_counter == self.args.accumulation_steps:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()
                    accumulation_counter = 0

                self.monitor.push(metrics)
                if total_steps % self.args.validation_freq == self.args.validation_freq - 1:
                    self._save_step_checkpoint(total_steps + 1)
                    self._run_validation()

                total_steps += 1
                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        self.monitor.close()
        return save_checkpoint(self.model.state_dict(), self.checkpoint_dir / f"{self.args.name}.pth")

    def _save_step_checkpoint(self, step: int) -> None:
        save_checkpoint(self.model.state_dict(), self.checkpoint_dir / f"{step}_{self.args.name}.pth")

    def _run_validation(self) -> None:
        results: dict[str, float] = {}
        for val_dataset in self.args.validation:
            if val_dataset == "chairs":
                results.update(validate_chairs(self.model.module, self.device))
            elif val_dataset == "sintel":
                results.update(validate_sintel(self.model.module, self.device))
            elif val_dataset == "kitti":
                results.update(validate_kitti(self.model.module, self.device))
            else:
                LOGGER.warning("Skipping unknown validation dataset: %s", val_dataset)

        self.monitor.write_dict(results)
        self.model.train()
        if self.args.stage != "chairs":
            self.model.module.freeze_bn()

