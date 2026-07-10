"""Training metric logging."""

from __future__ import annotations

import logging

from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


class TrainingMonitor:
    """Accumulate training metrics and write TensorBoard summaries."""

    def __init__(self, log_dir: str, scheduler, summary_freq: int = 100) -> None:
        """Initialize the monitor.

        Args:
            log_dir: TensorBoard output directory.
            scheduler: Learning-rate scheduler exposing ``get_last_lr``.
            summary_freq: Number of optimization steps per log event.
        """
        self.log_dir = log_dir
        self.scheduler = scheduler
        self.summary_freq = summary_freq
        self.total_steps = 0
        self.running_loss: dict[str, float] = {}
        self.writer: SummaryWriter | None = None

    def push(self, metrics: dict[str, float]) -> None:
        """Add one step of metrics.

        Args:
            metrics: Scalar metric values.

        Returns:
            None.
        """
        self.total_steps += 1
        for key, value in metrics.items():
            self.running_loss[key] = self.running_loss.get(key, 0.0) + value

        if self.total_steps % self.summary_freq == self.summary_freq - 1:
            self.flush_training_status()

    def write_dict(self, results: dict[str, float]) -> None:
        """Write validation metrics.

        Args:
            results: Scalar validation results.

        Returns:
            None.
        """
        writer = self._writer()
        for key, value in results.items():
            writer.add_scalar(key, value, self.total_steps)

    def flush_training_status(self) -> None:
        """Log averaged training metrics.

        Returns:
            None.
        """
        if not self.running_loss:
            return

        averaged = {
            key: value / self.summary_freq for key, value in sorted(self.running_loss.items())
        }
        lr = self.scheduler.get_last_lr()[0]
        metrics_text = ", ".join(f"{key}={value:.4f}" for key, value in averaged.items())
        LOGGER.info("step=%d lr=%.7f %s", self.total_steps + 1, lr, metrics_text)

        writer = self._writer()
        for key, value in averaged.items():
            writer.add_scalar(key, value, self.total_steps)
        self.running_loss = {}

    def close(self) -> None:
        """Close the TensorBoard writer.

        Returns:
            None.
        """
        if self.writer is not None:
            self.writer.close()

    def _writer(self) -> SummaryWriter:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        return self.writer

