"""Training command-line interface."""

from __future__ import annotations

import argparse
import logging

from flow_painter.config.defaults import default_base_model_checkpoint
from flow_painter.config.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description="Train FlowPainter optical-flow model.")
    parser.add_argument("--name", default="flowpainter", help="Experiment name.")
    parser.add_argument("--stage", required=True, help="Training stage: chairs, things, sintel, or kitti.")
    parser.add_argument("--restore_ckpt", default=None, help="Optional checkpoint to restore.")
    parser.add_argument("--base_model_ckpt", default=str(default_base_model_checkpoint()), help="Base flow model checkpoint.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--log_dir", default=None, help="TensorBoard directory. Defaults to runs/<name>.")
    parser.add_argument("--chairs_root", default=None, help="FlyingChairs dataset root.")
    parser.add_argument("--things_root", default=None, help="FlyingThings3D dataset root.")
    parser.add_argument("--sintel_root", default=None, help="Sintel dataset root.")
    parser.add_argument("--kitti_root", default=None, help="KITTI dataset root.")
    parser.add_argument("--hd1k_root", default=None, help="HD1K dataset root.")
    parser.add_argument("--small", action="store_true", help="Use the small model variant when supported.")
    parser.add_argument("--validation", type=str, nargs="+", default=[], help="Validation datasets.")
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--image_size", type=int, nargs="+", default=[384, 512])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision.")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--wdecay", type=float, default=0.00005)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.8, help="Exponential loss weighting.")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=3, help="Gradient accumulation steps.")
    parser.add_argument("--summary_freq", type=int, default=100, help="Training log frequency.")
    parser.add_argument("--validation_freq", type=int, default=5000, help="Validation/checkpoint frequency.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--log_level", default="INFO", help="Python logging level.")
    return parser


def main() -> None:
    """Run training from CLI arguments.

    Returns:
        None.
    """
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    from flow_painter.training.trainer import FlowTrainer

    final_checkpoint = FlowTrainer(args).fit()
    LOGGER.info("Training finished. Final checkpoint: %s", final_checkpoint)


if __name__ == "__main__":
    main()
