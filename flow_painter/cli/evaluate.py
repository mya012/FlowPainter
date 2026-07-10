"""Evaluation and submission command-line interface."""

from __future__ import annotations

import argparse
import logging
import os

from flow_painter.config.defaults import default_base_model_checkpoint
from flow_painter.config.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation argument parser.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description="Evaluate FlowPainter or create benchmark submissions.")
    parser.add_argument("--model", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--dataset", default="kitti", choices=["chairs", "sintel", "kitti"], help="Target dataset.")
    parser.add_argument("--task", default="submission", choices=["submission", "validation", "both", "stats"], help="Evaluation task.")
    parser.add_argument("--output_path", default=None, help="Submission output directory.")
    parser.add_argument("--chairs_root", default=None, help="FlyingChairs dataset root.")
    parser.add_argument("--sintel_root", default=None, help="Sintel dataset root.")
    parser.add_argument("--kitti_root", default=None, help="KITTI dataset root.")
    parser.add_argument("--base_model_ckpt", default=str(default_base_model_checkpoint()), help="Base flow model checkpoint.")
    parser.add_argument("--base_model_device", default=None, help="Device for the auxiliary base model.")
    parser.add_argument("--small", action="store_true", help="Use the small model variant when supported.")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision.")
    parser.add_argument("--alternate_corr", action="store_true", help="Use alternate correlation implementation when supported.")
    parser.add_argument("--iters", type=int, default=None, help="Override recurrent refinement iterations.")
    parser.add_argument("--warm_start", action="store_true", help="Warm start Sintel submission sequences.")
    parser.add_argument("--profile", action="store_true", help="Report parameter counts and optional FLOPs before running.")
    parser.add_argument("--hf_endpoint", default="https://hf-mirror.com", help="HuggingFace endpoint.")
    parser.add_argument("--mpl_config_dir", default="/tmp/matplotlib", help="Matplotlib config directory.")
    parser.add_argument("--log_level", default="INFO", help="Python logging level.")
    return parser


def main() -> None:
    """Run evaluation or submission creation.

    Returns:
        None.
    """
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("MPLCONFIGDIR", args.mpl_config_dir)
    if args.chairs_root:
        os.environ["FLOWPAINTER_CHAIRS_ROOT"] = args.chairs_root
    if args.sintel_root:
        os.environ["FLOWPAINTER_SINTEL_ROOT"] = args.sintel_root
    if args.kitti_root:
        os.environ["FLOWPAINTER_KITTI_ROOT"] = args.kitti_root

    from flow_painter.evaluation.profiling import report_model_stats
    from flow_painter.evaluation.submissions import create_kitti_submission, create_sintel_submission
    from flow_painter.evaluation.validators import validate_chairs, validate_kitti, validate_sintel
    from flow_painter.inference.engine import FlowInferenceEngine

    engine = FlowInferenceEngine(args)
    iters = args.iters or {"chairs": 24, "sintel": 32, "kitti": 24}[args.dataset]

    if args.profile or args.task == "stats":
        report_model_stats(engine.model, args.dataset, engine.device, iters)
        if args.task == "stats":
            return

    if args.task in {"validation", "both"}:
        if args.dataset == "chairs":
            results = validate_chairs(engine.model, engine.device, iters)
        elif args.dataset == "sintel":
            results = validate_sintel(engine.model, engine.device, iters)
        else:
            results = validate_kitti(engine.model, engine.device, iters)
        LOGGER.info("Validation results: %s", results)

    if args.task in {"submission", "both"}:
        if args.dataset == "sintel":
            create_sintel_submission(
                engine.model,
                engine.device,
                iters=iters,
                warm_start=args.warm_start,
                output_path=args.output_path or "sintel_submission",
            )
        elif args.dataset == "kitti":
            create_kitti_submission(
                engine.model,
                engine.device,
                iters=iters,
                output_path=args.output_path or "kitti_submission",
            )
        else:
            LOGGER.warning("FlyingChairs does not define a benchmark submission writer.")


if __name__ == "__main__":
    main()
