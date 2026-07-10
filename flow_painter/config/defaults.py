"""Default runtime values shared by CLI entry points."""

from __future__ import annotations

from pathlib import Path

from flow_painter.io.paths import project_root


def default_base_model_checkpoint() -> Path:
    """Return the default auxiliary optical-flow checkpoint.

    Returns:
        Path to the base model checkpoint relative to this repository.
    """
    return project_root() / "weights" / "100000_raft-sintel.pth"


def default_output_dir(name: str) -> Path:
    """Return the default output directory for an experiment.

    Args:
        name: Experiment name.

    Returns:
        Directory path under ``runs/``.
    """
    return project_root() / "runs" / name

