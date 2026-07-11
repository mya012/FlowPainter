"""Logging helpers for command-line entry points."""

from __future__ import annotations

import logging
import sys


class FlowPainterLogFilter(logging.Filter):
    """Normalize project logger names in command-line output."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "flowpainter" or record.name.startswith("flow_painter") or record.name in {"datasets"}:
            record.name = "flowpainter"
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide standard logging.

    Args:
        level: Logging level name such as ``"INFO"`` or ``"DEBUG"``.

    Returns:
        None.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(FlowPainterLogFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(
        level=numeric_level,
        handlers=[handler],
        force=True,
    )
