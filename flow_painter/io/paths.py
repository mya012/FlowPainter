"""Path utilities used by training, evaluation, and submission code."""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def project_root() -> Path:
    """Return the repository root.

    Returns:
        Absolute repository root path.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: PathLike, base_dir: PathLike | None = None) -> Path:
    """Resolve a user-provided path against the project root.

    Args:
        path: Absolute or relative file-system path.
        base_dir: Optional base directory. Defaults to the project root.

    Returns:
        Absolute normalized path.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    root = Path(base_dir).expanduser().resolve() if base_dir is not None else project_root()
    return (root / candidate).resolve()


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Absolute path to the directory.
    """
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

