"""Portable paths for ignored, machine-local manifests and workspace profiles."""

import os
from pathlib import Path


def relative_or_absolute(path: str | Path, start: str | Path) -> str:
    """Prefer a relative path; retain an absolute path across Windows drives/shares."""
    path = Path(path).resolve()
    try:
        return Path(os.path.relpath(path, start)).as_posix()
    except ValueError:
        # Windows cannot express a relative path between different drives or shares.
        return path.as_posix()
