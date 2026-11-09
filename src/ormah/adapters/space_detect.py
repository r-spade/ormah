"""Shared space detection for CLI and MCP adapters."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def detect_space_from_dir(path: str) -> str | None:
    """Detect the project space from an explicit directory path.

    Tries git repo basename first, falls back to directory basename.
    Returns None if the path is the home directory or one of its ancestors (root included).
    """
    path = os.path.realpath(path)

    # Don't auto-detect space for home directory or any folder above it
    home = os.path.realpath(os.path.expanduser("~"))
    if Path(home).is_relative_to(path):
        return None

    # Try git repo root basename
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=path,
        )
        repo_root = result.stdout.strip()
        # A dotfiles repo at home would claim every folder under it.
        if result.returncode == 0 and os.path.realpath(repo_root) != home:
            return os.path.basename(repo_root)
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Fall back to directory basename
    return os.path.basename(path)


def detect_space_from_cwd() -> str | None:
    """Detect the project space from the current working directory.

    Tries git repo basename first, falls back to cwd basename.
    Returns None if running from the home directory or one of its ancestors.
    """
    return detect_space_from_dir(os.getcwd())


def resolve_space(explicit: str | None = None) -> str | None:
    """Resolve space: explicit flag > ORMAH_SPACE env > cwd detection."""
    if explicit:
        return explicit
    env_space = os.environ.get("ORMAH_SPACE")
    if env_space:
        return env_space
    return detect_space_from_cwd()
