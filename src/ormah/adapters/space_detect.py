"""Shared space detection for CLI and MCP adapters."""

from __future__ import annotations

import os
import subprocess


def detect_space_from_dir(path: str) -> str | None:
    """Detect the project space from an explicit directory path.

    Tries git repo basename first, falls back to directory basename.
    Returns None if the path is a home directory or root.
    """
    path = os.path.realpath(path)

    # Don't auto-detect space for home directory or root
    home = os.path.expanduser("~")
    if path == home or path == "/":
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
        if result.returncode == 0:
            return os.path.basename(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fall back to directory basename
    return os.path.basename(path)


def detect_space_from_cwd() -> str | None:
    """Detect the project space from the current working directory.

    Tries git repo basename first, falls back to cwd basename.
    Returns None if running from a home directory or root.
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
