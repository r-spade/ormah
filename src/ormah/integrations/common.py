"""Host setup primitives. Host schemas and paths remain in their own modules."""
from __future__ import annotations

import os
import sys
from importlib.resources import files
from pathlib import Path

from .ownership import Installation


def receipt(host: str, project: Path | None = None) -> Path:
    root = (project / ".ormah" if project else
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ormah")
    return root / "integrations" / f"{host}.json"


def mcp_command(host: str, workspace: str | None = None) -> list[str]:
    command = [sys.executable, "-m", "ormah.integrations.mcp", "--host", host]
    if workspace:
        command += ["--workspace", workspace]
    return command


def instructions() -> str:
    return files("ormah.integrations").joinpath("assets/instructions.md").read_text()


def disconnect(host: str, project: Path | None = None, *, json5: bool = False) -> None:
    preserved = Installation(receipt(host, project), json5=json5).disconnect()
    if preserved:
        print("Preserved user-edited Ormah artifacts: " + ", ".join(preserved), file=sys.stderr)


def capability(host: str, *, whisper: str, detail: str,
               project: Path | None = None, json5: bool = False) -> dict:
    try:
        installation = Installation(receipt(host, project), json5=json5)
        tools = installation.intact("value")
        intact = installation.intact()
    except (OSError, ValueError, KeyError, TypeError):
        tools = intact = False
    return {
        "tools": "mcp" if tools else "unconfigured",
        "whisper": whisper if intact else "unconfigured",
        "maintenance": "deliberate" if tools else "unconfigured",
        "detail": detail,
        "verified_live": False,
    }
