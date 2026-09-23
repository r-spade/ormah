"""Antigravity shared customizations, including the native agy CLI."""
import platform
import shlex
import sys
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "antigravity"


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("agy") or _find_binary("antigravity")
                or any((Path.home() / ".gemini" / name).is_dir()
                       for name in ("antigravity", "antigravity-cli", "antigravity-ide")))


def connect(project: Path | None = None) -> None:
    if platform.system() == "Windows":
        raise ValueError("Antigravity hook shell quoting is validated on POSIX only; Windows setup is unavailable")
    root = project / ".agents" if project else Path.home() / ".gemini/config"
    install = Installation(common.receipt(HOST, project))
    command = common.mcp_command(HOST, str(project) if project else None)
    install.value(root / "mcp_config.json", ["mcpServers", "ormah"], {
        "command": command[0], "args": command[1:],
    })
    command = [sys.executable, "-m", "ormah.integrations.antigravity_hook"]
    if project:
        command += ["--workspace", str(project)]
    install.value(root / "hooks.json", ["ormah"], {
        "PreInvocation": [{"type": "command", "command": shlex.join(command), "timeout": 12}],
    })
    install.file(root / "ormah-instructions.md", common.instructions())
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, project)


def status(project: Path | None = None) -> dict:
    result = common.capability(HOST, whisper="native_hook", project=project, detail=(
        "Antigravity CLI PreInvocation whisper + MCP configured (POSIX). "
        "Desktop/IDE transcript compatibility unverified; Windows setup unavailable. "
        "User-scope installation uses global memory unless ORMAH_SPACE is set."
    ))
    result["surfaces"] = {"cli": result["whisper"], "desktop": "unverified", "ide": "unverified"}
    return result


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="Antigravity (including CLI)", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
