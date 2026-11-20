"""Current Devin Local/Desktop config, with explicit legacy Cascade limitations."""
import os
import platform
import shlex
import sys
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "devin_desktop"
ENV_KEYS = ("ORMAH_URL", "ORMAH_PORT", "ORMAH_SPACE", "ORMAH_WORKSPACE", "ORMAH_AUTH_TOKEN", "ORMAH_WHISPER_TIMEOUT")


def user_directory() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "devin"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "devin"


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("devin") or _find_binary("windsurf") or user_directory().is_dir()
                or (Path.home() / ".codeium/windsurf").is_dir())


def connect(project: Path | None = None) -> None:
    if platform.system() == "Windows":
        raise ValueError("Devin hook shell integration is verified for POSIX only; Windows is not configured")
    root = project / ".devin" if project else user_directory()
    install = Installation(common.receipt(HOST, project))
    command = common.mcp_command(HOST, str(project) if project else None)
    install.value(root / "mcp_config.json", ["mcpServers", "ormah"], {
        "command": command[0], "args": command[1:],
        "env": {key: "${env:" + key + "}" for key in ENV_KEYS if key in os.environ},
    })
    command = [sys.executable, "-m", "ormah.integrations.devin_desktop_hook"]
    if project:
        command += ["--workspace", str(project)]
    # Local hooks.v1.json is a different surface from Cascade hooks.json.
    path = root / "hooks.v1.json" if project else root / "config.json"
    keys = ["UserPromptSubmit"] if project else ["hooks", "UserPromptSubmit"]
    install.item(path, keys, {"matcher": "", "hooks": [{
        "type": "command", "command": shlex.join(command), "timeout": 12,
    }]})
    install.file(root / "ormah-instructions.md", common.instructions())
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, project)


def status(project: Path | None = None) -> dict:
    result = common.capability(HOST, whisper="native_hook", project=project, detail=(
        "Devin Local/Desktop and CLI: MCP + UserPromptSubmit hook. "
        "Cascade: shared user MCP only; automatic whisper blocked by host API. "
        "Project setup applies to Local only. POSIX; requires trusted workspace; no live GUI validation."
    ))
    result["surfaces"] = {
        "devin_local": {"tools": result["tools"], "whisper": result["whisper"]},
        "cascade": {"tools": result["tools"] if project is None else "unconfigured", "whisper": "blocked"},
    }
    return result


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(id=HOST, name="Devin Desktop / Windsurf", detect_fn=detected,
                           is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
                           unwire_fn=disconnect, capabilities_fn=status)
