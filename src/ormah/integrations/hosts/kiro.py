"""Kiro's current shared IDE/V3 hook format, not legacy embedded agent hooks."""
import json
import os
import platform
import shlex
import sys
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "kiro"
ENV_KEYS = ("ORMAH_URL", "ORMAH_PORT", "ORMAH_SPACE", "ORMAH_WORKSPACE", "ORMAH_AUTH_TOKEN", "ORMAH_WHISPER_TIMEOUT")


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("kiro") or _find_binary("kiro-cli") or (Path.home() / ".kiro").is_dir())


def connect(project: Path | None = None) -> None:
    if platform.system() == "Windows":
        raise ValueError("Kiro integration currently supports POSIX command hooks; Windows setup is unverified")
    root = (project or Path.home()) / ".kiro"
    install = Installation(common.receipt(HOST, project))
    command = common.mcp_command(HOST, str(project) if project else None)
    install.value(root / "settings/mcp.json", ["mcpServers", "ormah"], {
        "command": command[0], "args": command[1:],
        "env": {key: "${" + key + "}" for key in ENV_KEYS if key in os.environ},
    })
    command = [sys.executable, "-m", "ormah.integrations.kiro_hook"]
    if project:
        command += ["--workspace", str(project)]
    install.file(root / "hooks/ormah.json", json.dumps({
        "version": "v1", "hooks": [{
            "name": "Ormah whisper", "trigger": "UserPromptSubmit",
            "action": {"type": "command", "command": shlex.join(command)},
            "timeout": 12, "enabled": True,
        }],
    }, indent=2) + "\n")
    install.file(root / "steering/ormah.md", '---\ninclusion: always\n---\n\n' + common.instructions())
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, project)


def status(project: Path | None = None) -> dict:
    return common.capability(HOST, whisper="native_hook", project=project, detail=(
        "Kiro IDE >=1.0.293 and CLI V3 engine (kiro-cli --v3, launcher >=2.14.2): "
        "UserPromptSubmit stdout whisper + MCP. POSIX; review host hook trust and MCP env approvals. "
        "Legacy IDE/CLI engines and cloud sessions are not configured; no live authenticated validation."
    ))


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(id=HOST, name="Kiro (IDE / CLI V3)", detect_fn=detected,
                           is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
                           unwire_fn=disconnect, capabilities_fn=status)
