"""Cline VS Code: workspace-bound MCP and executable prompt hook."""
import os
import platform
import shlex
import sys
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "cline"


def config_path() -> Path:
    if override := os.environ.get("CLINE_MCP_SETTINGS_PATH"):
        return Path(override).expanduser().resolve()
    root = Path(os.environ.get("CLINE_DIR", Path.home() / ".cline"))
    data = Path(os.environ.get("CLINE_DATA_DIR", root / "data"))
    return data / "settings/cline_mcp_settings.json"


def detected() -> bool:
    return (config_path().exists()
            or any((Path.home() / ".vscode/extensions").glob("saoudrizwan.claude-dev-*")))


def connect(project: Path | None = None) -> None:
    if project is None:
        raise ValueError("Cline requires --project PATH to bind MCP and whisper to the same workspace")
    project = project.resolve()
    install = Installation(common.receipt(HOST))
    command = common.mcp_command(HOST, str(project))
    install.value(config_path(), ["mcpServers", "ormah"], {
        "type": "stdio", "command": command[0], "args": command[1:],
        "disabled": False, "autoApprove": [],
        "env": {key: "${env:" + key + "}" for key in
                ("ORMAH_URL", "ORMAH_PORT", "ORMAH_SPACE", "ORMAH_AUTH_TOKEN")
                if key in os.environ},
    })
    command = [sys.executable, "-m", "ormah.integrations.cline_hook", "--workspace", str(project)]
    root = project / ".clinerules"
    if root.is_file():
        raise ValueError("Cline uses a .clinerules file; migrate it to a directory before installing hooks")
    windows = platform.system() == "Windows"
    script = root / "hooks" / ("UserPromptSubmit.ps1" if windows else "UserPromptSubmit")
    if windows:
        # The native launcher uses PowerShell -File; stdin is not implicitly piped.
        content = "$payload = [Console]::In.ReadToEnd()\n$payload | & " + " ".join(
            "'" + arg.replace("'", "''") + "'" for arg in command) + "\nexit 0\n"
    else:
        content = "#!/bin/sh\nexec " + shlex.join(command) + "\n"
    install.file(script, content)
    install.file(root / "ormah.md", common.instructions())
    install.commit()
    if not windows:
        script.chmod(script.stat().st_mode | 0o100)


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST)


def status(project: Path | None = None) -> dict:
    result = common.capability(HOST, whisper="native_hook", detail=(
        "Cline VS Code: MCP + UserPromptSubmit configured for one workspace. "
        "Enable hooks in Cline settings; multi-root windows and CLI hooks are unconfigured."
    ))
    if result["whisper"] == "native_hook" and platform.system() != "Windows":
        install = Installation(common.receipt(HOST))
        scripts = [Path(r["path"]) for r in install.records
                   if r["kind"] == "file" and Path(r["path"]).name == "UserPromptSubmit"]
        if not scripts or not all(os.access(p, os.X_OK) for p in scripts):
            result["whisper"] = "unconfigured"
    return result


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="Cline (VS Code)", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
