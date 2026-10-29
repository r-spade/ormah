"""GitHub Copilot in the VS Code Local and shared Copilot runtimes."""
import json
import os
import platform
import shlex
import sys
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "github_copilot"
ENV_KEYS = ("ORMAH_URL", "ORMAH_PORT", "ORMAH_SPACE", "ORMAH_WORKSPACE", "ORMAH_AUTH_TOKEN", "ORMAH_WHISPER_TIMEOUT")


def user_directory() -> Path:
    if override := os.environ.get("ORMAH_VSCODE_USER_DIR"):
        return Path(override).expanduser().resolve()
    if platform.system() == "Darwin":
        return Path.home() / "Library/Application Support/Code/User"
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "Code/User"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Code/User"


def detected() -> bool:
    return (any((Path.home() / ".vscode/extensions").glob("github.copilot-chat-*"))
            or (user_directory() / "globalStorage/github.copilot-chat").is_dir())


def connect(project: Path | None = None) -> None:
    install = Installation(common.receipt(HOST, project))
    config = (project / ".vscode" if project else user_directory()) / "mcp.json"
    command = common.mcp_command(HOST, str(project) if project else "${workspaceFolder}")
    install.value(config, ["servers", "ormah"], {
        "type": "stdio", "command": command[0], "args": command[1:],
        "env": {key: "${env:" + key + "}" for key in ENV_KEYS if key in os.environ},
    })
    command = [sys.executable, "-m", "ormah.integrations.github_copilot_hook"]
    if project:
        command += ["--workspace", str(project)]
    hook = {
        "type": "command", "command": shlex.join(command),
        "windows": "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in command),
        "cwd": ".", "timeout": 12,
    }
    hooks = (project / ".github/hooks" if project else Path.home() / ".copilot/hooks")
    install.file(hooks / "ormah-vscode.json", json.dumps({
        "hooks": {"UserPromptSubmit": [hook]},
    }, indent=2) + "\n")
    # A separate file keeps the existing Local receipt stable during upgrades.
    native_command = command + ["--event", "userPromptTransformed"]
    native_hook = {
        "type": "command", "command": shlex.join(native_command),
        "windows": "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in native_command),
        "cwd": ".", "timeoutSec": 12,
    }
    install.file(hooks / "ormah-copilot.json", json.dumps({
        "version": 1, "hooks": {"userPromptTransformed": [native_hook]},
    }, indent=2) + "\n")
    if project:
        install.file(project / ".github/instructions/ormah.instructions.md",
                     '---\napplyTo: "**"\n---\n\n' + common.instructions())
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, project)


def status(project: Path | None = None) -> dict:
    return common.capability(HOST, whisper="native_hook", project=project, detail=(
        "VS Code Local and Copilot: MCP and prompt hooks configured. "
        "Requires trusted workspace and enabled hooks. Shared Copilot requires "
        "userPromptTransformed support. CLI and remote hosts are not verified."
    ))


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="GitHub Copilot (VS Code)", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
