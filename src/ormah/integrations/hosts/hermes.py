"""Hermes native per-turn hook and MCP, scoped to an explicit Hermes profile."""
import json
import os
import platform
import sys
from importlib.resources import files
from pathlib import Path

from ormah.integrations import common, yaml_config
from ormah.integrations.ownership import Installation

HOST = "hermes"
ENV_KEYS = ("ORMAH_URL", "ORMAH_PORT", "ORMAH_SPACE", "ORMAH_WORKSPACE", "ORMAH_AUTH_TOKEN", "ORMAH_WHISPER_TIMEOUT")


def home() -> Path:
    if override := os.environ.get("HERMES_HOME"):
        return Path(os.path.expandvars(override)).expanduser().resolve()
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "hermes"
    return Path.home() / ".hermes"


def installation() -> Installation:
    install = Installation(home() / ".ormah/installation.json")
    install.editor = yaml_config
    return install


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("hermes") or (home() / "config.yaml").exists())


def connect(project: Path | None = None) -> None:
    if os.environ.get("HERMES_SAFE_MODE", "").lower() in ("1", "true", "yes"):
        raise ValueError("HERMES_SAFE_MODE disables plugins; automatic whisper cannot be configured in safe mode")
    path = home() / "config.yaml"
    text = path.read_text() if path.exists() else "{}\n"
    disabled = yaml_config.get(text, ["plugins", "disabled"])[1] or []
    if "ormah" in disabled:
        raise ValueError("Hermes explicitly disables Ormah; preserving that choice")
    install = installation()
    command = common.mcp_command(HOST, str(project) if project else None)
    names = [key for key in ENV_KEYS if key in os.environ]
    install.value(path, ["mcp_servers", "ormah"], {
        "command": command[0], "args": command[1:],
        "env": {key: "${" + key + "}" for key in names},
    })
    install.item(path, ["plugins", "enabled"], "ormah")
    root = home() / "plugins/ormah"
    source = files("ormah.integrations").joinpath("assets/hermes/plugin.py").read_text()
    for key, value in (("__ORMAH_PYTHON__", sys.executable), ("__ORMAH_WORKSPACE__", str(project) if project else None),
                       ("__ORMAH_PROFILE__", str(home())), ("__ORMAH_ENV_KEYS__", names)):
        source = source.replace(key, repr(value))
    install.file(root / "__init__.py", source)
    install.file(root / "plugin.yaml", 'name: ormah\nversion: 1.0.0\ndescription: Automatic Ormah memory context\n'
                 'requires_hermes: ">=0.21.4"\npython_runtime: external\nprovides_hooks:\n  - pre_llm_call\n')
    install.file(root / "instructions.md", common.instructions())
    install.commit()


def disconnect(project: Path | None = None) -> None:
    preserved = installation().disconnect()
    if preserved:
        print("Preserved user-edited Ormah artifacts: " + ", ".join(preserved), file=sys.stderr)


def status(project: Path | None = None) -> dict:
    try:
        install = installation()
        tools, intact = install.intact("value"), install.intact()
        path = home() / "config.yaml"
        disabled = yaml_config.get(path.read_text(), ["plugins", "disabled"])[1] if path.exists() else []
        intact = intact and "ormah" not in (disabled or [])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        tools = intact = False
    intact = intact and os.environ.get("HERMES_SAFE_MODE", "").lower() not in ("1", "true", "yes")
    return {"tools": "mcp" if tools else "unconfigured", "whisper": "native_extension" if intact else "unconfigured",
            "maintenance": "deliberate" if tools else "unconfigured", "verified_live": False,
            "detail": "Hermes pre_llm_call whisper + MCP for the selected HERMES_HOME profile; reload Hermes. "
                      "Optional --project binds the entire profile's default memory space."}


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(id=HOST, name="Hermes Agent", detect_fn=detected,
                           is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
                           unwire_fn=disconnect, capabilities_fn=status)
