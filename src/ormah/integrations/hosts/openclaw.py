"""Native OpenClaw prompt hook plus MCP, bound to one configured workspace."""
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from ormah.integrations import common, json_config
from ormah.integrations.ownership import Installation

HOST = "openclaw"


def directory() -> Path:
    return Path(os.environ.get("OPENCLAW_STATE_DIR", Path.home() / ".openclaw")).expanduser()


def config_path() -> Path:
    return Path(os.environ.get("OPENCLAW_CONFIG_PATH", directory() / "openclaw.json")).expanduser()


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("openclaw") or config_path().is_file())


def connect(project: Path | None = None) -> None:
    config = config_path()
    data = json_config.parse(config.read_text() if config.exists() else "{}", json5=True).value
    if not isinstance(data, dict):
        raise ValueError("OpenClaw configuration must be an object")
    plugins = data.get("plugins", {})
    if plugins.get("enabled") is False or "ormah" in plugins.get("deny", []):
        raise ValueError("OpenClaw plugin policy disables Ormah; preserving that policy")
    configured = data.get("agents", {}).get("defaults", {}).get("workspace")
    workspace = (project or Path(configured or directory() / "workspace")).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError("Select an existing OpenClaw workspace with --project /absolute/path")
    install = Installation(common.receipt(HOST), json5=True)
    command = common.mcp_command(HOST, str(workspace))
    install.value(config, ["mcp", "servers", "ormah"], {
        "transport": "stdio", "command": command[0], "args": command[1:],
        "enabled": True,
        "env": {key: "${" + key + "}" for key in (
            "ORMAH_URL", "ORMAH_PORT", "ORMAH_AUTH_TOKEN", "ORMAH_SPACE",
        ) if os.environ.get(key)},
    })
    assets = directory() / "ormah-plugin"
    source = files("ormah.integrations").joinpath("assets/openclaw/index.mjs").read_text()
    source = source.replace("__ORMAH_PYTHON__", json.dumps(sys.executable))
    source = source.replace("__ORMAH_WORKSPACE__", json.dumps(str(workspace)))
    install.file(assets / "index.mjs", source)
    install.file(assets / "package.json", json.dumps({
        "name": "ormah-openclaw", "version": "0.0.0", "type": "module",
        "openclaw": {"extensions": ["./index.mjs"]},
    }, indent=2) + "\n")
    install.file(assets / "openclaw.plugin.json", json.dumps({
        "id": "ormah", "name": "Ormah", "categories": ["memory"],
        "activation": {"onStartup": True},
        "configSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    }, indent=2) + "\n")
    install.file(assets / "instructions.md", common.instructions())
    install.item(config, ["plugins", "load", "paths"], str(assets))
    install.value(config, ["plugins", "entries", "ormah"], {
        "enabled": True, "hooks": {"allowConversationAccess": True},
    })
    if "allow" in plugins:
        install.item(config, ["plugins", "allow"], "ormah")
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, json5=True)


def status(project: Path | None = None) -> dict:
    return common.capability(HOST, whisper="native_extension", json5=True, detail=(
        "OpenClaw native whisper + MCP bound to the selected workspace. "
        "Other workspaces do not receive whisper; review plugin trust and reload the Gateway."
    ))


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="OpenClaw", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
