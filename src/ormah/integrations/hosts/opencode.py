"""OpenCode native pre-model message extension plus MCP tools."""
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "opencode"


def directory() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "opencode"


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("opencode") or directory().is_dir())


def connect(project: Path | None = None) -> None:
    root = project if project else directory()
    config = root / "opencode.jsonc"
    if not config.exists() and (root / "opencode.json").exists():
        config = root / "opencode.json"
    assets = (project / ".opencode" if project else root) / "ormah"
    install = Installation(common.receipt(HOST, project))
    # OpenCode explicitly launches local MCP with InstanceState.directory as cwd.
    install.value(config, ["mcp", "ormah"], {
        "type": "local", "command": common.mcp_command(HOST, str(project) if project else "."),
        "enabled": True,
    })
    source = files("ormah.integrations").joinpath("assets/opencode/plugin.mjs").read_text()
    plugin = assets / "plugin.mjs"
    guide = assets / "instructions.md"
    install.file(plugin, source.replace("__ORMAH_PYTHON__", json.dumps(sys.executable)))
    install.file(guide, common.instructions())
    install.item(config, ["plugin"], plugin.as_uri())
    install.item(config, ["instructions"], str(guide))
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, project)


def status(project: Path | None = None) -> dict:
    return common.capability(HOST, whisper="native_extension", project=project, detail=(
        "MCP tools and automatic chat.message whisper configured; reload OpenCode. "
        "Contract checked against 1.18.32; no live model session validated."
    ))


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="OpenCode", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
