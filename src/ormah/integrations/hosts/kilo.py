"""Kilo native pre-model message extension plus MCP tools."""
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "kilo"


def directory() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "kilo"


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("kilo") or directory().is_dir()
                or any((Path.home() / ".vscode/extensions").glob("kilocode.kilo-code-*")))


def connect(project: Path | None = None) -> None:
    if os.environ.get("KILO_PURE", "").lower() in ("1", "true"):
        raise ValueError("KILO_PURE disables external plugins; unset it before configuring automatic whisper")
    root = project if project else directory()
    config = root / "kilo.jsonc"
    if not config.exists() and (root / "kilo.json").exists():
        config = root / "kilo.json"
    assets = (project / ".kilo" if project else root) / "ormah"
    install = Installation(common.receipt(HOST, project))
    # Kilo explicitly launches local MCP with InstanceState.directory as cwd.
    install.value(config, ["mcp", "ormah"], {
        "type": "local", "command": common.mcp_command(HOST, str(project) if project else "."),
        "enabled": True,
    })
    source = files("ormah.integrations").joinpath("assets/kilo/plugin.mjs").read_text()
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
    result = common.capability(HOST, whisper="native_extension", project=project, detail=(
        "MCP tools and automatic chat.message whisper configured; reload Kilo. "
        "CLI/VS Code backend 7.7.9 contract checked; KILO_PURE disables whisper. No live model session validated."
    ))
    if os.environ.get("KILO_PURE", "").lower() in ("1", "true"):
        result["whisper"] = "unconfigured"
        result["detail"] = "KILO_PURE disables external plugins; automatic whisper is inactive."
    return result


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="Kilo", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
