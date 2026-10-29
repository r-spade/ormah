"""OpenCode native pre-model message extension plus MCP tools."""
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from ormah.integrations import common, json_config
from ormah.integrations.ownership import Installation

HOST = "opencode"


def directory() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "opencode"


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("opencode") or directory().is_dir())



def _installation(project: Path | None) -> Installation:
    """Follow known config-key migrations without adopting edited values.

    Remap receipt locations in memory only. The normal ownership comparison still
    checks the original expected value, translated between enabled/disabled.
    Status is read-only; connect checkpoints remapped records with its writes.
    """
    install = Installation(common.receipt(HOST, project))
    if install.recovery:
        install._recover()
    records = []
    for original in install.records:
        record = original.copy()
        keys = record.get("keys")
        alternate = None
        if record["kind"] == "item" and keys in (["plugin"], ["plugins"]):
            alternate = ["plugins" if keys == ["plugin"] else "plugin"]
        elif record["kind"] == "value" and keys in (["mcp", "ormah"], ["mcp", "servers", "ormah"]):
            alternate = ["mcp", "servers", "ormah"] if keys == ["mcp", "ormah"] else ["mcp", "ormah"]
        path = Path(record["path"])
        if alternate and path.exists():
            text = path.read_text()
            found, current = json_config.get(text, keys)
            present = found and (record["kind"] != "item" or (
                isinstance(current, list) and any(json_config.values_equal(record["value"], v) for v in current)
            ))
            moved, _ = json_config.get(text, alternate)
            if not present and moved:
                record["keys"] = alternate
                if record["kind"] == "value":
                    value = record["value"].copy()
                    old, new = ("enabled", "disabled") if len(keys) == 2 else ("disabled", "enabled")
                    value[new] = not value.pop(old)
                    record["value"] = value
        records.append(record)
    install.records = records
    return install


def connect(project: Path | None = None) -> None:
    root = project if project else directory()
    config = root / "opencode.jsonc"
    if not config.exists() and (root / "opencode.json").exists():
        config = root / "opencode.json"
    assets = (project / ".opencode" if project else root) / "ormah"
    install = _installation(project)
    text = config.read_text() if config.exists() else "{}"
    native_plugins, _ = json_config.get(text, ["plugins"])
    native_mcp, _ = json_config.get(text, ["mcp", "servers"])
    for key in ("plugin", "plugins"):
        found, plugins = json_config.get(text, [key])
        if found and isinstance(plugins, list) and (assets / "plugin.mjs").as_uri() in plugins:
            raise ValueError("Legacy OpenCode plugin configured; disconnect Ormah, then connect again")
    if native_mcp and json_config.get(text, ["mcp", "ormah"])[0]:
        raise ValueError(f"Preserving existing configuration: {config}: mcp.ormah; disconnect first")
    # V2 normalizes V1 config; honor native keys when they already exist.
    mcp_keys = ["mcp", "servers", "ormah"] if native_mcp else ["mcp", "ormah"]
    # OpenCode launches local MCP with the session workspace as cwd.
    install.value(config, mcp_keys, {
        "type": "local", "command": common.mcp_command(HOST, str(project) if project else "."),
        **({"disabled": False} if native_mcp else {"enabled": True}),
    })
    source = files("ormah.integrations").joinpath("assets/opencode/plugin.mjs").read_text()
    plugin = assets / "plugin.mjs"
    guide = assets / "instructions.md"
    install.file(plugin, source.replace("__ORMAH_PYTHON__", json.dumps(sys.executable)))
    v2 = files("ormah.integrations").joinpath("assets/opencode/v2.mjs").read_text()
    install.file(assets / "v2.mjs", v2.replace("__ORMAH_PYTHON__", json.dumps(sys.executable)))
    # V1 loads main; V2 loads exports["."]. Neither needs an npm dependency.
    install.file(assets / "package.json", json.dumps({
        "name": "ormah-opencode", "type": "module", "main": "./plugin.mjs",
        "exports": {".": "./v2.mjs"},
    }, indent=2) + "\n")
    install.file(guide, common.instructions())
    install.item(config, ["plugins" if native_plugins else "plugin"], assets.as_uri())
    install.item(config, ["instructions"], str(guide))
    install.commit()


def disconnect(project: Path | None = None) -> None:
    preserved = _installation(project).disconnect()
    if preserved:
        print("Preserved user-edited Ormah artifacts: " + ", ".join(preserved), file=sys.stderr)


def status(project: Path | None = None) -> dict:
    try:
        install = _installation(project)
        tools, intact = install.intact("value"), install.intact()
    except (OSError, ValueError, KeyError, TypeError):
        tools = intact = False
    return {
        "tools": "mcp" if tools else "unconfigured",
        "whisper": "native_extension" if intact else "unconfigured",
        "maintenance": "deliberate" if tools else "unconfigured",
        "detail": "Supports OpenCode V1 and V2 with MCP tools and automatic memory recall. "
                  "Restart OpenCode after connecting; keep Ormah running.",
        "verified_live": False,
    }



def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="OpenCode", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp", wire_fn=connect,
        unwire_fn=disconnect, capabilities_fn=status,
    )
