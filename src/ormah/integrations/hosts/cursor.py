"""Cursor deliberate MCP support; prompt-time whisper is explicitly blocked."""
from pathlib import Path

from ormah.integrations import common
from ormah.integrations.ownership import Installation

HOST = "cursor"
DETAIL = (
    "MCP tools only. Cursor beforeSubmitPrompt cannot return model context; "
    "automatic prompt-time whisper is blocked. Use project setup for workspace scope."
)


def detected() -> bool:
    from ormah.setup import _find_binary
    return bool(_find_binary("cursor") or (Path.home() / ".cursor").is_dir()
                or Path("/Applications/Cursor.app").is_dir())


def connect(project: Path | None = None) -> None:
    directory = (project or Path.home()) / ".cursor"
    command = common.mcp_command(HOST, str(project) if project else None)
    install = Installation(common.receipt(HOST, project))
    install.value(directory / "mcp.json", ["mcpServers", "ormah"], {
        "command": command[0], "args": command[1:],
    })
    if project:
        install.file(directory / "rules" / "ormah.mdc",
                     "---\ndescription: Ormah memory guidance\nalwaysApply: true\n---\n\n"
                     + common.instructions())
    install.commit()


def disconnect(project: Path | None = None) -> None:
    common.disconnect(HOST, project)


def status(project: Path | None = None) -> dict:
    return common.capability(HOST, whisper="blocked", detail=DETAIL, project=project)


def descriptor():
    from ormah.setup import AgentDescriptor
    return AgentDescriptor(
        id=HOST, name="Cursor", detect_fn=detected,
        is_wired_fn=lambda: status()["tools"] == "mcp",
        wire_fn=connect, unwire_fn=disconnect,
        supports_maintenance=False, capabilities_fn=status,
    )
