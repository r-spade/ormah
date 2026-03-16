"""OpenAI function-calling schema adapter."""

from __future__ import annotations

from ormah.adapters.tool_schemas import ALL_TOOLS


def get_openai_tools() -> list[dict]:
    """Convert canonical tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in ALL_TOOLS
    ]
