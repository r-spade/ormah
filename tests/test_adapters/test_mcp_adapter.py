"""Tests for the MCP adapter."""

from __future__ import annotations

import asyncio
import json

import httpx

from ormah.adapters.mcp_adapter import _dispatch
from ormah.adapters.tool_schemas import TOOLS


def _mock_response(request: httpx.Request, data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response bound to *request*."""
    return httpx.Response(status_code=status_code, json=data, request=request)


def test_whisper_tool_schema_present():
    whisper = next((tool for tool in TOOLS if tool["name"] == "whisper"), None)
    assert whisper is not None
    assert whisper["parameters"]["required"] == ["prompt"]


def test_dispatch_whisper_posts_prompt_space_and_session(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/agent/whisper"
        body = json.loads(request.content.decode("utf-8"))
        assert body == {
            "prompt": "how does auth work?",
            "space": "ormah",
            "session_id": "sess-123",
        }
        return _mock_response(request, {"text": "whispered"})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("ormah.adapters.mcp_adapter.httpx.AsyncClient", fake_async_client)

    result = asyncio.run(
        _dispatch(
            "http://test",
            "whisper",
            {"prompt": "how does auth work?"},
            default_space="ormah",
            session_id="sess-123",
        )
    )

    assert result == "whispered"
