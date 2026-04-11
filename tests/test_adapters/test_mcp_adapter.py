from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ormah.adapters import mcp_adapter


class _FakeStdioServer:
    async def __aenter__(self):
        return ("read-stream", "write-stream")

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_run_mcp_stdio_generates_session_id_and_runs_server(monkeypatch):
    fake_uuid = "12345678-1234-5678-1234-567812345678"
    run = AsyncMock()
    server = MagicMock()
    server.run = run
    server.create_initialization_options.return_value = {"name": "ormah"}

    monkeypatch.setattr(mcp_adapter.uuid, "uuid4", lambda: fake_uuid)
    monkeypatch.setattr(mcp_adapter, "detect_space_from_cwd", lambda: "ormah")
    monkeypatch.setattr(mcp_adapter, "create_mcp_server", MagicMock(return_value=server))
    monkeypatch.setattr(mcp_adapter, "stdio_server", lambda: _FakeStdioServer())

    await mcp_adapter.run_mcp_stdio()

    mcp_adapter.create_mcp_server.assert_called_once_with(
        mcp_adapter._BASE_URL,
        default_space="ormah",
        session_id=fake_uuid,
    )
    run.assert_awaited_once_with("read-stream", "write-stream", {"name": "ormah"})


# ---------------------------------------------------------------------------
# Durable write buffer integration tests
# ---------------------------------------------------------------------------

import httpx
from ormah.store.write_buffer import WriteBuffer


class TestWriteBufferIntegration:
    """_dispatch remember — buffer-on-failure and drain-on-success behaviour."""

    @pytest.mark.asyncio
    async def test_remember_buffered_on_connect_error(self, tmp_path, monkeypatch):
        """ConnectError on remember → entry queued in buffer, not lost."""
        buf = WriteBuffer(tmp_path / "pending_writes.jsonl")
        monkeypatch.setattr(mcp_adapter, "_write_buffer", buf)

        async def _raise(*_, **__):
            raise httpx.ConnectError("down")

        # Patch AsyncClient.post to simulate server down
        monkeypatch.setattr(httpx.AsyncClient, "post", _raise)

        result = await mcp_adapter._dispatch(
            "http://localhost:9999",
            "remember",
            {"content": "must not be lost", "type": "fact", "tier": "working"},
        )

        assert "queued" in result.lower() or "buffer" in result.lower()
        assert not buf.is_empty()
        entries = buf.load()
        assert entries[0]["args"]["content"] == "must not be lost"

    @pytest.mark.asyncio
    async def test_buffered_writes_drained_on_next_success(self, tmp_path, monkeypatch):
        """Buffered entries are replayed before the next successful remember call."""
        buf = WriteBuffer(tmp_path / "pending_writes.jsonl")
        buf.append(args={"content": "buffered-one", "type": "fact", "tier": "working"}, params={})
        monkeypatch.setattr(mcp_adapter, "_write_buffer", buf)

        responses = iter([
            # First call: drain the buffered entry
            MagicMock(is_success=True, json=lambda: {"text": "ok-drained"}),
            # Second call: the new remember
            MagicMock(is_success=True, json=lambda: {"text": "ok-new"}),
        ])

        async def _fake_post(self_client, url, **kwargs):
            return next(responses)

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        result = await mcp_adapter._dispatch(
            "http://localhost:9999",
            "remember",
            {"content": "new-write", "type": "fact", "tier": "working"},
        )

        assert buf.is_empty()
        assert "Replayed" in result or "ok-new" in result

    @pytest.mark.asyncio
    async def test_no_drain_when_buffer_empty(self, tmp_path, monkeypatch):
        """No extra HTTP call is made when the buffer is empty."""
        buf = WriteBuffer(tmp_path / "pending_writes.jsonl")
        monkeypatch.setattr(mcp_adapter, "_write_buffer", buf)

        call_count = 0

        async def _fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(is_success=True, json=lambda: {"text": "stored"})

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        await mcp_adapter._dispatch(
            "http://localhost:9999",
            "remember",
            {"content": "single write", "type": "fact", "tier": "working"},
        )

        assert call_count == 1
