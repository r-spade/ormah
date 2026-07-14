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


@pytest.mark.asyncio
async def test_dispatch_uses_extended_timeout_for_maintenance(monkeypatch):
    captured: dict[str, float] = {}

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout):
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, path, json):
            assert path == "/agent/maintenance"
            resp = MagicMock()
            resp.is_success = True
            resp.text = "{}"
            resp.json.return_value = {
                "status": "awaiting_results",
                "job_id": "job-1",
                "batches": {
                    "summary": "nothing to process",
                    "link_candidates": [],
                    "conflict_candidates": [],
                    "merge_candidates": [],
                    "consolidation_clusters": [],
                },
            }
            return resp

    monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", _FakeAsyncClient)

    await mcp_adapter._dispatch("http://localhost:8787", "run_maintenance", {})

    assert captured["base_url"] == "http://localhost:8787"
    assert captured["timeout"] == mcp_adapter._MAINTENANCE_TIMEOUT_SECONDS


def test_format_timeout_error_for_maintenance_is_explicit():
    message = mcp_adapter._format_timeout_error("run_maintenance")
    assert "timed out after 300s" in message
    assert "run_maintenance" in message


@pytest.mark.asyncio
async def test_dispatch_submit_feedback_includes_whisper_log_id(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponse:
        is_success = True
        text = "{}"
        request = MagicMock()

        def json(self):
            return {"text": "Feedback recorded"}

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout):
            captured["base_url"] = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", _FakeAsyncClient)

    text = await mcp_adapter._dispatch(
        "http://localhost:8787",
        "submit_feedback",
        {
            "node_id": "node-1",
            "signal": 1,
            "source": "implicit",
            "whisper_log_id": 123,
        },
    )

    assert text == "Feedback recorded"
    assert captured["path"] == "/agent/feedback"
    assert captured["json"] == {
        "node_id": "node-1",
        "signal": 1,
        "source": "implicit",
        "whisper_log_id": 123,
    }


@pytest.mark.asyncio
async def test_dispatch_polls_until_phase1_batches_are_ready(monkeypatch):
    responses = [
        {"status": "running_phase1", "job_id": "job-1"},
        {
            "status": "awaiting_results",
            "job_id": "job-1",
            "batches": {
                "summary": "nothing to process",
                "link_candidates": [],
                "conflict_candidates": [],
                "merge_candidates": [],
                "consolidation_clusters": [],
            },
        },
    ]

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.is_success = True
            self.status_code = 200
            self.text = "{}"
            self.request = MagicMock()

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout):
            self.base_url = base_url
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, path, json):
            assert path == "/agent/maintenance"
            return _FakeResponse(responses.pop(0))

        async def get(self, path, params=None):
            assert path == "/agent/maintenance"
            return _FakeResponse(responses.pop(0))

    async def _no_sleep():
        return None

    monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(mcp_adapter, "_sleep_for_poll_interval", _no_sleep)

    text = await mcp_adapter._dispatch("http://localhost:8787", "run_maintenance", {}, session_id="s1")

    assert "Maintenance batches ready: nothing to process" in text


@pytest.mark.asyncio
async def test_dispatch_polls_until_phase2_apply_completes(monkeypatch):
    responses = [
        {"status": "running_phase2", "job_id": "job-1"},
        {"status": "completed", "job_id": "job-1", "apply_summary": {"edges": 1}},
    ]

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.is_success = True
            self.status_code = 200
            self.text = "{}"
            self.request = MagicMock()

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout):
            self.base_url = base_url
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, path, json):
            assert path == "/agent/maintenance"
            assert json["job_id"] == "job-1"
            return _FakeResponse(responses.pop(0))

        async def get(self, path, params=None):
            assert path == "/agent/maintenance"
            return _FakeResponse(responses.pop(0))

    async def _no_sleep():
        return None

    monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(mcp_adapter, "_sleep_for_poll_interval", _no_sleep)
    mcp_adapter._MAINTENANCE_JOB_IDS["s1"] = "job-1"

    text = await mcp_adapter._dispatch(
        "http://localhost:8787",
        "run_maintenance",
        {"results": {"edges": []}},
        session_id="s1",
    )

    assert '"status": "applied"' in text
    assert '"edges": 1' in text
