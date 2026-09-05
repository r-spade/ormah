"""API regressions for the review mechanism's silent first-turn inputs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api import routes_agent
from ormah.config import settings as global_settings


def test_whisper_route_marks_sessionless_first_and_gap_turns_as_first(monkeypatch):
    """All first-turn shapes pass ``recent_prompts=None`` to the builder.

    The review guard receives these exact route inputs.  Its context-builder
    regressions cover the eligible same-space history and assert that silence
    performs no review lookup or write.
    """
    routes_agent._session_buffers.clear()
    engine = MagicMock()
    engine.get_whisper_context.return_value = ""
    app = FastAPI()
    app.include_router(routes_agent.router)
    app.state.engine = engine

    monkeypatch.setattr(global_settings, "whisper_session_gap_minutes", 10)
    clock = iter([1000.0, 1601.0])
    monkeypatch.setattr(routes_agent, "time", SimpleNamespace(time=lambda: next(clock)))

    try:
        with TestClient(app) as client:
            sessionless = client.post(
                "/agent/whisper",
                json={"prompt": "Thanks, that helps.", "space": "myspace"},
            )
            first_turn = client.post(
                "/agent/whisper",
                json={
                    "prompt": "Thanks, that helps.",
                    "space": "myspace",
                    "session_id": "review-gap",
                },
            )
            after_gap = client.post(
                "/agent/whisper",
                json={
                    "prompt": "Thanks, that helps.",
                    "space": "myspace",
                    "session_id": "review-gap",
                },
            )

        assert sessionless.json() == {"text": "", "node_id": None}
        assert first_turn.json() == {"text": "", "node_id": None}
        assert after_gap.json() == {"text": "", "node_id": None}
        assert [call.kwargs["recent_prompts"] for call in engine.get_whisper_context.call_args_list] == [
            None,
            None,
            None,
        ]
        assert [call.kwargs["session_id"] for call in engine.get_whisper_context.call_args_list] == [
            "",
            "review-gap",
            "review-gap",
        ]
    finally:
        routes_agent._session_buffers.clear()
