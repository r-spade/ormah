"""API regression coverage for retired retrospective review notes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api import routes_agent
from ormah.config import settings as global_settings


def test_whisper_route_preserves_context_across_first_gap_and_ongoing_turns(monkeypatch):
    """Every route shape leaves the builder free to return only current context.

    The builder regression seeds historical withheld/review rows and asserts
    that none of these session shapes can append a retrospective assignment.
    This route test pins the exact ``recent_prompts`` inputs for sessionless,
    first, post-gap, and ongoing requests.
    """
    routes_agent._session_buffers.clear()
    engine = MagicMock()
    engine.get_whisper_context.return_value = "# Ormah whispers\n\nCurrent task context"
    app = FastAPI()
    app.include_router(routes_agent.router)
    app.state.engine = engine

    monkeypatch.setattr(global_settings, "whisper_session_gap_minutes", 10)
    clock = iter([1000.0, 1001.0, 1602.0])
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
                    "prompt": "First task request with enough words.",
                    "space": "myspace",
                    "session_id": "review-gap",
                },
            )
            ongoing = client.post(
                "/agent/whisper",
                json={
                    "prompt": "Ongoing task request with enough words.",
                    "space": "myspace",
                    "session_id": "review-gap",
                },
            )
            after_gap = client.post(
                "/agent/whisper",
                json={
                    "prompt": "Post gap task request with enough words.",
                    "space": "myspace",
                    "session_id": "review-gap",
                },
            )

        expected = {"text": "# Ormah whispers\n\nCurrent task context", "node_id": None}
        assert sessionless.json() == expected
        assert first_turn.json() == expected
        assert ongoing.json() == expected
        assert after_gap.json() == expected
        assert [call.kwargs["recent_prompts"] for call in engine.get_whisper_context.call_args_list] == [
            None,
            None,
            ["First task request with enough words."],
            None,
        ]
        assert [call.kwargs["session_id"] for call in engine.get_whisper_context.call_args_list] == [
            "",
            "review-gap",
            "review-gap",
            "review-gap",
        ]
    finally:
        routes_agent._session_buffers.clear()
