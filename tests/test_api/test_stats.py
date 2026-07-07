"""Tests for the canonical /stats endpoint."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api.routes_admin import router as admin_router
from ormah.api.routes_agent import router as agent_router
from ormah.api.routes_stats import router as stats_router
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine


@pytest.fixture
def stats_setup(tmp_memory_dir):
    settings = Settings(memory_dir=tmp_memory_dir, backup_dir=tmp_memory_dir.parent / "backups")
    engine = MemoryEngine(settings)
    engine.startup()

    test_app = FastAPI()
    test_app.include_router(agent_router)
    test_app.include_router(admin_router)
    test_app.include_router(stats_router)
    test_app.state.engine = engine

    with TestClient(test_app) as c:
        yield c, engine

    engine.shutdown()


def _log_whisper(engine, *, node_id, was_injected, logged_at, session_id="s1", prompt="hi"):
    """Insert a synthetic whisper_log row mirroring context_builder's writer."""
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO whisper_log "
            "(session_id, space, prompt_hash, prompt_text, prompt_vec, "
            "node_id, score, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, None, f"hash-{prompt}", prompt, b"\x00",
             node_id, 0.9, was_injected, logged_at),
        )


def _remember(client, content="A fact about the user."):
    resp = client.post("/agent/remember", json={"content": content, "type": "fact"})
    assert resp.status_code == 200
    return resp.json()["node_id"]


def test_stats_empty(stats_setup):
    client, _ = stats_setup
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["usage"]["whispers_used_total"] == 0
    assert data["usage"]["whispers_used_this_week"] == 0
    assert data["usage"]["memories_total"] == 0
    assert data["usage"]["memories_this_week"] == 0
    assert data["store"]["total_nodes"] >= 1
    assert "feedback_health" in data["whisper"]
    # window.days is days elapsed in the current calendar week (1-7), not a
    # fixed rolling-window size, since the default is now a fixed Mon-Sun week.
    assert 1 <= data["window"]["days"] <= 7
    assert "generated_at" in data


def test_old_stats_routes_removed(stats_setup):
    client, _ = stats_setup
    assert client.get("/agent/stats").status_code == 404
    assert client.get("/admin/stats").status_code == 404


def test_memories_counted(stats_setup):
    client, _ = stats_setup
    _remember(client, "Fact one.")
    _remember(client, "Fact two.")
    data = client.get("/stats").json()
    assert data["usage"]["memories_total"] == 2
    assert data["usage"]["memories_this_week"] == 2


def test_whispers_used_counts_distinct_calls(stats_setup):
    """One whisper call writes several candidate rows sharing logged_at.

    The counter must count the *call*, not each candidate row.
    """
    client, engine = stats_setup
    now = datetime.now(timezone.utc).isoformat()
    # Two candidates from a single whisper call, both injected.
    _log_whisper(engine, node_id="n1", was_injected=1, logged_at=now)
    _log_whisper(engine, node_id="n2", was_injected=1, logged_at=now)
    data = client.get("/stats").json()
    assert data["usage"]["whispers_used_total"] == 1
    assert data["usage"]["whispers_used_this_week"] == 1


def test_non_injected_whispers_excluded(stats_setup):
    """Candidates that were logged but not injected don't count as used."""
    client, engine = stats_setup
    now = datetime.now(timezone.utc).isoformat()
    _log_whisper(engine, node_id="n1", was_injected=0, logged_at=now, prompt="p1")
    data = client.get("/stats").json()
    assert data["usage"]["whispers_used_total"] == 0


def test_weekly_window_excludes_old(stats_setup):
    client, engine = stats_setup
    recent = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    _log_whisper(engine, node_id="n1", was_injected=1, logged_at=recent, prompt="recent")
    _log_whisper(engine, node_id="n2", was_injected=1, logged_at=old, prompt="old")
    data = client.get("/stats").json()
    assert data["usage"]["whispers_used_total"] == 2
    assert data["usage"]["whispers_used_this_week"] == 1


def test_clients_endpoint(stats_setup, monkeypatch):
    """GET /agent/clients returns the agent list with detection and wired status."""
    client, _ = stats_setup
    import ormah.setup as setup
    monkeypatch.setattr(
        setup, "list_agents",
        lambda: [
            {"id": "claude_code", "name": "Claude Code", "detected": True, "wired": False, "platform": None, "available_on_current_os": True},
            {"id": "codex", "name": "Codex CLI", "detected": False, "wired": False, "platform": None, "available_on_current_os": True},
        ],
    )
    resp = client.get("/agent/clients")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(a["id"] == "claude_code" and a["detected"] is True for a in data)
    assert any(a["id"] == "codex" and a["detected"] is False for a in data)


def test_custom_window(stats_setup):
    client, engine = stats_setup
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    _log_whisper(engine, node_id="n1", was_injected=1, logged_at=old, prompt="old")
    # default 7d window excludes it; 30d window includes it
    assert client.get("/stats").json()["usage"]["whispers_used_this_week"] == 0
    data = client.get("/stats", params={"days": 30}).json()
    assert data["usage"]["whispers_used_this_week"] == 1
    assert data["window"]["days"] == 30
