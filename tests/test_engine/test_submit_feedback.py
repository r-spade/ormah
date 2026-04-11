"""Tests for engine.submit_feedback and POST /agent/feedback route."""

from __future__ import annotations

import pytest

from ormah.models.node import CreateNodeRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_whisper_log(
    conn,
    node_id: str,
    session_id: str = "sess-abc",
    space: str = "myspace",
) -> None:
    conn.execute(
        "INSERT INTO whisper_log "
        "(node_id, score, session_id, space, prompt_text, prompt_vec, prompt_hash, was_injected, logged_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (node_id, 0.48, session_id, space, "how does auth work", b"", "hash-abc", 0),
    )
    conn.commit()


def _insert_review_log(conn, node_id: str, session_id: str = "sess-abc") -> int:
    cursor = conn.execute(
        "INSERT INTO review_log (node_id, session_id, surfaced_at, answered) "
        "VALUES (?, ?, datetime('now'), 0)",
        (node_id, session_id),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# TestSubmitFeedbackBasic
# ---------------------------------------------------------------------------


class TestSubmitFeedbackBasic:

    def test_no_whisper_log_returns_error(self, engine):
        result = engine.submit_feedback("nonexistent-id", 1)
        assert "No whisper_log entry" in result

    def test_explicit_feedback_inserts_affinity(self, engine):
        node_id = "node-explicit-001"
        _insert_whisper_log(engine.db.conn, node_id)
        engine.submit_feedback(node_id, 1, "explicit")

        row = engine.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["signal"] == 1
        assert row["source"] == "explicit"

    def test_implicit_feedback_inserts_affinity(self, engine):
        node_id = "node-implicit-001"
        _insert_whisper_log(engine.db.conn, node_id)
        engine.submit_feedback(node_id, 1, "implicit")

        row = engine.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["source"] == "implicit"

    def test_short_id_feedback_resolves_full_whisper_log_node_id(self, engine):
        node_id = "72a9ea26-1111-2222-3333-444444444444"
        _insert_whisper_log(engine.db.conn, node_id)

        engine.submit_feedback("72a9ea26", 1, "implicit")

        row = engine.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["source"] == "implicit"

    def test_short_id_feedback_returns_ambiguity_error_for_multiple_matches(self, engine):
        _insert_whisper_log(engine.db.conn, "72a9ea26-1111-2222-3333-444444444444")
        _insert_whisper_log(engine.db.conn, "72a9ea26-aaaa-bbbb-cccc-555555555555")

        result = engine.submit_feedback("72a9ea26", 1, "implicit")

        assert "Ambiguous node ID prefix 72a9ea26" in result
        rows = engine.db.conn.execute("SELECT * FROM affinity").fetchall()
        assert rows == []

    def test_idempotent_same_session(self, engine):
        node_id = "node-idempotent-001"
        _insert_whisper_log(engine.db.conn, node_id, session_id="sess-1")
        engine.submit_feedback(node_id, 1, "explicit")
        engine.submit_feedback(node_id, 1, "explicit")

        rows = engine.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchall()
        assert len(rows) == 1

    def test_explicit_updates_review_log(self, engine):
        node_id = "node-review-001"
        _insert_whisper_log(engine.db.conn, node_id)
        review_id = _insert_review_log(engine.db.conn, node_id)

        engine.submit_feedback(node_id, 1, "explicit")

        row = engine.db.conn.execute(
            "SELECT answered FROM review_log WHERE id = ?", (review_id,)
        ).fetchone()
        assert row["answered"] == 1

    def test_implicit_does_not_update_review_log(self, engine):
        node_id = "node-review-implicit-001"
        _insert_whisper_log(engine.db.conn, node_id)
        review_id = _insert_review_log(engine.db.conn, node_id)

        engine.submit_feedback(node_id, 1, "implicit")

        row = engine.db.conn.execute(
            "SELECT answered FROM review_log WHERE id = ?", (review_id,)
        ).fetchone()
        assert row["answered"] == 0

    def test_returns_success_message(self, engine):
        node_id = "node-success-001"
        _insert_whisper_log(engine.db.conn, node_id)
        result = engine.submit_feedback(node_id, 1, "explicit")
        assert "Feedback recorded" in result

    def test_recall_search_logs_memory_for_feedback(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="FastAPI is a web framework.",
            type="fact",
            title="FastAPI",
        ))

        text = engine.recall_search("FastAPI", session_id="recall-search-session")

        assert "FastAPI" in text
        row = engine.db.conn.execute(
            "SELECT session_id, prompt_text, was_injected FROM whisper_log WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert row is not None
        assert row["session_id"] == "recall-search-session"
        assert row["prompt_text"] == "FastAPI"
        assert row["was_injected"] == 1

        engine.submit_feedback(node_id, 1, "implicit")
        affinity = engine.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert affinity is not None

    def test_recall_node_logs_memory_for_feedback(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="SQLite is used for the graph index.",
            type="fact",
            title="SQLite choice",
        ))

        text = engine.recall_node(node_id, session_id="recall-node-session")

        assert "SQLite choice" in text
        row = engine.db.conn.execute(
            "SELECT session_id, prompt_text, was_injected FROM whisper_log WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert row is not None
        assert row["session_id"] == "recall-node-session"
        assert row["prompt_text"] == f"recall_node:{node_id}"
        assert row["was_injected"] == 1

        engine.submit_feedback(node_id, 1, "implicit")
        affinity = engine.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert affinity is not None


# ---------------------------------------------------------------------------
# TestSubmitFeedbackRoute
# ---------------------------------------------------------------------------


class TestSubmitFeedbackRoute:

    @pytest.fixture
    def client(self, tmp_memory_dir):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from ormah.api.routes_agent import router as agent_router
        from ormah.config import Settings
        from ormah.engine.memory_engine import MemoryEngine

        settings = Settings(memory_dir=tmp_memory_dir)
        eng = MemoryEngine(settings)
        eng.startup()

        test_app = FastAPI()
        test_app.include_router(agent_router)
        test_app.state.engine = eng

        with TestClient(test_app) as c:
            yield c, eng

        eng.shutdown()

    def test_route_no_whisper_log(self, client):
        c, eng = client
        resp = c.post("/agent/feedback", json={"node_id": "nonexistent", "signal": 1})
        assert resp.status_code == 200
        assert "No whisper_log entry" in resp.json()["text"]

    def test_route_explicit_feedback(self, client):
        c, eng = client
        node_id = "route-node-001"
        _insert_whisper_log(eng.db.conn, node_id)

        resp = c.post(
            "/agent/feedback",
            json={"node_id": node_id, "signal": 1, "source": "explicit"},
        )
        assert resp.status_code == 200
        assert "Feedback recorded" in resp.json()["text"]

        row = eng.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["signal"] == 1

    def test_route_short_id_feedback(self, client):
        c, eng = client
        node_id = "72a9ea26-1111-2222-3333-444444444444"
        _insert_whisper_log(eng.db.conn, node_id)

        resp = c.post(
            "/agent/feedback",
            json={"node_id": "72a9ea26", "signal": 1, "source": "implicit"},
        )
        assert resp.status_code == 200
        assert "Feedback recorded" in resp.json()["text"]

        row = eng.db.conn.execute(
            "SELECT * FROM affinity WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["signal"] == 1

    def test_route_feedback_after_recall_search(self, client):
        c, eng = client
        node_id, _ = eng.remember(CreateNodeRequest(
            content="FastAPI is a web framework.",
            type="fact",
            title="FastAPI",
        ))

        recall_resp = c.post(
            "/agent/recall",
            json={"query": "FastAPI", "session_id": "route-recall-session"},
        )
        assert recall_resp.status_code == 200
        assert "FastAPI" in recall_resp.json()["text"]

        feedback_resp = c.post(
            "/agent/feedback",
            json={"node_id": node_id, "signal": 1, "source": "implicit"},
        )
        assert feedback_resp.status_code == 200
        assert "Feedback recorded" in feedback_resp.json()["text"]

        row = eng.db.conn.execute(
            "SELECT session_id, prompt_text FROM affinity WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert row is not None
        assert row["session_id"] == "route-recall-session"
        assert row["prompt_text"] == "FastAPI"

    def test_route_feedback_after_recall_node(self, client):
        c, eng = client
        node_id, _ = eng.remember(CreateNodeRequest(
            content="SQLite is used for the graph index.",
            type="fact",
            title="SQLite choice",
        ))

        recall_resp = c.get(
            f"/agent/recall/{node_id}",
            params={"session_id": "route-recall-node-session"},
        )
        assert recall_resp.status_code == 200
        assert "SQLite choice" in recall_resp.json()["text"]

        feedback_resp = c.post(
            "/agent/feedback",
            json={"node_id": node_id, "signal": 1, "source": "implicit"},
        )
        assert feedback_resp.status_code == 200
        assert "Feedback recorded" in feedback_resp.json()["text"]

        row = eng.db.conn.execute(
            "SELECT session_id, prompt_text FROM affinity WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert row is not None
        assert row["session_id"] == "route-recall-node-session"
        assert row["prompt_text"] == f"recall_node:{node_id}"


# ---------------------------------------------------------------------------
# TestGateTuning
# ---------------------------------------------------------------------------


class TestGateTuning:

    def test_get_gate_returns_config_default_when_no_db_row(self, engine):
        gate = engine.get_gate()
        assert gate == engine.settings.whisper_injection_gate

    def test_positive_signal_raises_gate(self, engine):
        node_id = "node-gate-pos-001"
        _insert_whisper_log(engine.db.conn, node_id)
        initial = engine.get_gate()
        engine.submit_feedback(node_id, 1)
        assert engine.get_gate() > initial

    def test_negative_signal_lowers_gate(self, engine):
        node_id = "node-gate-neg-001"
        _insert_whisper_log(engine.db.conn, node_id)
        initial = engine.get_gate()
        engine.submit_feedback(node_id, -1)
        assert engine.get_gate() < initial

    def test_gate_persists_in_db(self, engine):
        node_id = "node-gate-persist-001"
        _insert_whisper_log(engine.db.conn, node_id)
        engine.submit_feedback(node_id, 1)
        row = engine.db.conn.execute(
            "SELECT value FROM gate_state WHERE key = 'injection_gate'"
        ).fetchone()
        assert row is not None
        assert row["value"] > engine.settings.whisper_injection_gate

    def test_gate_clamps_to_max(self, engine):
        engine.settings.gate_max = 0.52
        engine.settings.whisper_injection_gate = 0.50
        for _ in range(50):
            engine._update_gate(1)
        assert engine.get_gate() <= 0.52

    def test_gate_clamps_to_min(self, engine):
        engine.settings.gate_min = 0.48
        engine.settings.whisper_injection_gate = 0.50
        for _ in range(50):
            engine._update_gate(-1)
        assert engine.get_gate() >= 0.48

    def test_gate_tuning_disabled_skips_update(self, engine):
        engine.settings.gate_tuning_enabled = False
        node_id = "node-gate-disabled-001"
        _insert_whisper_log(engine.db.conn, node_id)
        engine.submit_feedback(node_id, 1)
        row = engine.db.conn.execute(
            "SELECT value FROM gate_state WHERE key = 'injection_gate'"
        ).fetchone()
        assert row is None
