"""Tests for proactive decay alerts."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from ormah.background.decay_manager import run_decay
from ormah.models.node import CreateNodeRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_node(conn, node_id, tier="working", stability=1.0, last_accessed=None):
    """Insert a minimal node directly into the DB with controllable stability."""
    if last_accessed is None:
        # Default: accessed long ago so R is very low
        last_accessed = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    conn.execute(
        "INSERT INTO nodes "
        "(id, type, tier, source, title, content, created, updated, "
        "last_accessed, access_count, confidence, importance, stability, file_path, file_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), "
        "?, 0, 1.0, 0.3, ?, ?, ?)",
        (node_id, "fact", tier, "test", f"title-{node_id}", "content",
         last_accessed, stability, f"/tmp/{node_id}.md", "abc"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# TestDecayAlertCreation
# ---------------------------------------------------------------------------


class TestDecayAlertCreation:

    def test_alert_created_for_working_node_below_threshold(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40

        node_id = "alert-working-001"
        _insert_node(engine.db.conn, node_id, tier="working", stability=1.0)

        run_decay(engine)

        row = engine.db.conn.execute(
            "SELECT * FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["tier"] == "working"
        assert row["retrievability"] < 0.40
        assert row["acknowledged"] == 0

    def test_alert_created_for_core_node_below_threshold(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40

        node_id = "alert-core-001"
        _insert_node(engine.db.conn, node_id, tier="core", stability=1.0)

        run_decay(engine)

        row = engine.db.conn.execute(
            "SELECT * FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row["tier"] == "core"

    def test_no_alert_for_healthy_node(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40

        node_id = "alert-healthy-001"
        # Accessed just now → R ≈ 1.0
        recent = datetime.now(timezone.utc).isoformat()
        _insert_node(engine.db.conn, node_id, tier="working", stability=30.0,
                     last_accessed=recent)

        run_decay(engine)

        row = engine.db.conn.execute(
            "SELECT * FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is None

    def test_alert_not_refired_within_refire_window(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40
        engine.settings.decay_alert_refire_days = 7

        node_id = "alert-refire-001"
        _insert_node(engine.db.conn, node_id, tier="working", stability=1.0)

        run_decay(engine)
        run_decay(engine)

        rows = engine.db.conn.execute(
            "SELECT * FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchall()
        assert len(rows) == 1  # not duplicated

    def test_alert_disabled_creates_no_rows(self, engine):
        engine.settings.decay_alert_enabled = False

        node_id = "alert-disabled-001"
        _insert_node(engine.db.conn, node_id, tier="working", stability=1.0)

        run_decay(engine)

        row = engine.db.conn.execute(
            "SELECT * FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# TestDecayAlertAcknowledgement
# ---------------------------------------------------------------------------


class TestDecayAlertAcknowledgement:

    def test_recall_node_acknowledges_alert(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40

        node_id = "alert-ack-001"
        _insert_node(engine.db.conn, node_id, tier="working", stability=1.0)
        run_decay(engine)

        # Verify alert exists and is unacknowledged
        row = engine.db.conn.execute(
            "SELECT acknowledged FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row["acknowledged"] == 0

        # recall_node should acknowledge it
        engine.recall_node(node_id, session_id="test-session")

        row = engine.db.conn.execute(
            "SELECT acknowledged FROM decay_alert_log WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row["acknowledged"] == 1


# ---------------------------------------------------------------------------
# TestDecayAlertWhisper
# ---------------------------------------------------------------------------


class TestDecayAlertWhisper:

    def test_alert_appears_in_first_message_whisper(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40

        node_id = "alert-whisper-001"
        _insert_node(engine.db.conn, node_id, tier="working", stability=1.0)
        run_decay(engine)

        # First message of session: recent_prompts=None triggers the alert block
        result = engine.get_whisper_context(
            prompt="what did we decide about auth",
            session_id="test-whisper-session",
        )
        assert "memory fading" in result.lower() or "fading" in result.lower() or node_id[:8] in result

    def test_alert_not_surfaced_mid_session(self, engine):
        engine.settings.decay_alert_enabled = True
        engine.settings.decay_alert_threshold = 0.40

        node_id = "alert-midsession-001"
        _insert_node(engine.db.conn, node_id, tier="working", stability=1.0)
        run_decay(engine)

        # Mid-session: pass a non-None recent_prompts list → no alert block
        result = engine.get_whisper_context(
            prompt="what did we decide about auth",
            recent_prompts=["previous prompt"],
            session_id="test-mid-session",
        )
        assert "memory fading" not in result.lower()
