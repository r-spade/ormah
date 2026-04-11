"""Tests for the recall-based auto-promotion background job."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ormah.background.recall_promoter import run_recall_promotion
from ormah.models.node import CreateNodeRequest, NodeType, Tier


def _get_tier(engine, node_id: str) -> str:
    row = engine.db.conn.execute(
        "SELECT tier FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return row["tier"] if row else None


def _add_affinity(engine, node_id: str, sessions: list[str], signal: int = 1) -> None:
    """Insert affinity rows for the given node across the listed session IDs."""
    now_iso = datetime.now(timezone.utc).isoformat()
    for session_id in sessions:
        engine.db.conn.execute(
            """
            INSERT INTO affinity
                (prompt_vec, prompt_text, node_id, signal, source, confirmed_at, space, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (node_id, session_id) DO NOTHING
            """,
            (b"", "test prompt", node_id, signal, "explicit", now_iso, None, session_id),
        )
    engine.db.conn.commit()


class TestRecallPromoterBasics:
    def test_working_node_promoted_after_enough_sessions(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Useful working memory",
            type=NodeType.fact,
            tier=Tier.working,
            title="Useful fact",
        ))
        _add_affinity(engine, node_id, ["s1", "s2", "s3"])

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "core"

    def test_working_node_not_promoted_with_too_few_sessions(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Rarely recalled memory",
            type=NodeType.fact,
            tier=Tier.working,
            title="Rare fact",
        ))
        _add_affinity(engine, node_id, ["s1", "s2"])  # only 2, default threshold is 3

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "working"

    def test_negative_signals_do_not_count(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Disliked memory",
            type=NodeType.fact,
            tier=Tier.working,
            title="Disliked",
        ))
        _add_affinity(engine, node_id, ["s1", "s2", "s3"], signal=-1)

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "working"

    def test_already_core_node_unaffected(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Already core memory",
            type=NodeType.fact,
            tier=Tier.core,
            title="Core fact",
        ))
        _add_affinity(engine, node_id, ["s1", "s2", "s3"])

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "core"

    def test_archival_node_not_promoted(self, engine):
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Archived memory",
            type=NodeType.fact,
            tier=Tier.archival,
            title="Archived fact",
        ))
        _add_affinity(engine, node_id, ["s1", "s2", "s3"])

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "archival"

    def test_no_nodes_to_promote_is_safe(self, engine):
        """Empty affinity table should not raise."""
        run_recall_promotion(engine)

    def test_duplicate_session_id_counts_once(self, engine):
        """Multiple affinity rows for the same session should count as one."""
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Memory with repeat sessions",
            type=NodeType.fact,
            tier=Tier.working,
            title="Repeat sessions",
        ))
        # Insert two rows for s1 manually via low-level to bypass UNIQUE constraint
        # by using different signals — but only one session distinct.
        now_iso = datetime.now(timezone.utc).isoformat()
        engine.db.conn.execute(
            """
            INSERT INTO affinity
                (prompt_vec, prompt_text, node_id, signal, source, confirmed_at, space, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (node_id, session_id) DO NOTHING
            """,
            (b"", "p1", node_id, 1, "explicit", now_iso, None, "s1"),
        )
        # s2 only — total 2 distinct sessions, below threshold of 3
        engine.db.conn.execute(
            """
            INSERT INTO affinity
                (prompt_vec, prompt_text, node_id, signal, source, confirmed_at, space, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (node_id, session_id) DO NOTHING
            """,
            (b"", "p2", node_id, 1, "explicit", now_iso, None, "s2"),
        )
        engine.db.conn.commit()

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "working"


class TestRecallPromoterConfig:
    def test_custom_session_threshold_respected(self, engine):
        engine.settings.recall_promotion_sessions = 2

        node_id, _ = engine.remember(CreateNodeRequest(
            content="Memory promoted at threshold 2",
            type=NodeType.fact,
            tier=Tier.working,
            title="Threshold 2",
        ))
        _add_affinity(engine, node_id, ["s1", "s2"])

        run_recall_promotion(engine)

        assert _get_tier(engine, node_id) == "core"

    def test_promotion_updates_db_tier(self, engine):
        """DB row should reflect the new tier after promotion."""
        node_id, _ = engine.remember(CreateNodeRequest(
            content="DB tier check",
            type=NodeType.fact,
            tier=Tier.working,
            title="DB check",
        ))
        _add_affinity(engine, node_id, ["s1", "s2", "s3"])

        run_recall_promotion(engine)

        row = engine.db.conn.execute(
            "SELECT tier FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        assert row["tier"] == "core"
