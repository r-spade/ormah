"""Tests for eval/whisper/seeder.py."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def tmp_engine(tmp_path):
    from eval.whisper.cli import _EVAL_SETTINGS_OVERRIDES
    from ormah.config import Settings
    from ormah.engine.memory_engine import MemoryEngine

    (tmp_path / "nodes").mkdir()
    settings = Settings(memory_dir=tmp_path, **_EVAL_SETTINGS_OVERRIDES)
    engine = MemoryEngine(settings)
    engine.startup()
    yield engine
    engine.shutdown()


_CASE = {
    "id": "t-001",
    "memories": [
        {
            "node_id": "aaa-portfact",
            "title": "Port fact",
            "content": "Server runs on port 8787.",
            "type": "fact",
            "tier": "working",
            "space": "ormah",
        },
        {
            "node_id": "bbb-userpref",
            "title": "User preference",
            "content": "User prefers dark themes.",
            "type": "preference",
            "tier": "core",
            "space": None,
        },
    ],
}


class TestSeedCase:
    def test_creates_node_files(self, tmp_engine):
        from eval.whisper.seeder import seed_case
        seed_case(tmp_engine, _CASE)
        nodes_dir = tmp_engine.file_store.nodes_dir
        files = list(nodes_dir.glob("*.md"))
        assert len(files) == 2

    def test_preserves_node_ids(self, tmp_engine):
        from eval.whisper.seeder import seed_case
        seed_case(tmp_engine, _CASE)
        node = tmp_engine.file_store.load("aaa-portfact")
        assert node is not None
        assert node.title == "Port fact"

    def test_clear_removes_previous_nodes(self, tmp_engine):
        from eval.whisper.seeder import seed_case, clear_eval_db
        seed_case(tmp_engine, _CASE)
        clear_eval_db(tmp_engine)
        nodes_dir = tmp_engine.file_store.nodes_dir
        files = list(nodes_dir.glob("*.md"))
        assert len(files) == 0

    def test_seed_replaces_prior_case(self, tmp_engine):
        from eval.whisper.seeder import seed_case
        seed_case(tmp_engine, _CASE)
        new_case = {
            "id": "t-002",
            "memories": [
                {"node_id": "ccc-newnode", "title": "New", "content": "New content.", "type": "fact", "tier": "working"},
            ],
        }
        seed_case(tmp_engine, new_case)
        nodes_dir = tmp_engine.file_store.nodes_dir
        files = list(nodes_dir.glob("*.md"))
        assert len(files) == 1
        assert tmp_engine.file_store.load("aaa-portfact") is None
        assert tmp_engine.file_store.load("ccc-newnode") is not None

    def test_seeds_relative_datetime_fields(self, tmp_engine):
        from eval.whisper.seeder import seed_case

        case = {
            "id": "dated",
            "memories": [
                {
                    "node_id": "dated-node",
                    "title": "Recent work",
                    "content": "Implemented eval pinning.",
                    "type": "fact",
                    "tier": "working",
                    "created_days_ago": 2,
                    "updated_hours_ago": 6,
                    "last_accessed_hours_ago": 3,
                    "valid_until": "2099-01-01T00:00:00Z",
                }
            ],
        }

        before = datetime.now(timezone.utc)
        seed_case(tmp_engine, case)
        after = datetime.now(timezone.utc)

        node = tmp_engine.file_store.load("dated-node")
        assert node is not None
        assert before - timedelta(days=3) <= node.created <= after - timedelta(days=1)
        assert before - timedelta(hours=7) <= node.updated <= after - timedelta(hours=5)
        assert before - timedelta(hours=4) <= node.last_accessed <= after - timedelta(hours=2)
        assert node.valid_until == datetime(2099, 1, 1, tzinfo=timezone.utc)

    def test_seeds_connections(self, tmp_engine):
        from eval.whisper.seeder import seed_case
        from ormah.models.node import EdgeType

        case = {
            "id": "connections",
            "memories": [
                {
                    "node_id": "a",
                    "title": "Node A",
                    "content": "A",
                    "type": "fact",
                    "tier": "working",
                    "connections": [{"target": "b", "edge": "supports", "weight": 0.9}],
                },
                {
                    "node_id": "b",
                    "title": "Node B",
                    "content": "B",
                    "type": "fact",
                    "tier": "working",
                },
            ],
        }

        seed_case(tmp_engine, case)

        node = tmp_engine.file_store.load("a")
        assert node is not None
        assert len(node.connections) == 1
        assert node.connections[0].target == "b"
        assert node.connections[0].edge == EdgeType.supports
        assert node.connections[0].weight == pytest.approx(0.9)

    def test_clear_eval_db_removes_scoring_state(self, tmp_engine):
        from eval.whisper.seeder import clear_eval_db, seed_case

        seed_case(tmp_engine, _CASE)

        with tmp_engine.db.transaction() as conn:
            retrieval_event_id = tmp_engine.db.insert_retrieval_event(
                conn,
                surface="whisper",
                session_id="sess",
                space="ormah",
                prompt_hash="hash",
                prompt_text="prompt",
                prompt_vec=b"1234",
                logged_at="2026-01-01T00:00:00Z",
            )
            conn.execute(
                "INSERT INTO whisper_log "
                "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, score, "
                "was_injected, logged_at, retrieval_event_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "sess",
                    "ormah",
                    "hash",
                    "prompt",
                    b"1234",
                    "aaa-portfact",
                    0.8,
                    1,
                    "2026-01-01T00:00:00Z",
                    retrieval_event_id,
                ),
            )
            conn.execute(
                "INSERT INTO whisper_decisions "
                "(session_id, space, prompt_hash, outcome, candidate_count, "
                "injected_count, max_gate_score, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "sess",
                    "ormah",
                    "hash",
                    "injected",
                    1,
                    1,
                    0.8,
                    "2026-01-01T00:00:00Z",
                ),
            )
            conn.execute(
                "INSERT INTO affinity "
                "(prompt_vec, prompt_text, node_id, signal, source, confirmed_at, space, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (b"1234", "prompt", "aaa-portfact", 1, "implicit", "2026-01-01T00:00:00Z", "ormah", "sess"),
            )
            conn.execute(
                "INSERT INTO review_log (node_id, session_id, surfaced_at, answered) "
                "VALUES (?, ?, ?, ?)",
                ("aaa-portfact", "sess", "2026-01-01T00:00:00Z", 0),
            )
            conn.execute(
                "INSERT INTO audit_log (operation, node_id, node_snapshot, detail, performed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("update", "aaa-portfact", "{}", "detail", "2026-01-01T00:00:00Z"),
            )
            conn.execute(
                "INSERT INTO auto_link_checked (node_a, node_b, result, checked_at) "
                "VALUES (?, ?, ?, ?)",
                ("aaa-portfact", "bbb-userpref", "none", "2026-01-01T00:00:00Z"),
            )

        clear_eval_db(tmp_engine)

        for table in (
            "whisper_log",
            "retrieval_events",
            "whisper_decisions",
            "affinity",
            "review_log",
            "audit_log",
            "auto_link_checked",
        ):
            row = tmp_engine.db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            assert row["n"] == 0

    def test_clear_eval_db_can_preserve_self_node(self, tmp_engine):
        from eval.whisper.seeder import clear_eval_db, seed_case

        seed_case(tmp_engine, _CASE)
        clear_eval_db(tmp_engine, preserve_self=True)

        row = tmp_engine.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'user_node_id'"
        ).fetchone()
        assert row is not None
        assert tmp_engine.file_store.load(row["value"]) is not None
