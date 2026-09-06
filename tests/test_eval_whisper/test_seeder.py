"""Tests for eval/whisper/seeder.py."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def tmp_engine(tmp_path, monkeypatch):
    from eval.whisper.cli import _EVAL_SETTINGS_OVERRIDES
    from ormah.config import Settings
    from ormah.engine.memory_engine import MemoryEngine

    # Seeder coverage exercises SQLite/file reset behavior, not embedding or
    # reranking. Keep these tests independent of model downloads while using
    # the real MemoryEngine schema and its foreign-key constraints.
    monkeypatch.setattr(MemoryEngine, "_index_embedding", lambda self, node: None)
    monkeypatch.setattr(MemoryEngine, "_warmup_embedder", lambda self: None)
    monkeypatch.setattr(MemoryEngine, "_warmup_reranker", lambda self: None)

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


_REUSED_ID_CASE = {
    "id": "t-002",
    "memories": [
        {
            "node_id": "aaa-portfact",
            "title": "Replacement fact",
            "content": "The replacement fixture has different content.",
            "type": "fact",
            "tier": "working",
            "space": "ormah",
        },
    ],
}


_CASE_SCOPED_DIAGNOSTIC_TABLES = (
    "retrieval_events",
    "whisper_log",
    "affinity",
    "signals",
    "confirmed_use_claims",
    "whisper_decisions",
)


def _insert_case_diagnostics(engine, node_id: str) -> None:
    """Record linked feedback/diagnostics using the production SQLite schema."""
    with engine.db.transaction() as conn:
        event = conn.execute(
            "INSERT INTO retrieval_events "
            "(surface, session_id, space, prompt_hash, prompt_text, prompt_vec, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("whisper", "case-a", "ormah", "case-a-hash", "case A prompt", b"1234", "2026-01-01T00:00:00Z"),
        )
        whisper_log = conn.execute(
            "INSERT INTO whisper_log "
            "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, score, "
            "was_injected, logged_at, retrieval_event_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "case-a",
                "ormah",
                "case-a-hash",
                None,
                b"",
                node_id,
                0.8,
                1,
                "2026-01-01T00:00:00Z",
                event.lastrowid,
            ),
        )
        whisper_log_id = whisper_log.lastrowid
        conn.execute(
            "INSERT INTO affinity "
            "(prompt_vec, prompt_text, node_id, signal, source, confirmed_at, space, session_id, whisper_log_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                b"1234",
                "case A prompt",
                node_id,
                1,
                "implicit",
                "2026-01-01T00:00:00Z",
                "ormah",
                "case-a",
                whisper_log_id,
            ),
        )
        conn.execute(
            "INSERT INTO signals "
            "(whisper_log_id, node_id, signal_type, polarity, strength, source, session_id, "
            "surface, space, prompt_hash, evidence, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                whisper_log_id,
                node_id,
                "implicit_confirmation",
                1,
                1.0,
                "agent",
                "case-a",
                "whisper",
                "ormah",
                "case-a-hash",
                "case A evidence",
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO confirmed_use_claims (whisper_log_id, node_id, claimed_at) "
            "VALUES (?, ?, ?)",
            (whisper_log_id, node_id, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO whisper_decisions "
            "(session_id, space, prompt_hash, intent, outcome, candidate_count, injected_count, "
            "max_gate_score, logged_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "case-a",
                "ormah",
                "case-a-hash",
                "factual",
                "injected",
                1,
                1,
                0.8,
                "2026-01-01T00:00:00Z",
            ),
        )


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
            conn.execute(
                "INSERT INTO whisper_log "
                "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, score, was_injected, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("sess", "ormah", "hash", "prompt", b"1234", "aaa-portfact", 0.8, 1, "2026-01-01T00:00:00Z"),
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

        for table in ("whisper_log", "affinity", "review_log", "audit_log", "auto_link_checked"):
            row = tmp_engine.db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            assert row["n"] == 0

    def test_seed_case_clears_linked_diagnostics_before_same_id_fixture_reuse(self, tmp_engine):
        """A later fixture must not retain evidence tied to the old fixture's node ID."""
        from eval.whisper.seeder import seed_case

        seed_case(tmp_engine, _CASE)
        _insert_case_diagnostics(tmp_engine, "aaa-portfact")

        for table in _CASE_SCOPED_DIAGNOSTIC_TABLES:
            row = tmp_engine.db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            assert row["n"] == 1

        # Case B intentionally reuses A's node ID. A stale row would look valid
        # after the reset, so checking only for orphaned node IDs would miss it.
        seed_case(tmp_engine, _REUSED_ID_CASE)

        for table in _CASE_SCOPED_DIAGNOSTIC_TABLES:
            row = tmp_engine.db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            assert row["n"] == 0

        node = tmp_engine.file_store.load("aaa-portfact")
        assert node is not None
        assert node.content == "The replacement fixture has different content."
        assert tmp_engine.db.conn.execute("PRAGMA foreign_key_check").fetchall() == []

    @pytest.mark.parametrize("preserve_self", [False, True])
    def test_clear_eval_db_is_idempotent_without_diagnostics_and_preserves_metadata(
        self, tmp_engine, preserve_self
    ):
        from eval.whisper.seeder import clear_eval_db

        with tmp_engine.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('onboarding_prompted', '1')"
            )

        # The second invocation starts with no case nodes or diagnostics.
        clear_eval_db(tmp_engine, preserve_self=preserve_self)
        clear_eval_db(tmp_engine, preserve_self=preserve_self)

        row = tmp_engine.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'onboarding_prompted'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "1"
        assert tmp_engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == (
            1 if preserve_self else 0
        )
        for table in _CASE_SCOPED_DIAGNOSTIC_TABLES:
            assert tmp_engine.db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert tmp_engine.db.conn.execute("PRAGMA foreign_key_check").fetchall() == []

    def test_runner_results_are_unchanged_by_prior_diagnostics(self, tmp_engine, monkeypatch):
        from eval.whisper.runner import run_whisper_eval

        case = {
            **_REUSED_ID_CASE,
            "prompts": [
                {
                    "text": "What does this fixture contain?",
                    "category": "factual",
                    "expected": {"should_inject": ["aaa-portfact"]},
                }
            ],
        }
        monkeypatch.setattr(tmp_engine.context_builder, "_get_classifier", lambda: None)
        monkeypatch.setattr(
            tmp_engine,
            "get_whisper_context",
            lambda **kwargs: ("fixture context", ["aaa-portfact"]),
        )

        baseline = run_whisper_eval([case], tmp_engine)
        _insert_case_diagnostics(tmp_engine, "aaa-portfact")
        after_prior_diagnostics = run_whisper_eval([case], tmp_engine)

        assert after_prior_diagnostics == baseline
        for table in _CASE_SCOPED_DIAGNOSTIC_TABLES:
            assert tmp_engine.db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    def test_clear_eval_db_can_preserve_self_node(self, tmp_engine):
        from eval.whisper.seeder import clear_eval_db, seed_case

        seed_case(tmp_engine, _CASE)
        clear_eval_db(tmp_engine, preserve_self=True)

        row = tmp_engine.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'user_node_id'"
        ).fetchone()
        assert row is not None
        assert tmp_engine.file_store.load(row["value"]) is not None
