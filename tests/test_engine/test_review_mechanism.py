"""Tests for the review mechanism in build_whisper_context."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ormah.engine.context_builder import ContextBuilder, _find_review_candidate
from ormah.engine.maintenance_signal import MAINTENANCE_DUE_SIGNAL
from ormah.index.db import Database
from ormah.index.graph import GraphIndex


def _make_node_dict(node_id, title, tier="core", space=None, importance=0.5, **kwargs):
    return {
        "id": node_id,
        "type": "fact",
        "tier": tier,
        "title": title,
        "content": kwargs.get("content", f"Content about {title}"),
        "space": space,
        "importance": importance,
        "confidence": 1.0,
        "valid_until": None,
        "source": "agent:test",
        "access_count": 0,
        "last_accessed": "2026-01-01T00:00:00Z",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
    }


def _insert_node(conn, node):
    conn.execute(
        "INSERT INTO nodes (id, type, tier, source, space, title, content, "
        "created, updated, last_accessed, access_count, confidence, importance, "
        "file_path, file_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (node["id"], node["type"], node["tier"], node["source"],
         node["space"], node["title"], node["content"],
         node["created"], node["updated"], node["last_accessed"],
         node["access_count"], node["confidence"], node["importance"],
         "/fake/path", "abc123"),
    )


@pytest.fixture
def mock_graph(tmp_path):
    db = Database(tmp_path / "index.db")
    db.init_schema()
    graph = GraphIndex(db.conn)
    return graph


# ---------------------------------------------------------------------------
# TestReviewCandidateSQL
# ---------------------------------------------------------------------------

class TestReviewCandidateSQL:
    """Tests for SQL-level eligibility in _find_review_candidate."""

    def test_no_whisper_log_returns_none(self, mock_graph):
        """Empty DB returns None."""
        result = _find_review_candidate(mock_graph.conn, threshold=0.70)
        assert result is None

    def test_injected_candidate_excluded(self, mock_graph):
        """was_injected=1 row is excluded."""
        conn = mock_graph.conn
        node = _make_node_dict("node-1", "Auth token storage")
        _insert_node(conn, node)
        conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-1 day'))",
            ("node-1", 0.60, "sess-abc", "myspace", "how does auth work", "hash1", b"", 1),
        )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is None

    def test_old_candidate_excluded(self, mock_graph):
        """Candidates older than 7 days are excluded."""
        conn = mock_graph.conn
        node = _make_node_dict("node-2", "Old memory")
        _insert_node(conn, node)
        conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-8 days'))",
            ("node-2", 0.48, "sess-abc", "myspace", "how does auth work", "hash2", b"", 0),
        )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is None

    def test_gated_out_candidate_found(self, mock_graph):
        """was_injected=0 row within 7 days returns a candidate dict."""
        conn = mock_graph.conn
        node = _make_node_dict("node-3", "Session management")
        _insert_node(conn, node)
        conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-1 day'))",
            ("node-3", 0.48, "sess-abc", "myspace", "how does auth work", "hash3", b"", 0),
        )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is not None
        assert result["node_id"] == "node-3"
        assert result["title"] == "Session management"
        assert result["content"] == "Content about Session management"
        assert result["prompt_text"] == "how does auth work"

    def test_node_with_successful_injection_excluded(self, mock_graph):
        """Node with both was_injected=0 and was_injected=1 within 7 days is excluded."""
        conn = mock_graph.conn
        node = _make_node_dict("node-4", "Mixed signal node")
        _insert_node(conn, node)
        # Gated-out row
        conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-2 days'))",
            ("node-4", 0.48, "sess-aaa", "myspace", "some prompt", "hash4a", b"", 0),
        )
        # Successful injection row for the same node
        conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-1 day'))",
            ("node-4", 0.70, "sess-bbb", "myspace", "another prompt", "hash4b", b"", 1),
        )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is None


# ---------------------------------------------------------------------------
# TestReviewPythonFilter
# ---------------------------------------------------------------------------

class TestReviewPythonFilter:
    """Tests for the Python-side filtering in _find_review_candidate."""

    def _insert_whisper_row(self, conn, node_id, prompt_vec_bytes=b""):
        conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-1 day'))",
            (node_id, 0.48, "sess-abc", "myspace", "how does auth work", "hash-py", prompt_vec_bytes, 0),
        )

    def test_strong_affinity_signal_skips_candidate(self, mock_graph):
        """Candidate with affinity row whose prompt_vec cosine-matches >= 0.70 is skipped."""
        conn = mock_graph.conn
        node = _make_node_dict("node-5", "Auth system")
        _insert_node(conn, node)

        vec = np.random.rand(768).astype(np.float32)
        vec_bytes = vec.tobytes()

        self._insert_whisper_row(conn, "node-5", vec_bytes)
        # Insert an affinity row with the same vector (cosine sim = 1.0)
        conn.execute(
            "INSERT INTO affinity (node_id, prompt_vec, prompt_text, signal, confirmed_at, session_id) "
            "VALUES (?, ?, ?, ?, datetime('now'), ?)",
            ("node-5", vec_bytes, "how does auth work", 1, "sess-affinity"),
        )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is None

    def test_recently_surfaced_skips_candidate(self, mock_graph):
        """Node surfaced in review_log within 14 days is skipped."""
        conn = mock_graph.conn
        node = _make_node_dict("node-6", "Recent review node")
        _insert_node(conn, node)
        self._insert_whisper_row(conn, "node-6")
        conn.execute(
            "INSERT INTO review_log (node_id, session_id, surfaced_at) VALUES (?, ?, datetime('now', '-7 days'))",
            ("node-6", "sess-prev"),
        )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is None

    def test_exhausted_candidate_skipped(self, mock_graph):
        """Node with 3+ unanswered review_log rows is skipped."""
        conn = mock_graph.conn
        node = _make_node_dict("node-7", "Exhausted candidate")
        _insert_node(conn, node)
        self._insert_whisper_row(conn, "node-7")
        # Insert 3 unanswered review_log rows (all older than 14 days to avoid the recency filter)
        for i in range(3):
            conn.execute(
                "INSERT INTO review_log (node_id, session_id, surfaced_at, answered) "
                "VALUES (?, ?, datetime('now', '-20 days'), 0)",
                ("node-7", f"sess-old-{i}"),
            )
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is None

    def test_eligible_candidate_passes_all_checks(self, mock_graph):
        """Candidate with no affinity, no recent review_log, under 3 unanswered → returned."""
        conn = mock_graph.conn
        node = _make_node_dict("node-8", "Eligible candidate")
        _insert_node(conn, node)
        self._insert_whisper_row(conn, "node-8")
        conn.commit()

        result = _find_review_candidate(conn, threshold=0.70)
        assert result is not None
        assert result["node_id"] == "node-8"


# ---------------------------------------------------------------------------
# TestReviewBlockInBuildCoreContext
# ---------------------------------------------------------------------------

class TestReviewBlockInBuildWhisperContext:
    """Integration tests for review mechanism inside build_whisper_context."""

    def _make_mock_engine(self, conn=None, threshold=0.70, maintenance_enabled=False):
        engine = MagicMock()
        settings = MagicMock()
        settings.affinity_similarity_threshold = threshold
        settings.claude_maintenance_enabled = maintenance_enabled
        engine.settings = settings
        # Disable embedding/search paths so they don't interfere
        engine._get_hybrid_search.return_value = None
        engine.recall_search_structured.return_value = []
        if conn is not None:
            @contextmanager
            def _fake_transaction():
                yield conn
            engine.db.transaction = _fake_transaction
        return engine

    @staticmethod
    def _admitted_ordinary_result(node_id="node-selected"):
        """A final ordinary selection that permits the optional review append."""
        node = _make_node_dict(
            node_id,
            "Current authentication design",
            space="myspace",
            content="Authentication uses scoped tokens for the current application.",
        )
        return {"node": node, "score": 0.80, "source": "hybrid"}

    def _insert_whisper_row(self, conn, node_id, prompt_text="how does auth work", prompt_vec_bytes=b""):
        cursor = conn.execute(
            "INSERT INTO whisper_log (node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, was_injected, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-1 day'))",
            (node_id, 0.48, "sess-abc", "myspace", prompt_text, "hash-blk", prompt_vec_bytes, 0),
        )
        return cursor.lastrowid

    def test_review_block_appended_when_eligible(self, mock_graph):
        """Eligible candidate causes review block to appear in result on first message."""
        conn = mock_graph.conn
        node = _make_node_dict("node-r1", "Auth token storage", space="myspace")
        _insert_node(conn, node)
        whisper_log_id = self._insert_whisper_row(conn, "node-r1")
        conn.commit()

        engine = self._make_mock_engine(conn)
        engine.recall_search_structured.return_value = [self._admitted_ordinary_result()]
        builder = ContextBuilder(mock_graph, engine=engine)
        result = builder.build_whisper_context(
            prompt="how does auth work", recent_prompts=None, session_id="test-session-123"
        )

        assert "one thing to review when you get a chance" in result
        assert "Current authentication design" in result
        assert "Auth token storage" in result
        assert f"whisper_log_id: {whisper_log_id}" in result
        assert f"whisper_log_id={whisper_log_id}" in result

    def test_review_log_row_inserted(self, mock_graph):
        """After build_whisper_context with eligible candidate, review_log has 1 row."""
        conn = mock_graph.conn
        node = _make_node_dict("node-r2", "DB schema design", space="myspace")
        _insert_node(conn, node)
        self._insert_whisper_row(conn, "node-r2")
        conn.commit()

        engine = self._make_mock_engine(conn)
        engine.recall_search_structured.return_value = [self._admitted_ordinary_result()]
        builder = ContextBuilder(mock_graph, engine=engine)
        builder.build_whisper_context(
            prompt="how does auth work", recent_prompts=None, session_id="test-session-456"
        )

        row_count = conn.execute(
            "SELECT COUNT(*) FROM review_log WHERE node_id = ?", ("node-r2",)
        ).fetchone()[0]
        assert row_count == 1

        row = conn.execute(
            "SELECT node_id, session_id FROM review_log WHERE node_id = ?", ("node-r2",)
        ).fetchone()
        assert row["node_id"] == "node-r2"
        assert row["session_id"] == "test-session-456"

    def test_review_block_not_appended_without_engine(self, mock_graph):
        """ContextBuilder with engine=None produces no review block."""
        conn = mock_graph.conn
        node = _make_node_dict("node-r3", "No engine node", space="myspace")
        _insert_node(conn, node)
        self._insert_whisper_row(conn, "node-r3")
        conn.commit()

        builder = ContextBuilder(mock_graph, engine=None)
        result = builder.build_whisper_context(
            prompt="how does auth work", recent_prompts=None, session_id="test-session-789"
        )

        assert "one thing to review when you get a chance" not in result

    def test_review_block_not_appended_on_subsequent_messages(self, mock_graph):
        """Review block only fires on first message (recent_prompts=None)."""
        conn = mock_graph.conn
        node = _make_node_dict("node-r5", "Subsequent message node", space="myspace")
        _insert_node(conn, node)
        self._insert_whisper_row(conn, "node-r5")
        conn.commit()

        engine = self._make_mock_engine(conn)
        engine.recall_search_structured.return_value = [self._admitted_ordinary_result()]
        builder = ContextBuilder(mock_graph, engine=engine)
        result = builder.build_whisper_context(
            prompt="how does auth work",
            recent_prompts=["previous message"],  # not first message
            session_id="test-session-subseq",
        )

        assert "one thing to review when you get a chance" not in result
        assert "Current authentication design" in result

    def test_prompt_text_truncated_to_300(self, mock_graph):
        """Long prompt_text is truncated at word boundary to ≤300 chars + ellipsis."""
        conn = mock_graph.conn
        node = _make_node_dict("node-r4", "Long prompt node", space="myspace")
        _insert_node(conn, node)

        long_prompt = ("This is a very long prompt that goes on and on about authentication "
                       "and authorization flows in the application. ") * 4
        assert len(long_prompt) > 300

        self._insert_whisper_row(conn, "node-r4", prompt_text=long_prompt)
        conn.commit()

        engine = self._make_mock_engine(conn)
        engine.recall_search_structured.return_value = [self._admitted_ordinary_result()]
        builder = ContextBuilder(mock_graph, engine=engine)
        result = builder.build_whisper_context(
            prompt="how does auth work", recent_prompts=None, session_id="test-session-truncate"
        )

        assert "one thing to review when you get a chance" in result
        assert "…" in result
        review_start = result.find("one thing to review when you get a chance")
        review_portion = result[review_start:]
        working_on_idx = review_portion.find('working on:\n"')
        if working_on_idx != -1:
            snippet_start = working_on_idx + len('working on:\n"')
            snippet_end = review_portion.find('"', snippet_start)
            snippet = review_portion[snippet_start:snippet_end]
            assert len(snippet) <= 301  # 300 chars + "…"

    def _seed_same_space_review_candidate(self, conn, node_id="node-review-history"):
        node = _make_node_dict(
            node_id,
            "Held-back authentication history",
            space="myspace",
        )
        _insert_node(conn, node)
        whisper_log_id = self._insert_whisper_row(conn, node_id)
        conn.commit()
        # This verifies the fixture is actually eligible before the silent
        # selection below proves it does not consult the review mechanism.
        candidate = _find_review_candidate(conn, threshold=0.70)
        assert candidate is not None
        assert candidate["node_id"] == node_id
        return whisper_log_id

    @pytest.mark.parametrize(
        ("prompt", "session_id", "recent_prompts"),
        [
            pytest.param("Thanks, that helps.", None, None, id="sessionless-acknowledgement"),
            pytest.param("Thanks, that helps.", "first-session", None, id="first-session-turn"),
            pytest.param("Thanks, that helps.", "after-gap", None, id="after-session-gap"),
            pytest.param(
                "What is the boiling point of ethanol at sea level?",
                None,
                None,
                id="unrelated-question",
            ),
            pytest.param("Thanks, that helps.", "same-space", None, id="same-space-history"),
        ],
    )
    def test_no_candidates_do_not_lookup_or_log_review(
        self, mock_graph, prompt, session_id, recent_prompts
    ):
        """A silent final selection stays silent even with eligible history."""
        conn = mock_graph.conn
        self._seed_same_space_review_candidate(conn)
        engine = self._make_mock_engine(conn)
        builder = ContextBuilder(mock_graph, engine=engine)

        with patch("ormah.engine.context_builder._find_review_candidate") as review_lookup:
            result = builder.build_whisper_context(
                prompt=prompt,
                space="myspace",
                recent_prompts=recent_prompts,
                session_id=session_id,
            )

        assert result == ""
        review_lookup.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0] == 0
        outcome = conn.execute(
            "SELECT outcome, injected_count FROM whisper_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert tuple(outcome) == ("silent_no_candidates", 0)

    def test_gate_rejected_candidates_do_not_lookup_or_log_review(self, mock_graph):
        """A non-empty retrieval pool that fails the final gate is still silence."""
        conn = mock_graph.conn
        self._seed_same_space_review_candidate(conn)
        engine = self._make_mock_engine(conn)
        engine.recall_search_structured.return_value = [self._admitted_ordinary_result()]
        builder = ContextBuilder(mock_graph, engine=engine)

        with patch("ormah.engine.context_builder._find_review_candidate") as review_lookup:
            result = builder.build_whisper_context(
                prompt="how does auth work",
                space="myspace",
                recent_prompts=None,
                session_id="gate-rejected",
                injection_gate=0.90,
            )

        assert result == ""
        review_lookup.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0] == 0
        outcome = conn.execute(
            "SELECT outcome, injected_count FROM whisper_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert tuple(outcome) == ("silent_gate", 0)

    def test_maintenance_only_output_does_not_trigger_review(self, mock_graph):
        """Rendered maintenance text must not stand in for an admitted memory."""
        conn = mock_graph.conn
        self._seed_same_space_review_candidate(conn)
        engine = self._make_mock_engine(conn, maintenance_enabled=True)
        builder = ContextBuilder(mock_graph, engine=engine)

        with patch("ormah.engine.context_builder._find_review_candidate") as review_lookup:
            result = builder.build_whisper_context(
                prompt="Thanks, that helps.",
                space="myspace",
                recent_prompts=None,
                session_id="maintenance-only",
            )

        assert result == MAINTENANCE_DUE_SIGNAL
        review_lookup.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0] == 0

    def test_short_prompt_stays_silent_without_review(self, mock_graph):
        """The existing short-prompt silence contract precedes review lookup."""
        conn = mock_graph.conn
        self._seed_same_space_review_candidate(conn)
        engine = self._make_mock_engine(conn)
        builder = ContextBuilder(mock_graph, engine=engine)

        with patch("ormah.engine.context_builder._find_review_candidate") as review_lookup:
            result = builder.build_whisper_context(
                prompt="ok",
                space="myspace",
                recent_prompts=None,
                session_id="short-prompt",
            )

        assert result == ""
        review_lookup.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0] == 0

    def test_preference_only_selection_allows_review_with_exact_attribution(self, mock_graph):
        """An admitted preference is a final candidate and can carry a review."""
        conn = mock_graph.conn
        whisper_log_id = self._seed_same_space_review_candidate(conn)
        preference = _make_node_dict(
            "node-preference",
            "Prefer concise architecture",
            space="myspace",
            content="Keep architecture decisions concise and written down.",
        )
        preference["type"] = "preference"
        engine = self._make_mock_engine(conn)
        engine.has_searchable_preferences.return_value = True
        engine.recall_search_structured.side_effect = [
            [],
            [{"node": preference, "score": 0.65, "source": "hybrid"}],
        ]
        builder = ContextBuilder(mock_graph, engine=engine)
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.rerank.return_value = [3.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_cross_encoder):
            result = builder.build_whisper_context(
                prompt="plan this architecture change",
                space="myspace",
                recent_prompts=None,
                session_id="preference-only",
                reranker_enabled=True,
                preference_applicability_enabled=True,
                preference_applicability_gate=0.40,
            )

        assert "Prefer concise architecture" in result
        assert "one thing to review when you get a chance" in result
        assert f"whisper_log_id: {whisper_log_id}" in result
        assert f"whisper_log_id={whisper_log_id}" in result
