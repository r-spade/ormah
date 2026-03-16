"""Tests for hybrid search scoring mechanics.

These test the RRF fusion, threshold filtering, and ranking logic
using mocked retriever outputs — no embedding model or database needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ormah.config import Settings
from ormah.embeddings.hybrid_search import HybridSearch, _is_question_query, _reciprocal_rank_fusion
from ormah.index.graph import _IDENTITY_TOKENS, _STOP_WORDS, _sanitize_fts_query


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_rrf_empty_lists():
    assert _reciprocal_rank_fusion([], [], k=60) == {}


def test_rrf_single_list():
    scores = _reciprocal_rank_fusion([["a", "b", "c"]], weights=[1.0], k=60)
    # rank 1 → 1/61, rank 2 → 1/62, rank 3 → 1/63
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["b"] == pytest.approx(1.0 / 62)
    assert scores["c"] == pytest.approx(1.0 / 63)
    # Descending order preserved
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_overlap_accumulates():
    """A node in both lists should score higher than one in only one list."""
    scores = _reciprocal_rank_fusion(
        ranked_lists=[["a", "b"], ["a", "c"]],
        weights=[1.0, 1.0],
        k=60,
    )
    # "a" appears at rank 1 in both → 2 / 61
    assert scores["a"] == pytest.approx(2.0 / 61)
    # "b" and "c" each appear once at rank 2 → 1 / 62
    assert scores["b"] == pytest.approx(1.0 / 62)
    assert scores["c"] == pytest.approx(1.0 / 62)
    assert scores["a"] > scores["b"]


def test_rrf_weights_scale_contribution():
    """Higher weight should give proportionally higher contribution."""
    scores = _reciprocal_rank_fusion(
        ranked_lists=[["a"], ["b"]],
        weights=[0.4, 0.6],
        k=60,
    )
    assert scores["a"] == pytest.approx(0.4 / 61)
    assert scores["b"] == pytest.approx(0.6 / 61)
    assert scores["b"] > scores["a"]


# ---------------------------------------------------------------------------
# Fusion: score combination and ranking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hybrid(tmp_path):
    """HybridSearch with mocked internals — no real DB or encoder."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        # Disable score blending so these tests exercise pure RRF behavior
        similarity_blend_weight=0.0,
        fts_only_dampening=1.0,
        min_result_score=0.0,
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        db = MagicMock()
        db.conn = conn
        hs = HybridSearch(db, settings)
        # Stub get_node to return minimal node dicts
        _node_dict = lambda nid: {
            "id": nid,
            "type": "fact",
            "tier": "working",
            "content": f"content of {nid}",
        }
        MockGraph.return_value.get_node.side_effect = _node_dict
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: _node_dict(nid) for nid in ids
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value
    return hs


def _run_fusion(hs, fts_results, vec_results):
    """Run search with controlled FTS and vector outputs."""
    hs.graph.fts_search.return_value = fts_results
    hs.vec_store.search.return_value = vec_results
    hs.encoder.encode.return_value = "fake_vec"
    return hs.search("test query", limit=10)


def test_high_vector_similarity_beats_fts_only(mock_hybrid):
    """A result with strong semantic match should outrank one with only keyword match."""
    results = _run_fusion(
        mock_hybrid,
        fts_results=[
            {"id": "fts_only", "score": 10.0},
        ],
        vec_results=[
            {"id": "vec_winner", "similarity": 0.9},
            {"id": "fts_only", "similarity": 0.1},  # below threshold, dropped
        ],
    )
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "vec_winner"


def test_result_in_both_retrievers_ranks_highest(mock_hybrid):
    """A result strong in both FTS and vector should outrank single-source results."""
    results = _run_fusion(
        mock_hybrid,
        fts_results=[
            {"id": "both", "score": 10.0},
            {"id": "fts_only", "score": 8.0},
        ],
        vec_results=[
            {"id": "both", "similarity": 0.9},
            {"id": "vec_only", "similarity": 0.85},
        ],
    )
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "both"


def test_threshold_filters_low_similarity(mock_hybrid):
    """Vector results below similarity_threshold should not contribute to scoring."""
    results = _run_fusion(
        mock_hybrid,
        fts_results=[
            {"id": "relevant", "score": 5.0},
        ],
        vec_results=[
            {"id": "noise", "similarity": 0.2},   # below 0.4 threshold
            {"id": "relevant", "similarity": 0.5},
        ],
    )
    ids = [r["node"]["id"] for r in results]
    # "noise" should only appear if FTS returned it — it didn't, so it's absent
    assert "noise" not in ids
    assert "relevant" in ids


def test_vector_fallback_on_encoder_failure(mock_hybrid):
    """If vector search fails, results should still come from FTS alone."""
    mock_hybrid.encoder.encode.side_effect = RuntimeError("model not loaded")
    mock_hybrid.graph.fts_search.return_value = [
        {"id": "a", "score": 5.0},
        {"id": "b", "score": 2.0},
    ]
    results = mock_hybrid.search("test query", limit=10)
    assert len(results) == 2
    # FTS-only: "a" had higher raw score → should be first
    assert results[0]["node"]["id"] == "a"


def test_limit_respected(mock_hybrid):
    """Should never return more results than the limit."""
    mock_hybrid.graph.fts_search.return_value = [{"id": f"n{i}", "score": float(10 - i)} for i in range(10)]
    mock_hybrid.vec_store.search.return_value = [{"id": f"n{i}", "similarity": 0.9 - i * 0.05} for i in range(10)]
    mock_hybrid.encoder.encode.return_value = "fake_vec"
    results = mock_hybrid.search("test query", limit=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Batch fetch verification
# ---------------------------------------------------------------------------


def test_batch_node_fetch_used(mock_hybrid):
    """Search should use get_nodes_batch instead of individual get_node calls."""
    _run_fusion(
        mock_hybrid,
        fts_results=[{"id": "a", "score": 5.0}, {"id": "b", "score": 3.0}],
        vec_results=[{"id": "a", "similarity": 0.8}],
    )
    mock_hybrid.graph.get_nodes_batch.assert_called_once()
    mock_hybrid.graph.get_node.assert_not_called()


def test_tag_filtering_uses_batch(mock_hybrid):
    """When tags filter is provided, get_tags_batch should be called once."""
    mock_hybrid.graph.get_tags_batch.side_effect = lambda ids: {
        "a": {"important"},
        "b": {"important"},
    }
    mock_hybrid.graph.fts_search.return_value = [
        {"id": "a", "score": 5.0},
        {"id": "b", "score": 3.0},
    ]
    mock_hybrid.vec_store.search.return_value = []
    mock_hybrid.encoder.encode.return_value = "fake_vec"
    results = mock_hybrid.search("test query", limit=10, tags=["important"])
    mock_hybrid.graph.get_tags_batch.assert_called_once()
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Score blending & minimum score filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hybrid_blended(tmp_path):
    """HybridSearch with blending enabled (default settings)."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        similarity_blend_weight=0.5,
        fts_only_dampening=0.5,
        min_result_score=0.1,
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        db = MagicMock()
        db.conn = conn
        hs = HybridSearch(db, settings)
        _node_dict = lambda nid: {
            "id": nid,
            "type": "fact",
            "tier": "working",
            "content": f"content of {nid}",
        }
        MockGraph.return_value.get_node.side_effect = _node_dict
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: _node_dict(nid) for nid in ids
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value
    return hs


def test_blended_scores_widen_range(mock_hybrid_blended):
    """High-similarity vector result should score significantly higher than low-similarity."""
    results = _run_fusion(
        mock_hybrid_blended,
        fts_results=[
            {"id": "high_sim", "score": 5.0},
            {"id": "low_sim", "score": 5.0},
        ],
        vec_results=[
            {"id": "high_sim", "similarity": 0.92},
            {"id": "low_sim", "similarity": 0.45},
        ],
    )
    scores = {r["node"]["id"]: r["score"] for r in results}
    assert scores["high_sim"] > scores["low_sim"]
    # The spread should be meaningful — at least 1.2x difference
    assert scores["high_sim"] / scores["low_sim"] > 1.2


def test_fts_only_dampened(tmp_path):
    """A result in FTS but not in vector results should score lower than one in both."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        similarity_blend_weight=0.5,
        fts_only_dampening=0.5,
        min_result_score=0.0,  # disable filtering so we can compare scores
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        db = MagicMock()
        db.conn = conn
        hs = HybridSearch(db, settings)
        _node_dict = lambda nid: {
            "id": nid,
            "type": "fact",
            "tier": "working",
            "content": f"content of {nid}",
        }
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: _node_dict(nid) for nid in ids
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value

    results = _run_fusion(
        hs,
        fts_results=[
            {"id": "both", "score": 5.0},
            {"id": "fts_only", "score": 10.0},  # higher FTS score but no vector match
        ],
        vec_results=[
            {"id": "both", "similarity": 0.8},
        ],
    )
    scores = {r["node"]["id"]: r["score"] for r in results}
    assert scores["both"] > scores["fts_only"]


def test_min_score_filters_noise(tmp_path):
    """Results below min_result_score should be excluded.

    With normalized RRF + dampening, an FTS-only result (no vector match)
    scores lower. We set min_result_score high enough to catch it.
    """
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        similarity_blend_weight=0.5,
        fts_only_dampening=0.5,
        min_result_score=0.6,  # high enough to filter dampened FTS-only results
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        db = MagicMock()
        db.conn = conn
        hs = HybridSearch(db, settings)
        _node_dict = lambda nid: {
            "id": nid,
            "type": "fact",
            "tier": "working",
            "content": f"content of {nid}",
        }
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: _node_dict(nid) for nid in ids
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value

    results = _run_fusion(
        hs,
        fts_results=[
            {"id": "relevant", "score": 5.0},
            {"id": "noise", "score": 2.0},
        ],
        vec_results=[
            {"id": "relevant", "similarity": 0.9},
            # "noise" not in vector results → dampened → below 0.6 threshold
        ],
    )
    ids = [r["node"]["id"] for r in results]
    assert "relevant" in ids
    assert "noise" not in ids


def test_min_score_disabled_when_zero(tmp_path):
    """With min_result_score=0, all results should pass through."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        similarity_blend_weight=0.5,
        fts_only_dampening=0.5,
        min_result_score=0.0,
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        db = MagicMock()
        db.conn = conn
        hs = HybridSearch(db, settings)
        _node_dict = lambda nid: {
            "id": nid,
            "type": "fact",
            "tier": "working",
            "content": f"content of {nid}",
        }
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: _node_dict(nid) for nid in ids
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value

    results = _run_fusion(
        hs,
        fts_results=[
            {"id": "good", "score": 5.0},
            {"id": "weak", "score": 1.0},
        ],
        vec_results=[
            {"id": "good", "similarity": 0.9},
            # "weak" not in vector → dampened but still returned
        ],
    )
    ids = [r["node"]["id"] for r in results]
    assert "good" in ids
    assert "weak" in ids


# ---------------------------------------------------------------------------
# Title match bonus & content length penalty
# ---------------------------------------------------------------------------


def _make_titled_hybrid(tmp_path, node_dicts, content_lengths=None, **overrides):
    """Create a HybridSearch with titled nodes and optional content lengths."""
    defaults = dict(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        similarity_blend_weight=0.5,
        fts_only_dampening=0.5,
        min_result_score=0.0,
        title_match_boost=2.0,
        length_penalty_threshold=300,
    )
    defaults.update(overrides)
    settings = Settings(**defaults)
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        db = MagicMock()
        db.conn = conn
        # Mock conn.execute for content length query
        if content_lengths:
            def _execute(sql, params=None):
                result = MagicMock()
                if "length(content)" in sql:
                    rows = [{"id": nid, "len": clen} for nid, clen in content_lengths.items()]
                    result.fetchall.return_value = rows
                else:
                    result.fetchall.return_value = []
                return result
            conn.execute.side_effect = _execute
        else:
            def _execute(sql, params=None):
                result = MagicMock()
                if "length(content)" in sql:
                    rows = [{"id": nid, "len": 100} for nid in (params or [])]
                    result.fetchall.return_value = rows
                else:
                    result.fetchall.return_value = []
                return result
            conn.execute.side_effect = _execute

        hs = HybridSearch(db, settings)
        node_map = {n["id"]: n for n in node_dicts}
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: node_map[nid] for nid in ids if nid in node_map
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value
    return hs


def test_title_match_boosts_score(tmp_path):
    """A node with the query term in its title scores higher than one with it only in content."""
    nodes = [
        {"id": "title_match", "type": "fact", "tier": "working", "title": "Dislikes grapes", "content": "Some info"},
        {"id": "content_only", "type": "fact", "tier": "working", "title": "Food preferences", "content": "Dislikes grapes a lot"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    hs.graph.fts_search.return_value = [
        {"id": "title_match", "score": 5.0},
        {"id": "content_only", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "title_match", "similarity": 0.7},
        {"id": "content_only", "similarity": 0.7},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("grapes", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    assert scores["title_match"] > scores["content_only"]


def test_fts_title_column_weight(tmp_path):
    """Verify that FTS query uses bm25 column weights (title 10x, tags 5x).

    This is an integration-style test that verifies the SQL in graph.py.
    We test via the GraphIndex directly with a real FTS5 table.
    """
    import sqlite3 as real_sqlite3

    conn = real_sqlite3.connect(":memory:")
    conn.row_factory = real_sqlite3.Row
    conn.execute(
        "CREATE VIRTUAL TABLE nodes_fts USING fts5(title, content, tags)"
    )
    # Node A: "grapes" in title only
    conn.execute(
        "INSERT INTO nodes_fts(rowid, title, content, tags) VALUES (1, 'Dislikes grapes', 'Some food info', '')"
    )
    # Node B: "grapes" in content only
    conn.execute(
        "INSERT INTO nodes_fts(rowid, title, content, tags) VALUES (2, 'Food preferences', 'I really dislike grapes very much', '')"
    )
    # We need an id column — FTS5 doesn't have one by default.
    # The actual schema uses content-sync; here we test the ranking SQL directly.
    rows = conn.execute(
        """
        SELECT rowid, bm25(nodes_fts, 10.0, 1.0, 5.0) as rank
        FROM nodes_fts
        WHERE nodes_fts MATCH 'grapes'
        ORDER BY rank
        """,
    ).fetchall()
    # bm25 returns negative scores (lower = better match)
    # Title match (rowid=1) should have a better (more negative) rank
    rank_map = {r["rowid"]: r["rank"] for r in rows}
    assert rank_map[1] < rank_map[2], "Title match should rank better (lower bm25) than content match"


def test_fts_porter_stemmer(tmp_path):
    """FTS5 with porter stemmer matches morphological variants (live → lives)."""
    import sqlite3 as real_sqlite3

    conn = real_sqlite3.connect(":memory:")
    conn.row_factory = real_sqlite3.Row
    conn.execute(
        "CREATE VIRTUAL TABLE nodes_fts USING fts5("
        "title, content, tags, tokenize='porter unicode61')"
    )
    conn.execute(
        "INSERT INTO nodes_fts(rowid, title, content, tags) "
        "VALUES (1, 'Lives in London', 'Alice lives in London, England.', '')"
    )
    conn.execute(
        "INSERT INTO nodes_fts(rowid, title, content, tags) "
        "VALUES (2, 'User email', 'The user email is test@test.com', '')"
    )
    # "live" should match "lives" via porter stemming
    rows = conn.execute(
        "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'live'"
    ).fetchall()
    matched = {r["rowid"] for r in rows}
    assert 1 in matched, "Porter stemmer should match 'live' to 'lives'"
    assert 2 not in matched


def test_title_punctuation_stripped(tmp_path):
    """Title match should work even when title contains punctuation (e.g., commas)."""
    nodes = [
        {"id": "dublin", "type": "fact", "tier": "working",
         "title": "Lives in Dublin, Ireland", "content": "Location info"},
        {"id": "other", "type": "fact", "tier": "working",
         "title": "Something else", "content": "Dublin mentioned here"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    hs.graph.fts_search.return_value = [
        {"id": "dublin", "score": 5.0},
        {"id": "other", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "dublin", "similarity": 0.6},
        {"id": "other", "similarity": 0.6},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("Dublin", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    # "dublin" should get title match bonus despite the comma in "Dublin,"
    assert scores["dublin"] > scores["other"]


def test_long_content_penalized(tmp_path):
    """A long document with moderate vector similarity scores lower than a short one with high similarity."""
    nodes = [
        {"id": "short", "type": "fact", "tier": "working", "title": "Short note", "content": "Brief"},
        {"id": "long", "type": "fact", "tier": "working", "title": "Long doc", "content": "x" * 3000},
    ]
    content_lengths = {"short": 50, "long": 3000}
    hs = _make_titled_hybrid(tmp_path, nodes, content_lengths=content_lengths)
    hs.graph.fts_search.return_value = []
    hs.vec_store.search.return_value = [
        {"id": "short", "similarity": 0.75},
        {"id": "long", "similarity": 0.70},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    # Long doc (3000 chars) should be penalized: penalty = max(0.1, 300/3000) = 0.1
    # Short doc (50 chars) gets no penalty
    assert scores["short"] > scores["long"]


def test_length_penalty_disabled_at_zero(tmp_path):
    """With length_penalty_threshold=0, no penalty is applied — long and short score the same."""
    nodes = [
        {"id": "short", "type": "fact", "tier": "working", "title": "A", "content": "Brief"},
        {"id": "long", "type": "fact", "tier": "working", "title": "B", "content": "x" * 3000},
    ]
    hs = _make_titled_hybrid(
        tmp_path, nodes, length_penalty_threshold=0, title_match_boost=0.0,
    )
    hs.graph.fts_search.return_value = []
    hs.vec_store.search.return_value = [
        {"id": "short", "similarity": 0.7},
        {"id": "long", "similarity": 0.7},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    # With no length penalty and no title boost, identical similarity → scores
    # should be very close (small RRF rank difference is acceptable)
    assert scores["short"] == pytest.approx(scores["long"], rel=0.01)


def test_very_long_content_heavily_penalized(tmp_path):
    """A 5000-char document should retain only 10% of its similarity (floor = 0.1)."""
    nodes = [
        {"id": "short", "type": "fact", "tier": "working", "title": "Short", "content": "Brief"},
        {"id": "huge", "type": "fact", "tier": "working", "title": "Huge doc", "content": "x" * 5000},
    ]
    content_lengths = {"short": 50, "huge": 5000}
    hs = _make_titled_hybrid(tmp_path, nodes, content_lengths=content_lengths, title_match_boost=0.0)
    hs.graph.fts_search.return_value = []
    hs.vec_store.search.return_value = [
        {"id": "short", "similarity": 0.75},
        {"id": "huge", "similarity": 0.75},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    # huge: penalty = max(0.1, 300/5000) = 0.1 → penalized sim = 0.075
    # short: no penalty → sim = 0.75
    # With normalized RRF + blend, short should clearly dominate
    assert scores["short"] > scores["huge"]
    assert scores["short"] / scores["huge"] > 1.5


def test_tag_filter_works(mock_hybrid):
    """Nodes with matching tags returned, non-matching filtered out."""
    mock_hybrid.graph.get_tags_batch.side_effect = lambda ids: {
        "a": {"important", "project"},
        "b": {"draft"},
    }
    mock_hybrid.graph.fts_search.return_value = [
        {"id": "a", "score": 5.0},
        {"id": "b", "score": 3.0},
    ]
    mock_hybrid.vec_store.search.return_value = []
    mock_hybrid.encoder.encode.return_value = "fake_vec"
    results = mock_hybrid.search("test query", limit=10, tags=["important"])
    ids = [r["node"]["id"] for r in results]
    assert "a" in ids
    assert "b" not in ids


# ---------------------------------------------------------------------------
# Question detection & question-aware scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "where does the user live",
        "What is the user's name",
        "who is the project owner",
        "How do I configure logging",
        "Is the server running",
        "does the user like grapes",
        "  When was the last deploy",
    ],
)
def test_question_detection_positive(query):
    assert _is_question_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Dublin",
        "grapes",
        "user email",
        "food preferences",
        "project architecture",
    ],
)
def test_question_detection_negative(query):
    assert _is_question_query(query) is False


def test_question_query_favors_vector_over_fts_title(tmp_path):
    """'where does the user live' should rank 'Lives in Dublin' above 'User email'.

    After stop-word removal, FTS sees {"user", "live"} and matches "User email"
    via the 10x title weight. For questions, we scale down FTS weight and disable
    the title-match boost so vector similarity (which correctly finds the semantic
    match) dominates.
    """
    nodes = [
        {"id": "email", "type": "fact", "tier": "working",
         "title": "User email", "content": "The user email is test@test.com"},
        {"id": "dublin", "type": "fact", "tier": "working",
         "title": "Lives in London", "content": "Alice lives in London, England."},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    # FTS ranks "email" first because "user" matches its title with 10x weight
    hs.graph.fts_search.return_value = [
        {"id": "email", "score": 12.0},
        {"id": "dublin", "score": 4.0},
    ]
    # Vector search correctly identifies the semantic match
    hs.vec_store.search.return_value = [
        {"id": "dublin", "similarity": 0.85},
        {"id": "email", "similarity": 0.45},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("where does the user live", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "dublin", f"Expected 'dublin' first, got {ids}"


def test_keyword_query_still_uses_title_boost(tmp_path):
    """Keyword query 'grapes' should still benefit from title matching — no regression."""
    nodes = [
        {"id": "grapes_title", "type": "fact", "tier": "working",
         "title": "Dislikes grapes", "content": "Some info about food"},
        {"id": "grapes_content", "type": "fact", "tier": "working",
         "title": "Food preferences", "content": "Dislikes grapes a lot"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    hs.graph.fts_search.return_value = [
        {"id": "grapes_title", "score": 8.0},
        {"id": "grapes_content", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "grapes_title", "similarity": 0.7},
        {"id": "grapes_content", "similarity": 0.7},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("grapes", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    assert scores["grapes_title"] > scores["grapes_content"]


def test_question_similarity_blend_dominates(tmp_path):
    """For questions, raw vector similarity should dominate over RRF rank position.

    A node that is #1 in FTS (high RRF) but has low vector similarity should
    lose to a node with high vector similarity but lower FTS rank. The 0.85
    similarity blend weight makes this possible.
    """
    nodes = [
        {"id": "fts_winner", "type": "fact", "tier": "working",
         "title": "API route groups", "content": "Routes for the server"},
        {"id": "semantic_match", "type": "fact", "tier": "working",
         "title": "MCP integration", "content": "MCP connects agents via stdio"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    # FTS strongly favors fts_winner (broad keyword match)
    hs.graph.fts_search.return_value = [
        {"id": "fts_winner", "score": 15.0},
        {"id": "semantic_match", "score": 3.0},
    ]
    # Vector search clearly prefers the semantic match
    hs.vec_store.search.return_value = [
        {"id": "semantic_match", "similarity": 0.82},
        {"id": "fts_winner", "similarity": 0.45},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("how does MCP connect to agents", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "semantic_match", f"Expected semantic_match first, got {ids}"


def test_question_blend_vs_keyword_blend_differ(tmp_path):
    """Question and keyword queries for identical retrieval results should
    produce different score ratios — questions should spread scores wider
    because similarity blend amplifies the vector magnitude gap.
    """
    nodes = [
        {"id": "high_sim", "type": "fact", "tier": "working",
         "title": "Target", "content": "The actual answer"},
        {"id": "low_sim", "type": "fact", "tier": "working",
         "title": "Noise", "content": "Unrelated stuff"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, title_match_boost=0.0)
    fts = [
        {"id": "high_sim", "score": 5.0},
        {"id": "low_sim", "score": 5.0},
    ]
    vec = [
        {"id": "high_sim", "similarity": 0.85},
        {"id": "low_sim", "similarity": 0.45},
    ]

    # Keyword query
    hs.graph.fts_search.return_value = list(fts)
    hs.vec_store.search.return_value = list(vec)
    hs.encoder.encode.return_value = "fake_vec"
    kw_results = hs.search("target answer", limit=10)
    kw_scores = {r["node"]["id"]: r["score"] for r in kw_results}
    kw_ratio = kw_scores["high_sim"] / kw_scores["low_sim"]

    # Question query — same retrieval results
    hs.graph.fts_search.return_value = list(fts)
    hs.vec_store.search.return_value = list(vec)
    hs.encoder.encode.return_value = "fake_vec"
    q_results = hs.search("what is the target answer", limit=10)
    q_scores = {r["node"]["id"]: r["score"] for r in q_results}
    q_ratio = q_scores["high_sim"] / q_scores["low_sim"]

    # Question query should produce a wider spread (higher ratio)
    assert q_ratio > kw_ratio, (
        f"Question ratio {q_ratio:.2f} should exceed keyword ratio {kw_ratio:.2f}"
    )


def test_question_fts_only_result_loses_to_vector_match(tmp_path):
    """An FTS-only result (no vector match) should rank below a vector-matched
    result for question queries, even if the FTS-only result has a much higher
    FTS score.
    """
    nodes = [
        {"id": "fts_only", "type": "fact", "tier": "working",
         "title": "User settings", "content": "Settings for the user account"},
        {"id": "vec_match", "type": "fact", "tier": "working",
         "title": "Lives in Dublin", "content": "Lives in Dublin, Ireland"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    # FTS strongly prefers fts_only; vec_match has low FTS score
    hs.graph.fts_search.return_value = [
        {"id": "fts_only", "score": 20.0},
        {"id": "vec_match", "score": 2.0},
    ]
    # Only vec_match has meaningful vector similarity; fts_only below threshold
    hs.vec_store.search.return_value = [
        {"id": "vec_match", "similarity": 0.78},
        {"id": "fts_only", "similarity": 0.3},  # below 0.4 threshold
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("where does the user live", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    assert scores["vec_match"] > scores["fts_only"], (
        f"vec_match ({scores['vec_match']:.4f}) should beat fts_only ({scores['fts_only']:.4f})"
    )


def test_question_title_boost_disabled(tmp_path):
    """For question queries, title match boost should be disabled — matching
    tokens in titles should not inflate scores.
    """
    nodes = [
        {"id": "title_trap", "type": "fact", "tier": "working",
         "title": "User email config", "content": "Email configuration details"},
        {"id": "correct", "type": "fact", "tier": "working",
         "title": "Dublin residence", "content": "Lives in Dublin since 2018"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    # Both in FTS with similar scores
    hs.graph.fts_search.return_value = [
        {"id": "title_trap", "score": 8.0},
        {"id": "correct", "score": 6.0},
    ]
    # Vector clearly favors the correct answer
    hs.vec_store.search.return_value = [
        {"id": "correct", "similarity": 0.80},
        {"id": "title_trap", "similarity": 0.50},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("where does the user live", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "correct", (
        f"Expected 'correct' first (title boost should be disabled for questions), got {ids}"
    )


def test_keyword_title_boost_still_active(tmp_path):
    """For keyword queries, title boost should still be active — a node with
    the keyword in its title should beat one with equal vector similarity but
    no title match.
    """
    nodes = [
        {"id": "titled", "type": "fact", "tier": "working",
         "title": "Dublin facts", "content": "Information about the city"},
        {"id": "untitled", "type": "fact", "tier": "working",
         "title": "City information", "content": "Dublin is the capital of Ireland"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    hs.graph.fts_search.return_value = [
        {"id": "titled", "score": 5.0},
        {"id": "untitled", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "titled", "similarity": 0.7},
        {"id": "untitled", "similarity": 0.7},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("Dublin", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    assert scores["titled"] > scores["untitled"], (
        "Keyword query should still get title boost"
    )


# ---------------------------------------------------------------------------
# "user" stop word
# ---------------------------------------------------------------------------


def test_user_in_stop_words():
    """'user' should be in the FTS stop words list."""
    assert "user" in _STOP_WORDS


def test_user_token_stripped_from_fts():
    """FTS query for 'user capitalism' should strip 'user' and produce just 'capitalism'."""
    queries = _sanitize_fts_query("what does the user think about capitalism")
    # After stop-word removal, only "think" and "capitalism" should survive
    # (all other words are stop words)
    assert len(queries) >= 1
    for q in queries:
        assert "user" not in q.lower()
        assert "capitalism" in q.lower()


def test_capitalism_query_ranks_correctly(tmp_path):
    """'what does the user think about capitalism' should rank 'Dislikes capitalism' highly.

    With 'user' removed from FTS, the query becomes 'think capitalism' which
    correctly matches 'Dislikes capitalism' without spuriously boosting nodes
    that merely mention 'the user'.
    """
    nodes = [
        {"id": "capitalism", "type": "preference", "tier": "working",
         "title": "Dislikes capitalism", "content": "The user thinks capitalism is exploitative"},
        {"id": "color", "type": "preference", "tier": "working",
         "title": "Favorite color: red", "content": "The user likes the color red"},
        {"id": "email", "type": "fact", "tier": "working",
         "title": "User email", "content": "The user email is test@test.com"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes)
    # With "user" as stop word, FTS for "think capitalism" matches capitalism node
    hs.graph.fts_search.return_value = [
        {"id": "capitalism", "score": 8.0},
        {"id": "color", "score": 2.0},
        {"id": "email", "score": 1.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "capitalism", "similarity": 0.85},
        {"id": "color", "similarity": 0.3},
        {"id": "email", "similarity": 0.2},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("what does the user think about capitalism", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "capitalism", f"Expected 'capitalism' first, got {ids}"


# ---------------------------------------------------------------------------
# Expired node exclusion & confidence factor
# ---------------------------------------------------------------------------


def test_expired_nodes_excluded(tmp_path):
    """A node with valid_until in the past should be completely excluded from results."""
    nodes = [
        {"id": "alive", "type": "fact", "tier": "working",
         "title": "Active node", "content": "Still valid"},
        {"id": "expired", "type": "fact", "tier": "working",
         "title": "Expired node", "content": "No longer valid",
         "valid_until": "2020-01-01T00:00:00+00:00"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, title_match_boost=0.0)
    hs.graph.fts_search.return_value = [
        {"id": "alive", "score": 5.0},
        {"id": "expired", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "alive", "similarity": 0.8},
        {"id": "expired", "similarity": 0.8},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert "alive" in ids
    assert "expired" not in ids, "Expired nodes should be hard-filtered from results"


def test_non_expired_node_not_filtered(tmp_path):
    """A node with valid_until in the future should not be filtered."""
    nodes = [
        {"id": "future", "type": "fact", "tier": "working",
         "title": "Future expiry", "content": "Still valid",
         "valid_until": "2099-01-01T00:00:00+00:00"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, title_match_boost=0.0)
    hs.graph.fts_search.return_value = [
        {"id": "future", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "future", "similarity": 0.8},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert "future" in ids


def test_low_confidence_penalized_more(tmp_path):
    """confidence=0.0 should give 0.4x factor (not 0.7x), widening the gap
    between confident and speculative nodes.
    """
    nodes = [
        {"id": "confident", "type": "fact", "tier": "working",
         "title": "Confident", "content": "Sure thing",
         "confidence": 1.0, "importance": 0.5},
        {"id": "unsure", "type": "fact", "tier": "working",
         "title": "Unsure", "content": "Maybe",
         "confidence": 0.0, "importance": 0.5},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, title_match_boost=0.0)
    hs.graph.fts_search.return_value = [
        {"id": "confident", "score": 5.0},
        {"id": "unsure", "score": 5.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "confident", "similarity": 0.7},
        {"id": "unsure", "similarity": 0.7},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=10)
    scores = {r["node"]["id"]: r["score"] for r in results}
    # With confidence_factor = 0.4 + 0.6*c:
    #   confident: 0.4 + 0.6*1.0 = 1.0
    #   unsure:    0.4 + 0.6*0.0 = 0.4
    # The ratio should be significantly > 1 (around 2.5x with importance factored in)
    ratio = scores["confident"] / scores["unsure"]
    assert ratio > 2.0, f"Expected ratio > 2.0, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# Min-max RRF normalization
# ---------------------------------------------------------------------------


def test_minmax_normalization_spreads_scores(tmp_path):
    """With 10 results in both lists, min-max normalization should produce
    a score spread > 0.3 (was ~0.13 with max-norm).
    """
    n = 10
    nodes = [
        {"id": f"n{i}", "type": "fact", "tier": "working", "content": f"content {i}"}
        for i in range(n)
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, similarity_blend_weight=0.0,
                             fts_only_dampening=1.0, min_result_score=0.0,
                             title_match_boost=0.0)
    # Both retrievers return the same 10 results in the same order
    fts = [{"id": f"n{i}", "score": float(10 - i)} for i in range(n)]
    vec = [{"id": f"n{i}", "similarity": 0.9 - i * 0.04} for i in range(n)]
    hs.graph.fts_search.return_value = fts
    hs.vec_store.search.return_value = vec
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=n)
    scores = [r["score"] for r in results]
    spread = max(scores) - min(scores)
    assert spread > 0.3, f"Expected score spread > 0.3, got {spread:.4f}"


def test_minmax_fallback_for_close_scores(tmp_path):
    """Two results at adjacent ranks in both lists (spread ~1.6%) should
    use max-norm fallback, keeping scores close (ratio < 1.05).
    """
    nodes = [
        {"id": "a", "type": "fact", "tier": "working", "content": "content a"},
        {"id": "b", "type": "fact", "tier": "working", "content": "content b"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, similarity_blend_weight=0.0,
                             fts_only_dampening=1.0, min_result_score=0.0,
                             title_match_boost=0.0)
    # Adjacent ranks in both lists → very small RRF spread
    fts = [{"id": "a", "score": 5.0}, {"id": "b", "score": 4.0}]
    vec = [{"id": "a", "similarity": 0.8}, {"id": "b", "similarity": 0.78}]
    hs.graph.fts_search.return_value = fts
    hs.vec_store.search.return_value = vec
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=2)
    scores = {r["node"]["id"]: r["score"] for r in results}
    ratio = scores["a"] / scores["b"]
    assert ratio < 1.05, f"Expected ratio < 1.05 (max-norm fallback), got {ratio:.4f}"


def test_single_result_normalization(tmp_path):
    """A single result should use max-norm fallback and score > 0.7."""
    nodes = [
        {"id": "solo", "type": "fact", "tier": "working", "content": "only one"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, similarity_blend_weight=0.0,
                             fts_only_dampening=1.0, min_result_score=0.0,
                             title_match_boost=0.0)
    fts = [{"id": "solo", "score": 5.0}]
    vec = [{"id": "solo", "similarity": 0.8}]
    hs.graph.fts_search.return_value = fts
    hs.vec_store.search.return_value = vec
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("test query", limit=1)
    assert len(results) == 1
    assert results[0]["score"] > 0.7, f"Single result should score > 0.7, got {results[0]['score']:.4f}"


def test_minmax_with_question_query(tmp_path):
    """Question query with FTS and vec disagreement — semantic match should
    still win because question blend (0.85) dominates over RRF normalization.
    """
    nodes = [
        {"id": "fts_top", "type": "fact", "tier": "working",
         "title": "General info", "content": "Broad topic match"},
        {"id": "semantic", "type": "fact", "tier": "working",
         "title": "Specific answer", "content": "The precise answer"},
    ]
    hs = _make_titled_hybrid(tmp_path, nodes, title_match_boost=0.0)
    # FTS favors fts_top; vector favors semantic
    hs.graph.fts_search.return_value = [
        {"id": "fts_top", "score": 12.0},
        {"id": "semantic", "score": 3.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "semantic", "similarity": 0.88},
        {"id": "fts_top", "similarity": 0.50},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    results = hs.search("what is the specific answer", limit=10)
    ids = [r["node"]["id"] for r in results]
    assert ids[0] == "semantic", f"Expected semantic match first, got {ids}"


# ---------------------------------------------------------------------------
# FTS identity signal via query rewriting
# ---------------------------------------------------------------------------


def test_identity_tokens_subset_of_stop_words():
    """All identity tokens should be in the stop words list."""
    assert _IDENTITY_TOKENS <= _STOP_WORDS


def test_fts_query_rewrite_user_name():
    """'what is the user's name' should inject about_self into FTS tokens."""
    queries = _sanitize_fts_query("what is the user's name")
    assert len(queries) >= 1
    assert "about_self" in queries[0]
    assert "name" in queries[0]


def test_fts_query_rewrite_no_identity():
    """'grapes' has no identity token — should NOT inject about_self."""
    queries = _sanitize_fts_query("grapes")
    assert len(queries) == 1
    assert "about_self" not in queries[0]


def test_fts_query_rewrite_user_likes_grapes():
    """'does the user like grapes' should inject about_self."""
    queries = _sanitize_fts_query("does the user like grapes")
    assert len(queries) >= 1
    # AND query should have all: like, grapes, about_self
    and_query = queries[0]
    assert "about_self" in and_query
    assert "like" in and_query
    assert "grapes" in and_query


def test_fts_query_rewrite_only_identity_tokens():
    """A query with only identity tokens (all stopped) should fall back
    to raw tokens AND inject about_self since identity tokens were present."""
    queries = _sanitize_fts_query("I me my")
    # All tokens are stop words → fallback to cleaned.split() with len > 1 → "me", "my"
    # Identity tokens were detected, so about_self is injected after fallback
    assert len(queries) >= 1
    for q in queries:
        assert "about_self" in q


def test_fts_query_rewrite_my_email():
    """'my email' should inject about_self alongside 'email'."""
    queries = _sanitize_fts_query("my email")
    assert len(queries) >= 1
    assert "about_self" in queries[0]
    assert "email" in queries[0]
