"""Tests for LLM-based edge type classification in auto_linker."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from ormah.models.node import CreateNodeRequest, NodeType

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _create_pair(engine, title_a="Python language", content_a="Python is a programming language.",
                 title_b="Python lang", content_b="Python is a popular programming language.",
                 node_type=NodeType.fact):
    """Helper: create two similar nodes without auto-linking, return their IDs."""
    # Suppress auto-link during creation so run_auto_linker controls edge creation
    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = engine.remember(
            CreateNodeRequest(content=content_a, type=node_type, title=title_a, tags=["test"]),
            agent_id="test",
        )
        id_b, _ = engine.remember(
            CreateNodeRequest(content=content_b, type=node_type, title=title_b, tags=["test"]),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold
    return id_a, id_b


def _edges_between(engine, id_a, id_b):
    """Return all edges between two nodes."""
    return engine.db.conn.execute(
        "SELECT edge_type FROM edges WHERE "
        "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
        (id_a, id_b, id_b, id_a),
    ).fetchall()


def _reset_adapter():
    from ormah.background.llm_client import reset_adapter
    reset_adapter()


def test_llm_classifies_supports(engine):
    """LLM classifies as supports -> edge created with type supports."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "supports",
        "reason": "Both describe Python as a programming language.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) >= 1
    assert edges[0]["edge_type"] == "supports"


def test_llm_classifies_contradicts(engine):
    """LLM classifies as contradicts -> edge created with type contradicts."""
    id_a, id_b = _create_pair(
        engine,
        title_a="Python is fast",
        content_a="Python is the fastest programming language.",
        title_b="Python is slow",
        content_b="Python is one of the slowest programming languages.",
    )

    llm_response = json.dumps({
        "relationship": "contradicts",
        "reason": "They make opposing claims about Python speed.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) >= 1
    assert edges[0]["edge_type"] == "contradicts"


def test_llm_classifies_none_no_edge(engine):
    """LLM classifies as none -> no edge created."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 0


def test_llm_unavailable_skips_edge(engine):
    """LLM returns None -> no edge created (no heuristic fallback)."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=None):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 0


def test_llm_disabled_skips_entirely(engine):
    """With llm_provider='none', LLM is never called and no edges are created."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "none"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    mock_llm.assert_not_called()

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 0


def test_checked_pairs_not_rechecked(engine):
    """Pairs already checked should not trigger a second LLM call on re-run."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    mock_llm = MagicMock(return_value=llm_response)
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    first_call_count = mock_llm.call_count
    assert first_call_count >= 1

    # Run again — the pair should be skipped
    mock_llm.reset_mock()
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)

    # LLM should not be called again for the same pair
    assert mock_llm.call_count == 0


def test_checked_pairs_recorded_for_none(engine):
    """Pairs classified as 'none' should be recorded in auto_link_checked."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    pair = tuple(sorted([id_a, id_b]))
    row = engine.db.conn.execute(
        "SELECT result FROM auto_link_checked WHERE node_a = ? AND node_b = ?",
        pair,
    ).fetchone()
    assert row is not None
    assert row["result"] == "none"


def test_max_nodes_per_run_default(engine):
    assert engine.settings.auto_link_max_nodes_per_run == 500


def test_seq_bumped_on_rewrite(engine):
    """Re-writing a node's content bumps its seq to the head (crit#2 mechanism)."""
    from ormah.models.node import UpdateNodeRequest
    id_a, id_b = _create_pair(engine)
    seq_before = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    max_before = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    engine.update_node(id_a, UpdateNodeRequest(content="rewritten content"))
    seq_after = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    assert seq_after > seq_before
    assert seq_after > max_before  # landed at the head


def test_metadata_update_does_not_bump_seq(engine):
    """A direct metadata UPDATE (not via the builder) must not change seq."""
    id_a, _ = _create_pair(engine)
    before = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    with engine.db.transaction() as conn:
        conn.execute("UPDATE nodes SET access_count = access_count + 1 WHERE id=?", (id_a,))
    after = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    assert after == before


def test_watermark_roundtrip(engine):
    from ormah.background.auto_linker import _get_watermark, _set_watermark
    assert _get_watermark(engine.db.conn) == 0
    _set_watermark(engine, 42)
    assert _get_watermark(engine.db.conn) == 42


def test_select_nodes_after_seq(engine):
    from ormah.background.auto_linker import _select_nodes_after
    id_a, id_b = _create_pair(engine)
    rows = _select_nodes_after(engine.db.conn, 0, limit=10)
    assert {id_a, id_b} <= {r["id"] for r in rows}
    last = rows[-1]
    rows2 = _select_nodes_after(engine.db.conn, last["seq"], limit=10)
    assert all(r["id"] != last["id"] for r in rows2)
    assert len(_select_nodes_after(engine.db.conn, 0, limit=1)) == 1


def test_run_advances_watermark(engine):
    from ormah.background.auto_linker import run_auto_linker, _get_watermark, _select_nodes_after
    _create_pair(engine)
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=json.dumps({"relationship": "none", "reason": "x"})):
        run_auto_linker(engine)
    last = _select_nodes_after(engine.db.conn, 0, limit=100)[-1]
    assert _get_watermark(engine.db.conn) == last["seq"]


def test_llm_none_does_not_advance_past_node(engine):
    """crit#1: a transient None must not let the watermark pass the node."""
    from ormah.background.auto_linker import run_auto_linker, _get_watermark
    _create_pair(engine)
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=None):
        run_auto_linker(engine)
    # No node fully resolved → watermark stays at 0
    assert _get_watermark(engine.db.conn) == 0
    # Next run with the LLM healthy re-evaluates the pair
    mock_llm = MagicMock(return_value=json.dumps({"relationship": "supports", "reason": "x"}))
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)
    assert mock_llm.call_count >= 1


def test_empty_vector_index_does_not_advance_watermark(engine):
    """Regression (#30): when node_vectors is empty/underfilled (e.g. mid full_rebuild,
    after vectors are deleted but before _reindex_all_embeddings restores them),
    vec_store.search returns no candidates. The watermark must NOT advance past those
    unchecked nodes, and once vectors are restored the pair must still be evaluated."""
    from ormah.background.auto_linker import run_auto_linker, _get_watermark

    id_a, id_b = _create_pair(engine)
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    # Simulate the rebuild window: vectors gone, not yet restored.
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    mock_llm = MagicMock(return_value=json.dumps({"relationship": "supports", "reason": "x"}))
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)

    # Nothing could be checked → no LLM call, watermark stays at 0, no edge.
    assert mock_llm.call_count == 0
    assert _get_watermark(engine.db.conn) == 0
    assert len(_edges_between(engine, id_a, id_b)) == 0

    # Vectors restored → the pair is finally evaluated and the watermark advances.
    engine._reindex_all_embeddings()
    mock_llm2 = MagicMock(return_value=json.dumps({"relationship": "supports", "reason": "x"}))
    with patch(_LLM_PATCH, mock_llm2):
        run_auto_linker(engine)
    assert mock_llm2.call_count >= 1
    assert _get_watermark(engine.db.conn) > 0


def test_max_edges_does_not_skip_interrupted_node(engine):
    """imp#4: max_edges mid-run must not advance the watermark past unprocessed nodes."""
    from ormah.background.auto_linker import run_auto_linker, _get_watermark, _select_nodes_after
    # three mutually-similar nodes
    _create_pair(engine, title_a="A", content_a="shared topic alpha", title_b="B", content_b="shared topic alpha beta")
    _create_pair(engine, title_a="C", content_a="shared topic alpha gamma", title_b="D", content_b="shared topic alpha delta")
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    engine.settings.auto_link_max_edges_per_run = 1
    _reset_adapter()
    rows = _select_nodes_after(engine.db.conn, 0, limit=100)
    with patch(_LLM_PATCH, return_value=json.dumps({"relationship": "supports", "reason": "x"})):
        run_auto_linker(engine)
    wm = _get_watermark(engine.db.conn)
    assert wm < rows[-1]["seq"]  # did not reach the last node


def test_full_rebuild_resets_watermark(engine):
    """A mass reindex must not leave a stale watermark hiding the whole store."""
    from ormah.background.auto_linker import _set_watermark, _get_watermark
    _create_pair(engine)
    _set_watermark(engine, 99999)
    engine.builder.full_rebuild()
    assert _get_watermark(engine.db.conn) == 0


def test_find_candidates_uses_window_without_advancing(engine):
    from ormah.background.auto_linker import _find_link_candidates, _get_watermark
    _create_pair(engine)
    engine.settings.auto_link_similarity_threshold = 0.0
    before = _get_watermark(engine.db.conn)
    cands = _find_link_candidates(engine, limit=8)
    assert all("node_a" in c and "node_b" in c and "similarity" in c for c in cands)
    assert _get_watermark(engine.db.conn) == before  # preview never advances the cursor


def test_invalid_llm_output_records_error_not_none(engine):
    """Malformed LLM JSON → recorded as result='error' (no edge), so the node resolves."""
    id_a, id_b = _create_pair(engine)
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    with patch(_LLM_PATCH, return_value="not valid json"):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)
    assert len(_edges_between(engine, id_a, id_b)) == 0  # no edge
    pair = tuple(sorted([id_a, id_b]))
    row = engine.db.conn.execute(
        "SELECT result FROM auto_link_checked WHERE node_a=? AND node_b=?", pair
    ).fetchone()
    assert row is not None and row["result"] == "error"


def test_checked_pairs_invalidated_on_update(engine):
    """Updating a node's content should clear its checked pairs so it gets re-evaluated."""
    from ormah.models.node import UpdateNodeRequest

    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    pair = tuple(sorted([id_a, id_b]))
    row = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert row is not None  # pair was recorded

    # Update node A's content
    engine.update_node(id_a, UpdateNodeRequest(content="Completely different content now"))

    # Checked pair should be cleared
    row = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert row is None  # pair invalidated

    # Next run should re-evaluate the pair
    mock_llm = MagicMock(return_value=json.dumps({
        "relationship": "supports",
        "reason": "Now they are related.",
    }))
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)

    assert mock_llm.call_count >= 1  # LLM was called again for this pair


def test_pairs_evaluated_counts_one_candidate_pair(engine):
    """Issue #90: pairs_evaluated must reflect exactly one LLM decision call.

    Uses the default similarity threshold (not 0.0): the auto-created "Self"
    node is far enough below threshold that it does not count as a second
    candidate, leaving exactly the id_a/id_b pair.
    """
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(
        "ormah.background.auto_linker._llm_classify_link",
        return_value={"relationship": "none", "reason": "not related"},
    ):
        from ormah.background.auto_linker import run_auto_linker
        stats = run_auto_linker(engine)

    assert stats["pairs_attempted"] == 1
    assert stats["pairs_evaluated"] == 1


def test_pairs_attempted_counts_llm_unavailable_pair_but_not_evaluated(engine):
    """Issue #90 (council finding 2): an LLM-unavailable pair (None decision)
    must count as attempted but NOT as evaluated — otherwise pairs_per_s is
    inflated by calls that never produced a decision."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(
        "ormah.background.auto_linker._llm_classify_link",
        return_value=None,
    ):
        from ormah.background.auto_linker import run_auto_linker
        stats = run_auto_linker(engine)

    assert stats["pairs_attempted"] == 1
    assert stats["pairs_evaluated"] == 0


def test_pairs_attempted_counts_invalid_llm_output_but_not_evaluated(engine):
    """Issue #90 council R2 finding 2: the 'error' sentinel (invalid/malformed
    LLM output) must count as attempted but NOT as a valid evaluation — a
    degraded provider must not report healthy pairs_per_s with zero real
    decisions. duplicate_merger/conflict_detector already exclude their
    equivalent None-decision case; auto_linker's sentinel is a dict, not
    None, so it needs its own check."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(
        "ormah.background.auto_linker._llm_classify_link",
        return_value={"relationship": "error", "reason": "invalid LLM output"},
    ):
        from ormah.background.auto_linker import run_auto_linker
        stats = run_auto_linker(engine)

    assert stats["pairs_attempted"] == 1
    assert stats["pairs_evaluated"] == 0
