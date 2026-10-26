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


def test_apply_edge_is_idempotent_when_edge_already_exists(engine):
    """A concurrent writer created the same edge between collection and apply.
    _apply_edge must not raise, and must still record the pair as checked."""
    from datetime import datetime, timezone
    from ormah.background.auto_linker import _apply_edge

    id_a, id_b = _create_pair(engine)
    now = datetime.now(timezone.utc).isoformat()
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
            "VALUES (?, ?, 'supports', 0.9, ?, 'created by someone else')",
            (id_a, id_b, now),
        )

    _apply_edge(engine, id_a, id_b, "supports", "auto-linker reason", 0.8)

    # The pre-existing edge survives untouched; no duplicate was created.
    rows = engine.db.conn.execute(
        "SELECT reason FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_b),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["reason"] == "created by someone else"

    # The pair is marked checked -> it will never be re-judged. This is exactly what
    # the rollback used to erase, which is why the pair poisoned every future run.
    pair = tuple(sorted([id_a, id_b]))
    assert engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone() is not None


def test_apply_edge_does_not_duplicate_the_markdown_connection(engine):
    """The winner of the race already wrote its Connection to the file. We must not
    append a second one for the same (target, edge)."""
    from datetime import datetime, timezone
    from ormah.models.node import Connection, EdgeType
    from ormah.background.auto_linker import _apply_edge

    id_a, id_b = _create_pair(engine)
    now = datetime.now(timezone.utc).isoformat()
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
            "VALUES (?, ?, 'supports', 0.9, ?, 'x')",
            (id_a, id_b, now),
        )
    node = engine.file_store.load(id_a)          # the winner persisted its markdown
    node.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.9))
    engine.file_store.save(node)

    _apply_edge(engine, id_a, id_b, "supports", "reason", 0.8)

    node = engine.file_store.load(id_a)
    assert len([c for c in node.connections if c.target == id_b]) == 1


def test_apply_edge_repairs_a_markdown_connection_the_winner_failed_to_save(engine):
    """The winner committed the DB row but crashed before saving its markdown. The
    file is the source of truth and a reindex rebuilds edges from it — so if we skip
    the append just because we lost the race, the next reindex deletes the edge while
    auto_link_checked stops the pair from ever being reconsidered. The link would be
    lost forever. We must repair the file instead. (Codex R1, critical #1.)"""
    from datetime import datetime, timezone
    from ormah.background.auto_linker import _apply_edge

    id_a, id_b = _create_pair(engine)
    now = datetime.now(timezone.utc).isoformat()
    with engine.db.transaction() as conn:        # DB row exists, markdown does NOT
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
            "VALUES (?, ?, 'supports', 0.9, ?, 'winner crashed before saving md')",
            (id_a, id_b, now),
        )
    assert [c for c in engine.file_store.load(id_a).connections if c.target == id_b] == []

    _apply_edge(engine, id_a, id_b, "supports", "reason", 0.8)

    conns = [c for c in engine.file_store.load(id_a).connections if c.target == id_b]
    assert len(conns) == 1
    assert conns[0].edge.value == "supports"


def test_run_survives_an_edge_apply_failure(engine, monkeypatch):
    """A pair whose edge write blows up must not abort the whole run."""
    import json
    from unittest.mock import patch
    from ormah.background import auto_linker as al

    _create_pair(engine)
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0

    def boom(*_args, **_kwargs):
        raise RuntimeError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(al, "_apply_edge", boom)

    llm_response = json.dumps({"relationship": "supports", "reason": "r"})
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=llm_response):
        al.run_auto_linker(engine)   # must return normally, not raise

    # Fail closed: the watermark must NOT have advanced past the unresolved node.
    watermark = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'auto_link_watermark'"
    ).fetchone()
    assert watermark is None or int(watermark["value"]) == 0


def test_a_failing_pair_does_not_block_progress_on_earlier_nodes(engine, monkeypatch):
    """Progress, not just survival (Codex R1, critical #2): the failing pair parks the
    cursor AT that node, but every node before it still advances the watermark. Without
    this, the fix would only be swapping one kind of total stall for another."""
    import json
    from unittest.mock import patch
    from ormah.background import auto_linker as al

    # A first pair that links cleanly, then a second pair whose apply always fails.
    good_a, good_b = _create_pair(engine)
    bad_a, bad_b = _create_pair(
        engine, title_a="Rust language", content_a="Rust is a systems language.",
        title_b="Rust lang", content_b="Rust is a popular systems language.",
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0

    real_apply = al._apply_edge

    def apply_or_boom(eng, a_id, b_id, *args, **kwargs):
        if a_id in (bad_a, bad_b):
            raise RuntimeError("FOREIGN KEY constraint failed")
        return real_apply(eng, a_id, b_id, *args, **kwargs)

    monkeypatch.setattr(al, "_apply_edge", apply_or_boom)

    good_seq = engine.db.conn.execute(
        "SELECT seq FROM nodes WHERE id = ?", (good_b,)
    ).fetchone()["seq"]

    llm_response = json.dumps({"relationship": "supports", "reason": "r"})
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=llm_response):
        al.run_auto_linker(engine)

    row = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'auto_link_watermark'"
    ).fetchone()
    assert row is not None, "the run made no progress at all — the failing pair stalled everything"
    assert int(row["value"]) >= good_seq, "the clean nodes before the failing pair must advance"


def test_apply_edge_reports_whether_it_actually_created_the_edge(engine):
    """An INSERT OR IGNORE that inserted nothing is not a creation. Counting it as one
    burns the run's edge budget on a link someone else already made, and logs a
    creation that never happened (Codex R2, medium)."""
    from datetime import datetime, timezone
    from ormah.background.auto_linker import _apply_edge

    id_a, id_b = _create_pair(engine)

    assert _apply_edge(engine, id_a, id_b, "supports", "r", 0.8) is True   # new edge

    # Same edge again: a concurrent writer already has it -> ignored, not created.
    now = datetime.now(timezone.utc).isoformat()
    assert now  # keep the import honest
    assert _apply_edge(engine, id_a, id_b, "supports", "r", 0.8) is False

    # 'none' records the pair as checked without ever creating an edge.
    id_c, id_d = _create_pair(
        engine, title_a="Go language", content_a="Go is a systems language.",
        title_b="Go lang", content_b="Go is a popular systems language.",
    )
    assert _apply_edge(engine, id_c, id_d, "none", "", 0.0) is False


def test_a_failed_markdown_save_does_not_leave_the_pair_marked_checked(engine, monkeypatch):
    """The markdown is the source of truth: a rebuild recreates the edge table from it.
    If the connection cannot be persisted, the pair must NOT stay marked as checked —
    otherwise the rebuild drops the DB-only edge and the checked row stops the pair from
    ever being judged again. The link would be lost for good (Codex, PR A round 2)."""
    import pytest
    from ormah.background.auto_linker import _apply_edge

    id_a, id_b = _create_pair(engine)

    def boom(_node):
        raise OSError("disk full")

    monkeypatch.setattr(engine.file_store, "save", boom)

    with pytest.raises(OSError):
        _apply_edge(engine, id_a, id_b, "supports", "r", 0.8)

    pair = tuple(sorted([id_a, id_b]))
    assert engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone() is None, "the pair stayed checked, so it will never be judged again"

    # The edge WE inserted is rolled back too, or the collection guard would skip the
    # pair forever on the strength of a row whose markdown never existed.
    assert engine.db.conn.execute(
        "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?", (id_a, id_b)
    ).fetchone() is None
