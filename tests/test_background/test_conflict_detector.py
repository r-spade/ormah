"""Tests for LLM-based contradiction detection in conflict_detector."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from ormah.models.node import CreateNodeRequest, NodeType

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _create_pair(engine, title_a="Use PostgreSQL", content_a="We decided to use PostgreSQL for the database.",
                 title_b="Use MySQL", content_b="We decided to use MySQL for the database.",
                 node_type=NodeType.decision):
    """Helper: create two similar nodes without auto-linking, return their IDs."""
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


def _reset_adapter():
    from ormah.background.llm_client import reset_adapter
    reset_adapter()


def test_llm_detects_evolution_creates_evolved_from_edge(engine):
    """LLM detects belief evolution -> evolved_from edge created, no proposal."""
    id_a, id_b = _create_pair(
        engine,
        title_a="Dislikes grapes",
        content_a="The user hates grapes.",
        title_b="Loves red grapes",
        content_b="The user loves red grapes but hates green grapes.",
        node_type=NodeType.preference,
    )

    # ORDER BY RANDOM() means node_a/node_b ordering is non-deterministic.
    # Use side_effect to return evolved_node="b" when "Loves red grapes" is
    # presented as Memory B, and evolved_node="a" otherwise — so the
    # direction assertion is always semantically correct.
    def dynamic_llm_response(settings, prompt, json_mode=True):
        parts = prompt.split("\nMemory B")
        evolved = "b" if len(parts) > 1 and "Loves red grapes" in parts[1] else "a"
        return json.dumps({
            "same_subject": True,
            "conflict": True,
            "type": "evolution",
            "evolved_node": evolved,
            "explanation": "Refined from blanket dislike to nuanced preference by grape type.",
        })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, side_effect=dynamic_llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # Should create an evolved_from edge, not a proposal
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type = 'evolved_from'"
    ).fetchall()
    assert len(edges) >= 1
    edge = edges[0]
    assert edge["source_id"] == id_b  # newer (evolved) node
    assert edge["target_id"] == id_a  # older node
    assert "Refined from blanket dislike" in edge["reason"]

    # No proposals should be created
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_detects_tension_creates_contradicts_edge(engine):
    """LLM detects genuine tension -> contradicts edge created with reason, no proposal."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": True,
        "type": "tension",
        "explanation": "Cannot use both PostgreSQL and MySQL as the primary database.",
    })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # Should create a contradicts edge with reason
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    assert len(edges) >= 1
    edge = edges[0]
    assert "Cannot use both PostgreSQL and MySQL" in edge["reason"]

    # No proposals should be created
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_rejects_contradiction_no_edge(engine):
    """LLM rejects contradiction -> no edge, no proposal."""
    id_a, id_b = _create_pair(
        engine,
        title_a="Use PostgreSQL",
        content_a="We decided to use PostgreSQL for the database.",
        title_b="PostgreSQL config",
        content_b="PostgreSQL should be configured with connection pooling.",
        node_type=NodeType.fact,
    )

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": False,
        "type": "none",
        "explanation": "These are complementary — one is a decision, the other is a configuration detail.",
    })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # No edges of conflict type
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type IN ('contradicts', 'evolved_from')"
    ).fetchall()
    assert len(edges) == 0

    # No proposals
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_unavailable_skips_pair(engine):
    """LLM returns None -> pair is skipped, no proposals created."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=None):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # No proposals or edges should be created
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0

    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type IN ('contradicts', 'evolved_from')"
    ).fetchall()
    assert len(edges) == 0


def test_llm_disabled_skips_detection(engine):
    """With llm_provider='none', LLM is never called and no proposals created."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "none"
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    mock_llm.assert_not_called()

    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_project_scoped_nodes_checked_when_flag_enabled(engine):
    """With conflict_check_all_spaces=True, project-scoped nodes are checked."""
    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = engine.remember(
            CreateNodeRequest(
                content="We decided to use PostgreSQL for the database.",
                type=NodeType.fact,
                title="Use PostgreSQL",
                space="myproject",
            ),
            agent_id="test",
        )
        id_b, _ = engine.remember(
            CreateNodeRequest(
                content="We decided to use MySQL for the database.",
                type=NodeType.fact,
                title="Use MySQL",
                space="myproject",
            ),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": True,
        "type": "tension",
        "explanation": "Cannot use both PostgreSQL and MySQL as the primary database.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.conflict_check_all_spaces = True
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    assert len(edges) >= 1


def test_project_scoped_nodes_skipped_by_default(engine):
    """By default (conflict_check_all_spaces=False), project-scoped nodes are not checked."""
    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        engine.remember(
            CreateNodeRequest(
                content="We decided to use PostgreSQL for the database.",
                type=NodeType.fact,
                title="Use PostgreSQL",
                space="myproject",
            ),
            agent_id="test",
        )
        engine.remember(
            CreateNodeRequest(
                content="We decided to use MySQL for the database.",
                type=NodeType.fact,
                title="Use MySQL",
                space="myproject",
            ),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    engine.settings.llm_provider = "ollama"
    engine.settings.conflict_check_all_spaces = False
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # LLM should never be called since project-scoped nodes are skipped
    mock_llm.assert_not_called()


# --- #81 delta-selection ---

def _make_belief(engine, title, content):
    """Create a belief-type node without auto-linking; return (id, seq)."""
    original = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        node_id, _ = engine.remember(
            CreateNodeRequest(content=content, type=NodeType.fact, title=title, tags=["test"]),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original
    seq = engine.db.conn.execute("SELECT seq FROM nodes WHERE id = ?", (node_id,)).fetchone()["seq"]
    return node_id, seq


def test_delta_finder_skips_seeds_at_or_below_watermark(engine):
    from ormah.background.conflict_detector import _find_conflict_candidates
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, set_watermark

    _, seq_a = _make_belief(engine, "Coffee is healthy", "Coffee improves focus and health.")
    _make_belief(engine, "Coffee is unhealthy", "Coffee harms sleep and health.")

    # Watermark past ALL nodes -> no seeds -> no candidates
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    set_watermark(engine, CONFLICT_WATERMARK_KEY, max_seq)
    candidates, seeds = _find_conflict_candidates(engine, limit=100, delta=True)
    assert candidates == [] and seeds == []

    # Watermark below the pair -> pair found again
    set_watermark(engine, CONFLICT_WATERMARK_KEY, seq_a - 1)
    candidates, _ = _find_conflict_candidates(engine, limit=100, delta=True)
    assert len(candidates) >= 1


def test_legacy_mode_ignores_watermark(engine):
    """Default call (agent path) keeps today's selection: nodes below the
    watermark are still reachable and the return shape is a plain list."""
    from ormah.background.conflict_detector import _find_conflict_candidates
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, set_watermark

    _make_belief(engine, "Milk is good", "Milk strengthens bones at any age.")
    _make_belief(engine, "Milk is bad", "Milk weakens bones at any age.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    set_watermark(engine, CONFLICT_WATERMARK_KEY, max_seq)

    candidates = _find_conflict_candidates(engine, limit=100)  # no delta kwarg
    assert isinstance(candidates, list)
    assert len(candidates) >= 1


def test_new_seed_pairs_with_old_neighbor(engine):
    """Neighbors are age-unfiltered: an OLD node below the watermark is still
    reachable as the neighbor of a NEW seed."""
    from ormah.background.conflict_detector import _find_conflict_candidates
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, set_watermark

    old_id, old_seq = _make_belief(engine, "Tabs are best", "The project uses tabs for indentation.")
    set_watermark(engine, CONFLICT_WATERMARK_KEY, old_seq)  # old node is below the cursor

    new_id, _ = _make_belief(engine, "Spaces are best", "The project uses spaces for indentation.")

    candidates, _ = _find_conflict_candidates(engine, limit=100, delta=True)
    pair_ids = {(c["node_a"]["id"], c["node_b"]["id"]) for c in candidates}
    assert any(old_id in p and new_id in p for p in pair_ids)


def test_finder_respects_max_seeds_and_seq_order(engine):
    from ormah.background.conflict_detector import _find_conflict_candidates

    ids = [_make_belief(engine, f"Fact {i}", f"The sky color observation number {i} is blue.")
           for i in range(3)]
    candidates, seeds = _find_conflict_candidates(engine, limit=100, max_seeds=2, delta=True)
    # Only the 2 lowest-seq nodes were seeds, in ascending order
    assert [s[0] for s in seeds] == [ids[0][0], ids[1][0]]
    assert [s[1] for s in seeds] == sorted(s[1] for s in seeds)
    for c in candidates:
        assert c["seed_seq"] in {ids[0][1], ids[1][1]}


def test_finder_never_advances_watermark(engine):
    """Agent path calls the finder directly; the cursor must not move."""
    from ormah.background.conflict_detector import _find_conflict_candidates
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark

    _make_belief(engine, "Bikes are green", "Cycling is an eco-friendly transport choice.")
    _find_conflict_candidates(engine, limit=8)              # legacy mode
    _find_conflict_candidates(engine, limit=8, delta=True)  # delta mode
    assert get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY) == 0


def test_empty_vector_index_does_not_advance_conflict_selection(engine):
    """Fail-closed (overview invariant): a seed with text but NO persisted
    vector must not drain — mirrors test_empty_vector_index_does_not_advance_watermark
    in test_auto_linker.py."""
    from ormah.background.conflict_detector import _find_conflict_candidates

    node_id, seq = _make_belief(engine, "Vectorless claim", "A statement whose vector is missing.")
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")  # simulate rebuild-before-backfill window

    _, seeds = _find_conflict_candidates(engine, limit=100, delta=True)
    assert (node_id, seq) not in seeds  # not drained -> cursor cannot pass it


def test_seedless_nodes_are_still_drained(engine):
    """A seed whose pairs are all prefiltered still appears in the drained list
    (it must not block the cursor)."""
    from ormah.background.conflict_detector import _find_conflict_candidates

    node_id, seq = _make_belief(engine, "Lone fact", "A completely unrelated singleton statement.")
    _, seeds = _find_conflict_candidates(engine, limit=100, delta=True)
    assert (node_id, seq) in seeds


def test_scope_toggle_resets_delta_selection(engine):
    """Nodes ingested while conflict_check_all_spaces was OFF must become
    reachable when it turns ON, even if the cursor already passed them."""
    from ormah.background.conflict_detector import _find_conflict_candidates
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, set_watermark

    engine.settings.conflict_check_all_spaces = False
    node_id, seq = _make_belief(engine, "Global claim", "A plain global-space statement.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    set_watermark(engine, CONFLICT_WATERMARK_KEY, max_seq)
    with engine.db.transaction() as conn:  # stamp as if advanced under scope=global
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES "
            "('conflict_check_watermark_scope', 'global')"
        )

    engine.settings.conflict_check_all_spaces = True  # operator flips the flag
    _, seeds = _find_conflict_candidates(engine, limit=100, delta=True)
    assert (node_id, seq) in seeds  # stamp mismatch -> watermark treated as 0


def test_scope_toggle_run_persists_watermark_reset_on_vectorless_barrier(engine):
    """A scope flip must persist the watermark reset during THIS run, even if
    the run itself drains nothing (vectorless barrier) — otherwise the reset
    is lost and the next run's stamp already matches (#81 regression)."""
    from ormah.background.conflict_detector import (
        CONFLICT_SCOPE_STAMP_KEY, run_conflict_detection,
    )
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark, set_watermark

    engine.settings.conflict_check_all_spaces = False
    lowest_id, lowest_seq = _make_belief(engine, "Global claim", "A plain global-space statement.")
    _make_belief(engine, "Another global claim", "A second plain global-space statement.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    # Simulate: already advanced under scope=global
    set_watermark(engine, CONFLICT_WATERMARK_KEY, max_seq)
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (CONFLICT_SCOPE_STAMP_KEY, "global"),
        )

    engine.settings.conflict_check_all_spaces = True  # operator flips the flag
    # Vectorless barrier: the first newly-eligible seed has no vector
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (lowest_id,))

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_conflict_response()):
        run_conflict_detection(engine)

    # Nothing drained this run, but the reset must persist for next run.
    assert get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY) == 0


def _conflict_response():
    return json.dumps({
        "conflict": True, "same_subject": True, "relationship": "tension",
        "reason": "Opposing claims.",
    })


def test_clean_run_advances_watermark_past_all_seeds(engine):
    from ormah.background.conflict_detector import run_conflict_detection
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark

    _make_belief(engine, "Tea is calming", "Tea makes the user calm in the evening.")
    _make_belief(engine, "Tea is agitating", "Tea makes the user agitated in the evening.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_conflict_response()):
        run_conflict_detection(engine)

    assert get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY) == max_seq


def test_llm_failure_parks_watermark_before_failed_seed(engine):
    """Seed A succeeds, seed B's LLM check returns None -> cursor stops at A;
    the next run re-selects B."""
    from ormah.background.conflict_detector import (
        _find_conflict_candidates, run_conflict_detection,
    )
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark

    _make_belief(engine, "Cats are aloof", "The cat ignores everyone at home.")
    _make_belief(engine, "Cats are clingy", "The cat follows everyone at home.")
    b_id, b_seq = _make_belief(engine, "Dogs bark a lot", "The dog barks at everything.")
    _make_belief(engine, "Dogs are silent", "The dog never barks at anything.")

    def llm_fails_for_b(settings, prompt, *args, **kwargs):
        if "barks" in prompt:
            return None          # seed involving the dog pair fails
        return _conflict_response()

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, side_effect=llm_fails_for_b):
        run_conflict_detection(engine)

    wm = get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY)
    assert wm < b_seq  # cursor did not pass the failed seed

    # next run re-selects the failed seed
    candidates = _find_conflict_candidates(engine, limit=100)
    assert any(b_id in (c["node_a"]["id"], c["node_b"]["id"]) for c in candidates)


def test_conflict_run_llm_disabled_does_not_advance_watermark(engine):
    """The llm_enabled guard must run BEFORE selection: a disabled-LLM run
    must not move the cursor (guard-reorder regression trap)."""
    from ormah.background.conflict_detector import run_conflict_detection
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark

    _make_belief(engine, "Any claim", "A statement that would otherwise be a seed.")
    engine.settings.llm_provider = "none"
    _reset_adapter()
    run_conflict_detection(engine)
    assert get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY) == 0


def test_conflict_run_vectorless_seed_blocks_watermark(engine):
    """A vectorless seed must be a barrier: no later seed may drain past it,
    or the watermark jumps a hole and the vectorless seed's pairs are never
    re-checked once vectors are restored (#81 regression)."""
    from ormah.background.conflict_detector import run_conflict_detection
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark

    id_a, seq_a = _make_belief(engine, "Vectorless claim", "A statement whose vector went missing.")
    id_b, seq_b = _make_belief(engine, "Second claim", "A second, unrelated statement.")
    assert seq_a < seq_b

    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (id_a,))  # only the LOWER-seq seed loses its vector

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_conflict_response()):
        run_conflict_detection(engine)

    # The vectorless seed (seq_a) is the lowest-seq selected node: with the
    # `break` fix the finder stops there and drains nothing, so the
    # watermark must stay at 0 (never jump the hole to seq_b).
    assert get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY) == 0


def test_run_with_no_new_nodes_is_a_noop(engine):
    from ormah.background.conflict_detector import run_conflict_detection
    from ormah.background.watermark import CONFLICT_WATERMARK_KEY, get_watermark, set_watermark

    _make_belief(engine, "Solo fact", "One isolated statement about nothing else.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    set_watermark(engine, CONFLICT_WATERMARK_KEY, max_seq)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    llm = MagicMock(return_value=_conflict_response())
    with patch(_LLM_PATCH, llm):
        run_conflict_detection(engine)

    llm.assert_not_called()
    assert get_watermark(engine.db.conn, CONFLICT_WATERMARK_KEY) == max_seq
