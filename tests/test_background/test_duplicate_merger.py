"""Tests for LLM-based duplicate consolidation in duplicate_merger."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch, MagicMock

from ormah.models.node import CreateNodeRequest, NodeType

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _create_pair(engine, title_a="Python language", content_a="Python is a programming language.",
                 title_b="Python lang", content_b="Python is a popular programming language.",
                 node_type=NodeType.fact):
    """Helper: create two similar nodes and return their IDs."""
    id_a, _ = engine.remember(
        CreateNodeRequest(content=content_a, type=node_type, title=title_a, tags=["test"]),
        agent_id="test",
    )
    id_b, _ = engine.remember(
        CreateNodeRequest(content=content_b, type=node_type, title=title_b, tags=["test"]),
        agent_id="test",
    )
    return id_a, id_b


def _reset_adapter():
    from ormah.background.llm_client import reset_adapter
    reset_adapter()


def test_llm_confirms_duplicate_auto_merge(engine):
    """LLM confirms duplicate -> auto-merge with merged content."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "is_duplicate": True,
        "merged_title": "Python Programming Language",
        "merged_content": "Python is a popular programming language used widely.",
        "reason": "Both describe Python as a programming language.",
    })

    # Force auto-merge threshold low so the pair qualifies
    engine.settings.auto_merge_threshold = 0.0
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    # One of the two nodes should have been removed; the kept one should
    # have the LLM-generated content.
    kept = engine.file_store.load(id_a) or engine.file_store.load(id_b)
    assert kept is not None
    assert kept.content == "Python is a popular programming language used widely."
    assert kept.title == "Python Programming Language"


def test_llm_rejects_duplicate_no_merge(engine):
    """LLM rejects duplicate -> no merge or proposal despite high composite score."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "is_duplicate": False,
        "merged_title": "",
        "merged_content": "",
        "reason": "These describe different aspects of Python.",
    })

    engine.settings.auto_merge_threshold = 0.0
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    # Both nodes should still exist
    assert engine.file_store.load(id_a) is not None
    assert engine.file_store.load(id_b) is not None


def test_llm_unavailable_skips_merge(engine):
    """LLM returns None -> pair is skipped, both nodes survive, no proposals."""
    id_a, id_b = _create_pair(engine)

    engine.settings.auto_merge_threshold = 0.0
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=None):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    # Both nodes should still exist
    assert engine.file_store.load(id_a) is not None
    assert engine.file_store.load(id_b) is not None

    # No proposals
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'merge' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_disabled_skips_detection(engine):
    """With llm_provider='none', LLM is never called."""
    id_a, id_b = _create_pair(engine)

    engine.settings.auto_merge_threshold = 0.0
    engine.settings.llm_provider = "none"
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    mock_llm.assert_not_called()


def test_merged_content_stored_in_proposal(engine):
    """For medium-confidence pairs, proposal contains merged content preview."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "is_duplicate": True,
        "merged_title": "Python Programming Language",
        "merged_content": "Python is a popular programming language used widely.",
        "reason": "Both describe Python as a programming language.",
    })

    # Set threshold high so pair goes to proposal instead of auto-merge
    engine.settings.auto_merge_threshold = 0.99
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    # Both nodes should still exist (no auto-merge)
    assert engine.file_store.load(id_a) is not None
    assert engine.file_store.load(id_b) is not None

    # A proposal should have been created with merged content preview
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'merge' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) >= 1

    proposal = proposals[0]
    assert "Merged content preview:" in proposal["proposed_action"]
    assert "Python Programming Language" in proposal["proposed_action"]
    assert "Python is a popular programming language used widely." in proposal["proposed_action"]
    assert "Both describe Python" in proposal["reason"]


# --- #81 delta-selection ---

def _make_fact(engine, title, content):
    """Create a node without auto-linking; return (id, seq)."""
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


def test_dedup_finder_skips_seeds_at_or_below_watermark(engine):
    from ormah.background.duplicate_merger import _find_merge_candidates
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, set_watermark

    _make_fact(engine, "Python is dynamic", "Python is a dynamically typed language.")
    _make_fact(engine, "Python typing", "Python is a dynamically typed programming language.")

    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    set_watermark(engine, DUPLICATE_WATERMARK_KEY, max_seq)
    candidates, seeds = _find_merge_candidates(engine, limit=100, delta=True)
    assert candidates == [] and seeds == []
    # legacy mode (agent path) ignores the watermark entirely
    legacy = _find_merge_candidates(engine, limit=100)
    assert isinstance(legacy, list) and len(legacy) >= 1


def test_dedup_new_seed_pairs_with_old_neighbor(engine):
    from ormah.background.duplicate_merger import _find_merge_candidates
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, set_watermark

    old_id, old_seq = _make_fact(engine, "Server port", "The ormah server listens on port 8787.")
    set_watermark(engine, DUPLICATE_WATERMARK_KEY, old_seq)

    new_id, _ = _make_fact(engine, "Ormah port", "The ormah server runs on port 8787.")

    candidates, _ = _find_merge_candidates(engine, limit=100, delta=True)
    pair_ids = {(c["node_a"]["id"], c["node_b"]["id"]) for c in candidates}
    assert any(old_id in p and new_id in p for p in pair_ids)


def test_empty_vector_index_does_not_drain_dedup_seeds(engine):
    """Fail-closed (overview invariant): seed with text but no persisted
    vector must not drain (empty/backfilling node_vectors window)."""
    from ormah.background.duplicate_merger import _find_merge_candidates

    node_id, seq = _make_fact(engine, "Vectorless note", "A note whose vector is missing.")
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    _, seeds = _find_merge_candidates(engine, limit=100, delta=True)
    assert (node_id, seq) not in seeds


def test_dedup_finder_delta_reports_drained_in_seq_order(engine):
    from ormah.background.duplicate_merger import _find_merge_candidates

    made = [_make_fact(engine, f"Note {i}", f"Unrelated singleton note number {i}.")
            for i in range(3)]
    _, seeds = _find_merge_candidates(engine, limit=100, delta=True)
    seed_ids = [s[0] for s in seeds]
    for node_id, _seq in made:
        assert node_id in seed_ids  # zero-candidate seeds still drained
    assert [s[1] for s in seeds] == sorted(s[1] for s in seeds)


def test_dedup_barrier_logs_once_per_run(engine, caplog):
    """The vectorless drain barrier warns once per run, not once per seed."""
    from ormah.background.duplicate_merger import _find_merge_candidates

    id_a, _ = _make_fact(engine, "Vectorless note A", "First note whose vector is missing.")
    id_b, _ = _make_fact(engine, "Vectorless note B", "Second note whose vector is missing.")
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id IN (?, ?)", (id_a, id_b))

    with caplog.at_level(logging.WARNING):
        _find_merge_candidates(engine, limit=100, delta=True)

    matches = [r for r in caplog.records if "no persisted vector" in r.message]
    assert len(matches) == 1


def _duplicate_response():
    return json.dumps({
        "is_duplicate": True,
        "merged_title": "Merged fact",
        "merged_content": "The merged content.",
        "reason": "Same statement.",
    })


def test_run_does_not_rejudge_pair_below_watermark(engine):
    """Reproduces #81: with the cursor past both nodes, a run must not spend
    LLM calls on them again."""
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark, set_watermark

    _make_fact(engine, "Editor choice", "The user edits everything in neovim.")
    _make_fact(engine, "Editor pick", "The user does all editing in neovim.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    set_watermark(engine, DUPLICATE_WATERMARK_KEY, max_seq)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    llm = MagicMock(return_value=_duplicate_response())
    with patch(_LLM_PATCH, llm):
        run_duplicate_detection(engine)

    llm.assert_not_called()
    assert get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY) == max_seq


def test_run_creates_proposal_for_delta_pair_and_advances(engine):
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

    engine.settings.auto_merge_threshold = 999.0  # force proposal path, not auto-merge
    _make_fact(engine, "Backup time", "Backups run every night at 2am.")
    _make_fact(engine, "Backup schedule", "The backup runs nightly at 2am.")
    max_seq = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_duplicate_response()):
        run_duplicate_detection(engine)

    proposals = engine.db.conn.execute(
        "SELECT 1 FROM proposals WHERE type = 'merge' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) >= 1
    assert get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY) == max_seq


def test_run_judges_full_content_not_400_char_preview(engine):
    """The LLM must receive the node's untruncated row (merge safety, parity
    with today's run), not the finder's 400-char preview. The marker sits
    beyond 400 chars but under _llm_check_duplicate's own pre-existing
    2000-char ceiling."""
    from ormah.background.duplicate_merger import run_duplicate_detection

    marker = "UNIQUE-TAIL-MARKER-9137"
    long_content = "The deploy procedure is documented step by step. " * 12 + marker
    _make_fact(engine, "Deploy procedure", long_content)
    _make_fact(engine, "Deployment steps", long_content.replace("documented", "written"))
    assert len(long_content) > 400

    seen_prompts: list[str] = []

    # NOTE: _llm_check_duplicate calls llm_generate(settings, prompt, json_mode=True),
    # so the mock MUST take `settings` FIRST — otherwise settings lands in `prompt`
    # and `marker in p` silently reads False (bug caught in Task 3's analogous mock).
    def capture(settings, prompt, *args, **kwargs):
        seen_prompts.append(prompt)
        return json.dumps({"is_duplicate": False, "reason": "distinct"})

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, side_effect=capture):
        run_duplicate_detection(engine)

    assert seen_prompts, "expected at least one LLM call for the near-duplicate pair"
    assert any(marker in p for p in seen_prompts)


def test_run_llm_failure_parks_dedup_watermark_exactly(engine):
    """A clean seed batch BEFORE the failing pair advances; the cursor stops
    exactly at the last clean seed before the failure (no `or wm == 0`
    escape hatch — the advance must be exact)."""
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

    # unrelated singleton first: a clean, candidate-less seed with low seq
    _, clean_seq = _make_fact(engine, "Lone note", "A singleton note about nothing similar.")
    # then the near-duplicate pair whose LLM check will fail
    _, pair_seq_a = _make_fact(engine, "Coffee dose", "The user drinks two espressos daily.")
    _make_fact(engine, "Espresso habit", "The user has two espressos every day.")

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=None):  # LLM unavailable for every pair
        run_duplicate_detection(engine)

    wm = get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY)
    assert wm >= clean_seq      # clean prefix advanced
    assert wm < pair_seq_a      # cursor parked before the failed seed


def test_dedup_run_llm_disabled_does_not_advance_watermark(engine):
    """Guard order: `if not settings.llm_enabled: return` fires BEFORE any
    selection or advance."""
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

    _make_fact(engine, "Any note", "A note that would otherwise be a seed.")
    engine.settings.llm_provider = "none"
    _reset_adapter()
    run_duplicate_detection(engine)
    assert get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY) == 0


def test_dedup_run_vectorless_seed_blocks_watermark(engine):
    """A vectorless seed must be a barrier: no later seed may drain past it,
    or the watermark jumps a hole and the vectorless seed's pairs are never
    re-checked once vectors are restored (#81 regression)."""
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

    id_a, seq_a = _make_fact(engine, "Vectorless note", "A note whose vector went missing.")
    id_b, seq_b = _make_fact(engine, "Second note", "A second, unrelated note.")
    assert seq_a < seq_b

    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (id_a,))  # only the LOWER-seq seed loses its vector

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_duplicate_response()):
        run_duplicate_detection(engine)

    # With the `break` fix the finder stops at the vectorless barrier
    # (seq_a): the cursor may advance past legitimately-drained seeds before
    # it (e.g. the engine's own user_node), but must never reach seq_a or
    # jump the hole to seq_b.
    assert get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY) < seq_a


def test_dedup_run_judges_pair_already_in_auto_link_checked(engine):
    """Background dedup must not skip a pair merely because auto_linker
    already recorded a link decision for it (#81 regression: auto_link_checked
    is a LINK-decision log, not a dedup-skip list)."""
    from datetime import datetime, timezone

    from ormah.background.duplicate_merger import run_duplicate_detection

    engine.settings.auto_merge_threshold = 999.0  # force proposal path
    id_a, _ = _make_fact(engine, "Editor choice", "The user edits everything in neovim.")
    id_b, _ = _make_fact(engine, "Editor pick", "The user does all editing in neovim.")

    pair = tuple(sorted([id_a, id_b]))
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, ?, ?)",
            (*pair, "none", datetime.now(timezone.utc).isoformat()),
        )

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_duplicate_response()):
        run_duplicate_detection(engine)

    proposals = engine.db.conn.execute(
        "SELECT 1 FROM proposals WHERE type = 'merge' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) >= 1


def test_dedup_run_processes_seeds_after_vectorless_barrier(engine):
    """A vectorless barrier parks the cursor but must not stop later seeds
    from being judged (liveness, mirrors upstream auto_linker)."""
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

    engine.settings.auto_merge_threshold = 999.0  # force proposal path
    barrier_id, barrier_seq = _make_fact(engine, "Vectorless note", "A note whose vector went missing.")
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (barrier_id,))

    id_a, _ = _make_fact(engine, "Editor choice", "The user edits everything in neovim.")
    id_b, _ = _make_fact(engine, "Editor pick", "The user does all editing in neovim.")

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_duplicate_response()):
        run_duplicate_detection(engine)

    proposals = engine.db.conn.execute(
        "SELECT 1 FROM proposals WHERE type = 'merge' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) >= 1, "later seeds past the barrier must still be judged"

    wm = get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY)
    assert wm < barrier_seq  # cursor still parked before the barrier


def test_auto_merge_survivor_requeues_into_delta(engine):
    """When a pair auto-merges mid-run, the survivor's content rewrite
    allocates a fresh seq (see test_seq_bumped_on_rewrite), so it re-enters
    the delta on the next run — skipping its stale pairs loses no work."""
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

    engine.settings.auto_merge_threshold = 0.0  # force the auto-merge path
    id_a, _ = _make_fact(engine, "Deploy cmd", "Deploy with make release every Friday.")
    id_b, _ = _make_fact(engine, "Release cmd", "Release with make release every Friday.")

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=_duplicate_response()):
        run_duplicate_detection(engine)

    survivors = [r["id"] for r in engine.db.conn.execute(
        "SELECT id FROM nodes WHERE id IN (?, ?)", (id_a, id_b)).fetchall()]
    assert len(survivors) == 1  # one node merged away
    surv_seq = engine.db.conn.execute(
        "SELECT seq FROM nodes WHERE id = ?", (survivors[0],)).fetchone()["seq"]
    wm = get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY)
    assert surv_seq > wm  # survivor sits ABOVE the cursor: re-selected next run
