"""Tests for LLM-based duplicate consolidation in duplicate_merger."""

from __future__ import annotations

import json
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


def test_pairs_evaluated_counts_one_candidate_pair(engine):
    """Issue #90: pairs_evaluated must reflect exactly one LLM decision call."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(
        "ormah.background.duplicate_merger._llm_check_duplicate",
        return_value={"is_duplicate": False, "reason": "not a duplicate"},
    ):
        from ormah.background.duplicate_merger import run_duplicate_detection
        stats = run_duplicate_detection(engine)

    assert stats["pairs_attempted"] == 1
    assert stats["pairs_evaluated"] == 1
    # duration_s must have millisecond resolution — a fast mocked-LLM run
    # must not silently round down to 0.0 (issue #90 finding 2).
    assert stats["duration_s"] > 0


def test_pairs_attempted_counts_llm_unavailable_pair_but_not_evaluated(engine):
    """Issue #90 (council finding 2): an LLM-unavailable pair (None decision)
    must count as attempted but NOT as evaluated."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(
        "ormah.background.duplicate_merger._llm_check_duplicate",
        return_value=None,
    ):
        from ormah.background.duplicate_merger import run_duplicate_detection
        stats = run_duplicate_detection(engine)

    assert stats["pairs_attempted"] == 1
    assert stats["pairs_evaluated"] == 0


# --- #87 pair batching ---

def test_duplicate_prompt_is_composed_from_parts():
    from ormah.background import duplicate_merger as dm
    assert dm._LLM_DUPLICATE_PROMPT == (
        dm._LLM_DUP_INTRO + "\n\n" + dm._LLM_DUP_PAIR + "\n\n" + dm._LLM_DUP_RULES
    )
    assert dm._LLM_DUP_INSTRUCTIONS == dm._LLM_DUP_INTRO + "\n\n" + dm._LLM_DUP_RULES


def test_scan_order_randomizes_only_when_capped():
    """Council C2: capped runs randomize scan order for fair coverage across runs."""
    from ormah.background import duplicate_merger as dm
    assert dm._scan_order(0) == ""
    assert dm._scan_order(5) == " ORDER BY RANDOM()"


def test_batched_dedup_creates_proposals(engine):
    from ormah.background import duplicate_merger as dm
    for _ in range(3):
        engine.remember(CreateNodeRequest(
            content="ormah stores memories in sqlite with fts5", title="ormah storage"))
    engine.settings.llm_provider = "ollama"
    engine.settings.maintenance_pairs_per_call = 2
    engine.settings.auto_merge_threshold = 2.0     # force proposal path, not auto-merge
    _reset_adapter()

    def fake_batch(settings, prompt, json_mode=True, **kw):
        n = prompt.count("### Pair ")
        return json.dumps({"verdicts": [
            {"pair_id": i, "is_duplicate": True, "merged_title": "t",
             "merged_content": "c", "reason": "same fact"} for i in range(n)]})

    single = {"is_duplicate": True, "merged_title": "t", "merged_content": "c",
              "reason": "same fact"}
    with patch("ormah.background.llm.pair_batch.llm_generate", fake_batch), \
            patch("ormah.background.duplicate_merger._llm_check_duplicate", return_value=single):
        stats = dm.run_duplicate_detection(engine)

    n_props = engine.db.conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE type = 'merge'").fetchone()[0]
    assert n_props >= 1
    assert stats["pairs_evaluated"] >= 1


def test_batched_dedup_skips_pairs_whose_node_was_merged_away(engine):
    """Codex council finding (#87): overlapping pairs in one window (e.g. (A,B)
    and (B,C)) — once an auto-merge deletes a shared node, a later pair must be
    skipped, not re-merged on the stale (now-missing) node. execute_merge silently
    no-ops on a missing node ('Node X not found.'), which would otherwise miscount
    it as a successful merge."""
    from ormah.background import duplicate_merger as dm
    for _ in range(3):
        engine.remember(CreateNodeRequest(
            content="ormah stores memories in sqlite with fts5", title="ormah storage"))
    engine.settings.llm_provider = "ollama"
    engine.settings.maintenance_pairs_per_call = 3      # all candidate pairs in one window
    engine.settings.auto_merge_threshold = 0.0          # force the auto-merge path for every pair
    _reset_adapter()

    real_merge = engine.execute_merge
    bad_calls = []

    def spy_merge(node_id_a, node_id_b, **kw):
        for nid in (node_id_a, node_id_b):
            if engine.db.conn.execute("SELECT 1 FROM nodes WHERE id = ?", (nid,)).fetchone() is None:
                bad_calls.append(nid)
        return real_merge(node_id_a, node_id_b, **kw)

    def fake_batch(settings, prompt, json_mode=True, **kw):
        n = prompt.count("### Pair ")
        return json.dumps({"verdicts": [
            {"pair_id": i, "is_duplicate": True, "merged_title": "t",
             "merged_content": "c", "reason": "same"} for i in range(n)]})

    single = {"is_duplicate": True, "merged_title": "t", "merged_content": "c", "reason": "same"}
    with patch("ormah.background.llm.pair_batch.llm_generate", fake_batch), \
            patch("ormah.background.duplicate_merger._llm_check_duplicate", return_value=single), \
            patch.object(engine, "execute_merge", spy_merge):
        dm.run_duplicate_detection(engine)

    assert bad_calls == [], f"execute_merge called on already-deleted node(s): {bad_calls}"
