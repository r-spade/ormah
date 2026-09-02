"""Tests for LLM-based duplicate consolidation in duplicate_merger."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from ormah.models.node import CreateNodeRequest, NodeType, UpdateNodeRequest

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


def test_duplicate_detection_does_not_hold_the_lock_across_the_llm_call(engine):
    from tests.test_background.lock_probe import install_probe

    id_a, id_b = _create_pair(engine)
    # Force the auto-merge branch (the one wrapping engine.execute_merge, itself
    # lock-decorated) so the probe also exercises the nested-lock path.
    engine.settings.auto_merge_threshold = 0.0
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    probe = install_probe(engine)
    lock_held_at_call: list[bool] = []

    def fake_llm(*args, **kwargs):
        lock_held_at_call.append(probe.held)
        return json.dumps({
            "is_duplicate": True, "reason": "same fact",
            "merged_title": "merged", "merged_content": "merged content",
        })

    with patch(_LLM_PATCH, side_effect=fake_llm):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    assert lock_held_at_call, "the fake LLM was never called — the fixture stopped exercising the job"
    assert not any(lock_held_at_call)


def test_duplicate_detection_aborts_when_a_restore_lands_mid_run(engine):
    id_a, id_b = _create_pair(engine)
    # Force the auto-merge branch so the node-count assertion below actually
    # proves that path aborted too, not only the (mutually exclusive) proposal path.
    engine.settings.auto_merge_threshold = 0.0
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    proposals_before = engine.db.conn.execute(
        "SELECT COUNT(*) AS c FROM proposals").fetchone()["c"]
    nodes_before = engine.db.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]
    epoch_before = engine.restore_epoch

    # The bump must land AFTER the job read its entry epoch: restore_aware_job reads
    # engine.restore_epoch at call time, so bumping before the call would hand the job the
    # new value and leave no mismatch to detect. Inside the fake LLM is where a real restore
    # lands — between the unlocked LLM call and the apply step that follows it.
    def fake_llm(*args, **kwargs):
        engine._restore_epoch += 1
        return json.dumps({
            "is_duplicate": True, "reason": "same fact",
            "merged_title": "merged", "merged_content": "merged content",
        })

    with patch(_LLM_PATCH, side_effect=fake_llm):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)  # returns cleanly

    # Guard against silent vacuousness: the abort assertions below hold trivially if the
    # job never reached an apply step at all. Since the bump lives inside the fake LLM, a
    # moved epoch is proof the job actually got there.
    assert engine.restore_epoch > epoch_before, \
        "the fake LLM was never called — the fixture stopped exercising the job"
    assert engine.db.conn.execute(
        "SELECT COUNT(*) AS c FROM proposals").fetchone()["c"] == proposals_before
    assert engine.db.conn.execute(
        "SELECT COUNT(*) AS c FROM nodes").fetchone()["c"] == nodes_before


def test_a_node_edited_during_the_llm_call_is_not_merged_over(engine):
    """The merged text was written from pre-edit content; do not apply it to edited nodes."""
    import json
    from unittest.mock import patch

    id_a, id_b = _create_pair(engine)
    engine.settings.llm_provider = "ollama"
    engine.settings.auto_merge_threshold = 0.0

    edited = {"done": False}

    def edit_then_answer(*args, **kwargs):
        """The LLM call is the unlocked phase: a foreground edit lands here."""
        if not edited["done"]:
            edited["done"] = True
            engine.update_node(id_a, UpdateNodeRequest(
                content="the user rewrote this node while the merger was thinking"))
        return json.dumps({
            "is_duplicate": True, "reason": "same fact",
            "merged_title": "merged", "merged_content": "merged content",
        })

    nodes_before = engine.db.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]

    with patch("ormah.background.llm_client.llm_generate", side_effect=edit_then_answer):
        from ormah.background.duplicate_merger import run_duplicate_detection
        run_duplicate_detection(engine)

    assert edited["done"], "the fake LLM was never called — the fixture stopped exercising the job"
    row = engine.db.conn.execute(
        "SELECT content FROM nodes WHERE id = ?", (id_a,)).fetchone()
    assert row is not None, "the edited node was merged away"
    assert "the user rewrote this node" in row["content"], "the stale merged text overwrote a fresh edit"
    assert engine.db.conn.execute(
        "SELECT COUNT(*) AS c FROM nodes").fetchone()["c"] == nodes_before
