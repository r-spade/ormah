"""Tests for LLM-based duplicate consolidation in duplicate_merger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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


def _make_engine(tmp_path):
    from ormah.config import Settings
    from ormah.engine.memory_engine import MemoryEngine

    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    settings = Settings(memory_dir=tmp_path)
    eng = MemoryEngine(settings)
    eng.startup()
    return eng


def _make_engine_with_two_similar_nodes(tmp_path):
    """Engine with a temp db + two near-identical nodes (pair the pre-filter)."""
    eng = _make_engine(tmp_path)
    eng.settings.llm_provider = "ollama"  # llm_enabled derives from this
    _create_pair(eng)
    return eng


def _make_engine_with_many_similar_nodes(tmp_path, n):
    """Engine with n near-identical nodes so the pre-filter pairs several of them."""
    eng = _make_engine(tmp_path)
    eng.settings.llm_provider = "ollama"  # llm_enabled derives from this
    for i in range(n):
        eng.remember(
            CreateNodeRequest(
                content=f"Python is a programming language, variant {i}.",
                type=NodeType.fact,
                title=f"Python language {i}",
                tags=["test"],
            ),
            agent_id="test",
        )
    return eng


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


def test_llm_check_passes_json_schema_response_format(monkeypatch):
    import ormah.background.llm_client as llm_client
    from ormah.background import duplicate_merger as dm
    captured = {}
    def _fake_generate(settings, prompt, json_mode=True, **kwargs):
        captured.update(kwargs)
        return '{"is_duplicate": false, "merged_title": null, "merged_content": null, "reason": "x"}'
    monkeypatch.setattr(llm_client, "llm_generate", _fake_generate)
    result = dm._llm_check_duplicate(object(),
        {"title": "A", "type": "fact", "content": "a"}, {"title": "B", "type": "fact", "content": "b"})
    rf = captured.get("response_format")
    assert rf and rf["type"] == "json_schema"
    assert "is_duplicate" in rf["json_schema"]["schema"]["properties"]
    assert result == {"is_duplicate": False, "merged_title": None, "merged_content": None, "reason": "x"}


def test_run_dedup_records_only_not_duplicate_never_duplicate(monkeypatch, tmp_path):
    from ormah.background import duplicate_merger as dm
    monkeypatch.setattr(dm, "_llm_check_duplicate", lambda s, a, b: {"is_duplicate": False})
    engine = _make_engine_with_two_similar_nodes(tmp_path)
    try:
        dm.run_duplicate_detection(engine)
        rows = engine.db.conn.execute("SELECT result FROM duplicate_checked").fetchall()
        assert rows and all(r[0] == "not_duplicate" for r in rows)
    finally:
        engine.shutdown()


def test_run_dedup_records_error_and_circuit_breaks(monkeypatch, tmp_path):
    from ormah.background import duplicate_merger as dm
    calls = {"n": 0}
    def _fail(s, a, b):
        calls["n"] += 1
        return None
    monkeypatch.setattr(dm, "_llm_check_duplicate", _fail)
    engine = _make_engine_with_many_similar_nodes(tmp_path, n=20)
    try:
        # Settings is a pydantic model with extra="ignore" and no such field yet
        # (added in a later task) — bypass __setattr__ to stash it for getattr().
        object.__setattr__(engine.settings, "duplicate_check_max_llm_calls_per_run", 100)
        dm.run_duplicate_detection(engine)
        assert calls["n"] <= 3
        errs = engine.db.conn.execute(
            "SELECT count(*) FROM duplicate_checked WHERE result='error'"
        ).fetchone()[0]
        assert errs >= 1
    finally:
        engine.shutdown()


def test_run_dedup_stops_at_cap(monkeypatch, tmp_path):
    from ormah.background import duplicate_merger as dm
    calls = {"n": 0}
    monkeypatch.setattr(
        dm, "_llm_check_duplicate",
        lambda s, a, b: (calls.__setitem__("n", calls["n"] + 1) or {"is_duplicate": False}),
    )
    engine = _make_engine_with_many_similar_nodes(tmp_path, n=10)
    try:
        engine.settings.duplicate_check_max_llm_calls_per_run = 3
        dm.run_duplicate_detection(engine)
        assert calls["n"] == 3
    finally:
        engine.shutdown()


# --- Shared pair_skip_sql routing (fixes lexical error-backoff bug) ---


def test_dedup_error_row_backoff(monkeypatch, tmp_path):
    """A fresh 'error' row hides the pair within the backoff window; a stale one
    (past the window) lets the pair be re-checked. Guards against the lexical
    ISO-vs-SQLite datetime() compare bug (checked_at stored with 'T'/tz)."""
    from ormah.background import duplicate_merger as dm

    engine = _make_engine_with_two_similar_nodes(tmp_path)
    try:
        ids = [r[0] for r in engine.db.conn.execute("SELECT id FROM nodes WHERE type = 'fact'").fetchall()]
        pair = tuple(sorted(ids))

        calls = {"n": 0}

        def _count(s, a, b):
            calls["n"] += 1
            return {"is_duplicate": False}

        monkeypatch.setattr(dm, "_llm_check_duplicate", _count)

        # Fresh error row -> within backoff window -> pair skipped, LLM never called.
        with engine.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO duplicate_checked (node_a, node_b, result, checked_at) "
                "VALUES (?, ?, 'error', ?)",
                (*pair, datetime.now(timezone.utc).isoformat()),
            )
        dm.run_duplicate_detection(engine)
        assert calls["n"] == 0

        # Stale error row (past the 6h backoff window) -> pair re-checked.
        stale = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
        with engine.db.transaction() as conn:
            conn.execute(
                "UPDATE duplicate_checked SET checked_at = ? WHERE node_a = ? AND node_b = ?",
                (stale, *pair),
            )
        dm.run_duplicate_detection(engine)
        assert calls["n"] == 1
    finally:
        engine.shutdown()


def test_dedup_ordered_pair_skip(tmp_path):
    """A terminal row written as (a, b) also skips the reversed query (b, a)."""
    from ormah.background.pair_skip import normalize_pair, pair_skip_sql

    engine = _make_engine_with_two_similar_nodes(tmp_path)
    try:
        ids = [r[0] for r in engine.db.conn.execute("SELECT id FROM nodes WHERE type = 'fact'").fetchall()]
        a_id, b_id = ids
        pair = normalize_pair(a_id, b_id)

        with engine.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO duplicate_checked (node_a, node_b, result, checked_at) "
                "VALUES (?, ?, 'not_duplicate', ?)",
                (*pair, datetime.now(timezone.utc).isoformat()),
            )

        reversed_pair = normalize_pair(b_id, a_id)
        assert reversed_pair == pair
        skip = engine.db.conn.execute(
            pair_skip_sql("duplicate_checked", ("not_duplicate",)), (*reversed_pair, "-6 hours")
        ).fetchone()
        assert skip is not None
    finally:
        engine.shutdown()
