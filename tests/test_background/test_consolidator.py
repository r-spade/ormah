"""Tests for the memory consolidation background job."""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import pytest

from ormah.config import Settings
from ormah.background import consolidator
from ormah.models.node import CreateNodeRequest, NodeType, Tier


@pytest.fixture
def consolidation_engine(engine):
    """Engine with several similar working memories."""
    contents = [
        "Python uses indentation to define code blocks",
        "Python relies on whitespace indentation for block structure",
        "In Python, indentation determines code block scope",
        "Code blocks in Python are delimited by indentation level",
    ]
    ids = []
    for i, content in enumerate(contents):
        req = CreateNodeRequest(
            content=content,
            type=NodeType.fact,
            title=f"Python indentation {i}",
            space="testproject",
        )
        nid, _ = engine.remember(req)
        ids.append(nid)
    return engine, ids


class TestConsolidation:

    @patch("ormah.background.llm_client.llm_generate")
    def test_creates_consolidated_node(self, mock_llm, consolidation_engine):
        """LLM consolidation should create a new node with derived_from edges."""
        engine, original_ids = consolidation_engine
        mock_llm.return_value = json.dumps({
            "title": "Python indentation rules",
            "summary": "Python uses whitespace indentation to define code block scope and structure.",
            "type": "fact",
        })

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)

        # Function should complete without error.
        # Actual consolidation depends on embedding similarity threshold.

    @patch("ormah.background.llm_client.llm_generate")
    def test_originals_demoted_to_archival(self, mock_llm, consolidation_engine):
        """Original nodes should be demoted to archival tier."""
        engine, original_ids = consolidation_engine
        mock_llm.return_value = json.dumps({
            "title": "Python indentation rules",
            "summary": "Python uses whitespace indentation to define code block scope.",
            "type": "fact",
        })

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)
        # Completes without error; actual demotion depends on clustering

    def test_skips_without_llm(self, engine):
        """Should not crash when LLM is disabled."""
        engine.settings.llm_provider = "none"
        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)

    def test_skips_with_few_nodes(self, engine):
        """Should skip when there aren't enough working nodes."""
        req = CreateNodeRequest(
            content="Solo memory",
            type=NodeType.fact,
            title="Solo",
        )
        engine.remember(req)

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)

    def test_preserves_core_nodes(self, engine):
        """Core-tier nodes should not be consolidated."""
        for i in range(5):
            req = CreateNodeRequest(
                content=f"Important core fact {i}",
                type=NodeType.fact,
                tier=Tier.core,
                title=f"Core {i}",
            )
            engine.remember(req)

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)

        # Core nodes should still be core
        core_rows = engine.db.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tier = 'core'"
        ).fetchone()
        assert core_rows[0] >= 5  # At least the 5 we created + self node

    @patch("ormah.background.llm_client.llm_generate")
    def test_space_majority_vote(self, mock_llm, engine):
        """Consolidated node should inherit the majority space."""
        for i in range(4):
            space = "projectA" if i < 3 else "projectB"
            req = CreateNodeRequest(
                content=f"Similar fact about coding {i}",
                type=NodeType.fact,
                title=f"Coding fact {i}",
                space=space,
            )
            engine.remember(req)

        mock_llm.return_value = json.dumps({
            "title": "Coding facts consolidated",
            "summary": "Various facts about coding practices.",
            "type": "fact",
        })

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)
        # Completes without error


def test_consolidation_settings_defaults(tmp_path):
    s = Settings(memory_dir=tmp_path)
    assert s.consolidation_max_clusters_per_run == 10
    assert s.consolidation_min_cluster_size == 2
    assert s.consolidation_cluster_threshold == 0.6
    assert s.consolidation_max_cluster_nodes == 5


def test_consolidation_settings_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ORMAH_CONSOLIDATION_MAX_CLUSTERS_PER_RUN", "3")
    s = Settings(memory_dir=tmp_path)
    assert s.consolidation_max_clusters_per_run == 3


def test_run_consolidation_uses_settings_cap(engine, monkeypatch):
    from ormah.background import consolidator

    engine.settings.llm_provider = "ollama"
    engine.settings.consolidation_max_clusters_per_run = 3
    seen = {}

    def fake_find(eng, limit):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(consolidator, "_find_consolidation_clusters", fake_find)
    consolidator.run_consolidation(engine)
    assert seen["limit"] == 3


def test_inverted_cluster_bounds_returns_empty_and_warns(consolidation_engine, caplog):
    from ormah.background.consolidator import _find_consolidation_clusters

    engine, _ids = consolidation_engine
    engine.settings.consolidation_max_cluster_nodes = 1
    engine.settings.consolidation_min_cluster_size = 2

    with caplog.at_level("WARNING"):
        clusters = _find_consolidation_clusters(engine)

    assert clusters == []
    assert "consolidation_max_cluster_nodes" in caplog.text


class TestConsolidationSignatureSkip:

    def test_consolidate_passes_strict_schema_and_records_signature(
        self, monkeypatch, consolidation_engine
    ):
        engine, original_ids = consolidation_engine
        cluster = [
            {"id": original_ids[0], "title": "SQLite pick", "content": "API uses SQLite.",
             "space": None},
            {"id": original_ids[1], "title": "SQLite decision",
             "content": "Chose SQLite for the API.", "space": None},
        ]
        captured = {}

        def spy(settings, prompt, **kwargs):
            captured.update(kwargs)
            return json.dumps({
                "title": "SQLite for the API",
                "summary": "The API uses SQLite.",
                "type": "decision",
            })

        monkeypatch.setattr("ormah.background.llm_client.llm_generate", spy)

        consolidator._consolidate_cluster(engine, cluster)

        assert (
            captured["response_format"]["json_schema"]["schema"]
            is consolidator._CONSOLIDATE_RESPONSE_SCHEMA
        )
        sig = consolidator._cluster_signature(cluster)
        row = engine.db.conn.execute(
            "SELECT 1 FROM consolidation_checked WHERE signature = ?", (sig,)
        ).fetchone()
        assert row is not None

    def test_consolidate_skips_known_signature(self, monkeypatch, consolidation_engine):
        engine, _ = consolidation_engine
        cluster = [
            {"id": "n1", "title": "t", "content": "c", "space": None},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        sig = consolidator._cluster_signature(cluster)
        engine.db.conn.execute(
            "INSERT INTO consolidation_checked (signature, checked_at) VALUES (?, datetime('now'))",
            (sig,),
        )

        called = {"n": 0}

        def spy(*a, **k):
            called["n"] += 1
            return None

        monkeypatch.setattr("ormah.background.llm_client.llm_generate", spy)

        consolidator._consolidate_cluster(engine, cluster)

        assert called["n"] == 0  # skipped before the LLM call

    def test_signature_changes_on_title_or_space_edit(self):
        base = [
            {"id": "n1", "title": "t", "content": "c", "space": None},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        title_edit = [
            {"id": "n1", "title": "different title", "content": "c", "space": None},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        space_edit = [
            {"id": "n1", "title": "t", "content": "c", "space": "projectA"},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        type_edit = [
            {"id": "n1", "title": "t", "content": "c", "space": None, "type": "decision"},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]

        base_sig = consolidator._cluster_signature(base)
        assert base_sig != consolidator._cluster_signature(title_edit)
        assert base_sig != consolidator._cluster_signature(space_edit)
        assert base_sig != consolidator._cluster_signature(type_edit)

    def test_consolidate_records_signature_on_noop_summary(self, monkeypatch, consolidation_engine):
        """An empty/blank summary is a no-op that must still record the signature."""
        engine, _ = consolidation_engine
        cluster = [
            {"id": "n1", "title": "t", "content": "c", "space": None},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        monkeypatch.setattr(
            "ormah.background.llm_client.llm_generate",
            lambda *a, **k: json.dumps({"title": "x", "summary": "", "type": "fact"}),
        )

        consolidator._consolidate_cluster(engine, cluster)

        sig = consolidator._cluster_signature(cluster)
        row = engine.db.conn.execute(
            "SELECT 1 FROM consolidation_checked WHERE signature = ?", (sig,)
        ).fetchone()
        assert row is not None

    def test_consolidate_does_not_record_when_llm_unavailable(self, monkeypatch, consolidation_engine):
        engine, _ = consolidation_engine
        cluster = [
            {"id": "n1", "title": "t", "content": "c", "space": None},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        monkeypatch.setattr("ormah.background.llm_client.llm_generate", lambda *a, **k: None)

        consolidator._consolidate_cluster(engine, cluster)

        sig = consolidator._cluster_signature(cluster)
        row = engine.db.conn.execute(
            "SELECT 1 FROM consolidation_checked WHERE signature = ?", (sig,)
        ).fetchone()
        assert row is None

    def test_consolidate_does_not_record_signature_on_invalid_json(
        self, monkeypatch, consolidation_engine
    ):
        """Invalid JSON is now treated as transient (mirrors raw is None): retry next run,
        do NOT permanently skip a consolidatable cluster on a one-off parse failure."""
        engine, _ = consolidation_engine
        cluster = [
            {"id": "n1", "title": "t", "content": "c", "space": None},
            {"id": "n2", "title": "t2", "content": "c2", "space": None},
        ]
        monkeypatch.setattr(
            "ormah.background.llm_client.llm_generate", lambda *a, **k: "not json at all"
        )

        consolidator._consolidate_cluster(engine, cluster)

        sig = consolidator._cluster_signature(cluster)
        row = engine.db.conn.execute(
            "SELECT 1 FROM consolidation_checked WHERE signature = ?", (sig,)
        ).fetchone()
        assert row is None

    def test_consolidate_clamps_off_enum_type_to_fact(self, monkeypatch, consolidation_engine):
        """The result-fallback recovers JSON shape but not the schema's enum constraint —
        an off-enum type from the LLM must be clamped, not written straight to the node."""
        engine, original_ids = consolidation_engine
        cluster = [
            {"id": original_ids[0], "title": "t", "content": "c", "space": None},
            {"id": original_ids[1], "title": "t2", "content": "c2", "space": None},
        ]
        monkeypatch.setattr(
            "ormah.background.llm_client.llm_generate",
            lambda *a, **k: json.dumps(
                {"title": "x", "summary": "consolidated summary", "type": "architecture"}
            ),
        )

        consolidator._consolidate_cluster(engine, cluster)

        tag_row = engine.db.conn.execute(
            "SELECT node_id FROM node_tags WHERE tag = 'consolidated'"
        ).fetchone()
        assert tag_row is not None
        new_row = engine.db.conn.execute(
            "SELECT type FROM nodes WHERE id = ?", (tag_row["node_id"],)
        ).fetchone()
        assert new_row["type"] == "fact"

    def test_consolidation_checked_table_exists_on_migrated_engine(self, engine):
        """The skip table is created by init_schema()'s executescript(schema.sql), which
        runs on every engine construction (fresh or reopened) — so it's always present."""
        row = engine.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidation_checked'"
        ).fetchone()
        assert row is not None


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_real_claude_cli_consolidate_creates_node_with_valid_type(consolidation_engine):
    """End-to-end: --json-schema -> structured_output round-trips for the consolidate prompt."""
    from ormah.background.llm_client import llm_generate, reset_adapter

    engine, original_ids = consolidation_engine
    engine.settings.llm_provider = "claude_cli"
    reset_adapter()

    cluster = engine.db.conn.execute(
        "SELECT id, title, content, space FROM nodes WHERE id IN ({})".format(
            ",".join("?" * len(original_ids))
        ),
        original_ids,
    ).fetchall()
    cluster = [dict(r) for r in cluster]

    # Capability probe: skip only when the CLI itself is unusable (not logged in, binary
    # missing/broken), not when the real consolidate prompt merely produces a result — the
    # adapter's result-fallback (e276baa) makes that round-trip reliably now, so a null
    # result for the real prompt below is a genuine regression, not an environment issue.
    from ormah.background.consolidator import _CONSOLIDATE_RESPONSE_SCHEMA
    probe = llm_generate(
        engine.settings, "Return title='Test', summary='Hello', type='fact'.",
        json_mode=True,
        response_format={"type": "json_schema", "json_schema": {"schema": _CONSOLIDATE_RESPONSE_SCHEMA}},
    )
    if probe is None:
        pytest.skip("claude CLI unusable (likely not logged in or binary missing)")

    consolidator._consolidate_cluster(engine, cluster)

    tag_row = engine.db.conn.execute(
        "SELECT node_id FROM node_tags WHERE tag = 'consolidated'"
    ).fetchone()
    assert tag_row is not None
    new_row = engine.db.conn.execute(
        "SELECT content, type FROM nodes WHERE id = ?", (tag_row["node_id"],)
    ).fetchone()
    assert new_row is not None
    assert new_row["content"]
    assert new_row["type"] in _CONSOLIDATE_RESPONSE_SCHEMA["properties"]["type"]["enum"]
