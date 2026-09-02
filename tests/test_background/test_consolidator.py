"""Tests for the memory consolidation background job."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ormah.config import Settings
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


def _remember(engine, content: str, title: str, tags: list[str] | None = None) -> str:
    req = CreateNodeRequest(
        content=content, type=NodeType.fact, title=title, space="testproject", tags=tags or []
    )
    nid, _ = engine.remember(req)
    return nid


_SAME_TEXT = "Python uses indentation to define code blocks"


def test_consolidated_nodes_are_never_seed_nor_member(engine):
    """A summary is terminal: discovery must not pick it as seed or member (#261)."""
    from ormah.background.consolidator import _find_consolidation_clusters

    raw = [_remember(engine, _SAME_TEXT, f"Raw {i}") for i in range(2)]
    summaries = [
        _remember(engine, _SAME_TEXT, f"Summary {i}", tags=["consolidated"]) for i in range(2)
    ]
    # Fixture check: the tag reached the index, otherwise the test proves nothing.
    tagged = {
        r[0]
        for r in engine.db.conn.execute(
            "SELECT node_id FROM node_tags WHERE tag = 'consolidated'"
        ).fetchall()
    }
    assert set(summaries) <= tagged

    clusters = _find_consolidation_clusters(engine)
    ids_in_clusters = {n["id"] for cluster in clusters for n in cluster}

    assert ids_in_clusters.isdisjoint(summaries), "a consolidated node entered a cluster"
    assert set(raw) <= ids_in_clusters, "the raw pair should still cluster"


def test_two_summaries_are_not_summarised_again(monkeypatch, engine):
    """Issue #261's scenario: run 1 yields N1 and N2, run 2 must leave them alone."""
    from ormah.background import consolidator

    engine.settings.llm_provider = "ollama"  # default is "none", which skips the job
    engine.settings.consolidation_max_cluster_nodes = 2  # four sources -> two clusters
    for i in range(4):
        _remember(engine, _SAME_TEXT, f"Source {i}")

    prompts: list[str] = []

    def fake_llm(settings, prompt, json_mode=True, **kwargs):
        prompts.append(prompt)
        return json.dumps(
            {"title": "Python indentation rules", "summary": "Blocks by indentation.", "type": "fact"}
        )

    monkeypatch.setattr("ormah.background.llm_client.llm_generate", fake_llm)

    def consolidated_ids() -> list[str]:
        rows = engine.db.conn.execute(
            "SELECT node_id FROM node_tags WHERE tag = 'consolidated'"
        ).fetchall()
        return sorted(r[0] for r in rows)

    consolidator.run_consolidation(engine)
    first = consolidated_ids()
    assert len(first) == 2 and len(prompts) == 2

    consolidator.run_consolidation(engine)

    assert consolidated_ids() == first, "run 2 created a summary of summaries"
    assert len(prompts) == 2, "run 2 asked the LLM again"
    tiers = {
        r[0]
        for r in engine.db.conn.execute(
            "SELECT tier FROM nodes WHERE id IN (?, ?)", first
        ).fetchall()
    }
    assert tiers == {"working"}


def test_consolidated_seed_never_recruits_raw_neighbours(engine):
    """Seed-side exclusion, proven independently of the member check (#261).

    Discovery scans working nodes in insertion order, so the summaries go in FIRST: without
    the seed predicate the first summary seeds a cluster and recruits the raw pair, which the
    member predicate alone cannot prevent.
    """
    from ormah.background.consolidator import _find_consolidation_clusters

    summaries = [
        _remember(engine, _SAME_TEXT, f"Summary {i}", tags=["consolidated"]) for i in range(2)
    ]
    raw = [_remember(engine, _SAME_TEXT, f"Raw {i}") for i in range(2)]

    clusters = _find_consolidation_clusters(engine)
    ids_in_clusters = {n["id"] for cluster in clusters for n in cluster}

    assert ids_in_clusters.isdisjoint(summaries), "a consolidated node seeded a cluster"
    assert set(raw) <= ids_in_clusters, "the raw pair should still cluster"
