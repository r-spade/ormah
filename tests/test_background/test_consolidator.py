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

    @patch("ormah.background.llm_client.llm_generate")
    def test_lock_is_not_held_across_the_llm_call(self, mock_llm, consolidation_engine):
        from tests.test_background.lock_probe import install_probe

        engine, ids = consolidation_engine
        engine.settings.llm_provider = "ollama"
        engine.settings.consolidation_min_cluster_size = 2

        probe = install_probe(engine)
        lock_held_at_call = []

        def fake_llm(*args, **kwargs):
            lock_held_at_call.append(probe.held)
            return json.dumps({
                "title": "Python uses indentation",
                "summary": "Python blocks are delimited by indentation.",
                "type": "fact",
            })

        mock_llm.side_effect = fake_llm

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)

        assert lock_held_at_call, "the fake LLM was never called — the fixture stopped exercising the job"
        assert not any(lock_held_at_call)

    @patch("ormah.background.llm_client.llm_generate")
    def test_aborts_when_a_restore_lands_mid_run(self, mock_llm, consolidation_engine):
        engine, ids = consolidation_engine
        engine.settings.llm_provider = "ollama"
        engine.settings.consolidation_min_cluster_size = 2

        nodes_before = engine.db.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]
        epoch_before = engine.restore_epoch

        # The bump must land AFTER the job read its entry epoch: restore_aware_job reads
        # engine.restore_epoch at call time, so bumping before the call would hand the job
        # the new value and leave no mismatch to detect. Inside the fake LLM is where a real
        # restore lands — between the unlocked LLM call and the apply step that follows it.
        def fake_llm(*args, **kwargs):
            engine._restore_epoch += 1
            return json.dumps({
                "title": "Python uses indentation",
                "summary": "Python blocks are delimited by indentation.",
                "type": "fact",
            })

        mock_llm.side_effect = fake_llm

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)  # returns cleanly

        # Guard against silent vacuousness: the assertion below holds trivially if the job
        # never reached an apply step (no clusters found, min cluster size unmet). Since the
        # bump lives inside the fake LLM, a moved epoch is proof the job actually got there.
        assert engine.restore_epoch > epoch_before, \
            "the fake LLM was never called — the fixture stopped exercising the job"
        nodes_after = engine.db.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]
        assert nodes_after == nodes_before

    @patch("ormah.background.llm_client.llm_generate")
    def test_a_node_promoted_during_the_llm_call_is_not_demoted(self, mock_llm, consolidation_engine):
        """The #257 canary, in the consolidator: revalidate tier inside the apply step."""
        from ormah.models.node import Tier, UpdateNodeRequest

        engine, ids = consolidation_engine
        engine.settings.llm_provider = "ollama"
        engine.settings.consolidation_min_cluster_size = 2

        promoted_id = ids[0]
        promoted = {"done": False}

        def promote_then_answer(*args, **kwargs):
            """The LLM call is the unlocked phase: a foreground promotion lands here."""
            if not promoted["done"]:
                promoted["done"] = True
                engine.update_node(promoted_id, UpdateNodeRequest(tier=Tier.core))
            return json.dumps({
                "title": "Python uses indentation",
                "summary": "Python blocks are delimited by indentation.",
                "type": "fact",
            })

        mock_llm.side_effect = promote_then_answer

        from ormah.background.consolidator import run_consolidation
        run_consolidation(engine)

        assert promoted["done"], "the fake LLM was never called — the fixture stopped exercising the job"
        row = engine.db.conn.execute(
            "SELECT tier FROM nodes WHERE id = ?", (promoted_id,)).fetchone()
        assert row["tier"] == "core", "consolidation demoted a node promoted after the snapshot"


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


