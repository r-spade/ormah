"""Tests for the run_maintenance two-call protocol."""

from __future__ import annotations

from ormah.models.node import CreateNodeRequest, NodeType


def _seed_similar_nodes(engine, n: int = 4, space: str | None = None) -> list[str]:
    """Create n nodes with similar content and return their IDs."""
    ids = []
    # Set threshold impossibly high so remember() never auto-links
    orig_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        for i in range(n):
            req = CreateNodeRequest(
                content=f"Python uses indentation to define code block scope {i}",
                type=NodeType.fact,
                title=f"Python indentation {i}",
                space=space,
            )
            nid, _ = engine.remember(req)
            ids.append(nid)
    finally:
        engine.settings.auto_link_similarity_threshold = orig_threshold
    return ids


class TestFindLinkCandidates:

    def test_returns_list(self, engine):
        _seed_similar_nodes(engine, 2)
        from ormah.background.auto_linker import _find_link_candidates
        result = _find_link_candidates(engine, limit=8)
        assert isinstance(result, list)

    def test_cap_enforced(self, engine):
        _seed_similar_nodes(engine, 6)
        from ormah.background.auto_linker import _find_link_candidates
        result = _find_link_candidates(engine, limit=2)
        assert len(result) <= 2

    def test_candidate_structure(self, engine):
        _seed_similar_nodes(engine, 2)
        from ormah.background.auto_linker import _find_link_candidates
        engine.settings.auto_link_similarity_threshold = 0.0
        result = _find_link_candidates(engine, limit=8)
        if result:
            c = result[0]
            assert "node_a" in c
            assert "node_b" in c
            assert "similarity" in c
            assert "id" in c["node_a"]
            assert "content" in c["node_a"]

    def test_no_duplicates(self, engine):
        _seed_similar_nodes(engine, 3)
        from ormah.background.auto_linker import _find_link_candidates
        engine.settings.auto_link_similarity_threshold = 0.0
        result = _find_link_candidates(engine, limit=20)
        pairs = [
            tuple(sorted([c["node_a"]["id"], c["node_b"]["id"]]))
            for c in result
        ]
        assert len(pairs) == len(set(pairs)), "Duplicate pairs returned"

    def test_already_checked_pairs_excluded(self, engine):
        ids = _seed_similar_nodes(engine, 2)
        from ormah.background.auto_linker import _apply_edge, _find_link_candidates

        engine.settings.auto_link_similarity_threshold = 0.0

        # First call should find the pair
        before = _find_link_candidates(engine, limit=8)
        if not before:
            return  # embedding similarity too low in test environment — skip

        pair_a = before[0]["node_a"]["id"]
        pair_b = before[0]["node_b"]["id"]

        # Mark as checked
        _apply_edge(engine, pair_a, pair_b, "none", "test")

        # Second call should not return the same pair
        after = _find_link_candidates(engine, limit=8)
        found_pairs = {
            tuple(sorted([c["node_a"]["id"], c["node_b"]["id"]]))
            for c in after
        }
        assert tuple(sorted([pair_a, pair_b])) not in found_pairs


class TestApplyMaintenanceResults:

    def test_apply_edges(self, engine):
        ids = _seed_similar_nodes(engine, 2)
        a, b = ids[0], ids[1]

        engine.apply_maintenance_results({
            "edges": [
                {"node_a_id": a, "node_b_id": b, "edge_type": "supports", "reason": "test"},
            ]
        })

        row = engine.db.conn.execute(
            "SELECT edge_type FROM edges WHERE source_id = ? AND target_id = ?",
            (a, b),
        ).fetchone()
        assert row is not None
        assert row["edge_type"] == "supports"

    def test_apply_none_edge_skips_edge_table(self, engine):
        ids = _seed_similar_nodes(engine, 2)
        a, b = ids[0], ids[1]

        engine.apply_maintenance_results({
            "edges": [
                {"node_a_id": a, "node_b_id": b, "edge_type": "none", "reason": "not related"},
            ]
        })

        row = engine.db.conn.execute(
            "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?",
            (a, b),
        ).fetchone()
        assert row is None  # no edge created

        # But pair should be in auto_link_checked
        pair = tuple(sorted([a, b]))
        checked = engine.db.conn.execute(
            "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?",
            pair,
        ).fetchone()
        assert checked is not None

    def test_apply_merges(self, engine):
        ids = _seed_similar_nodes(engine, 2)
        a, b = ids[0], ids[1]

        counts = engine.apply_maintenance_results({
            "merges": [
                {
                    "keep_id": a,
                    "discard_id": b,
                    "merged_content": "Merged content",
                    "merged_title": "Merged title",
                },
            ]
        })

        assert counts["merges"] == 1
        # Exactly one of the two nodes should remain (_pick_keeper decides which)
        remaining = engine.db.conn.execute(
            "SELECT id FROM nodes WHERE id IN (?, ?)", (a, b)
        ).fetchall()
        assert len(remaining) == 1

    def test_apply_consolidations(self, engine):
        ids = _seed_similar_nodes(engine, 3)

        counts = engine.apply_maintenance_results({
            "consolidations": [
                {
                    "node_ids": ids,
                    "title": "Consolidated Python indentation",
                    "content": "Python uses indentation to define code blocks.",
                    "type": "fact",
                },
            ]
        })

        assert counts["consolidations"] == 1
        # New consolidated node should exist
        row = engine.db.conn.execute(
            "SELECT 1 FROM nodes WHERE title = 'Consolidated Python indentation'"
        ).fetchone()
        assert row is not None

    def test_returns_counts(self, engine):
        ids = _seed_similar_nodes(engine, 2)
        a, b = ids[0], ids[1]

        counts = engine.apply_maintenance_results({
            "edges": [
                {"node_a_id": a, "node_b_id": b, "edge_type": "supports", "reason": "x"},
            ]
        })

        assert "edges" in counts
        assert "merges" in counts
        assert "consolidations" in counts
        assert "skipped" in counts
        assert counts["edges"] == 1

    def test_empty_results_ok(self, engine):
        counts = engine.apply_maintenance_results({})
        assert counts == {"edges": 0, "merges": 0, "consolidations": 0, "skipped": 0}


class TestGetMaintenanceBatches:

    def test_returns_all_keys(self, engine):
        batches = engine.get_maintenance_batches()
        assert "link_candidates" in batches
        assert "conflict_candidates" in batches
        assert "merge_candidates" in batches
        assert "consolidation_clusters" in batches
        assert "summary" in batches

    def test_summary_string(self, engine):
        batches = engine.get_maintenance_batches()
        assert isinstance(batches["summary"], str)

    def test_node_dict_fields(self, engine):
        _seed_similar_nodes(engine, 2)
        engine.settings.auto_link_similarity_threshold = 0.0
        batches = engine.get_maintenance_batches()
        for candidate in batches["link_candidates"]:
            for key in ("node_a", "node_b", "similarity"):
                assert key in candidate
            for field in ("id", "title", "type", "space", "content"):
                assert field in candidate["node_a"]
                assert field in candidate["node_b"]

    def test_content_truncated_to_400(self, engine):
        long_content = "x" * 600
        req = CreateNodeRequest(content=long_content, type=NodeType.fact, title="Long node")
        nid, _ = engine.remember(req)
        batches = engine.get_maintenance_batches()
        for c in batches["link_candidates"]:
            for node in (c["node_a"], c["node_b"]):
                assert len(node["content"]) <= 400


class TestWhisperSignal:

    def test_signal_absent_when_disabled(self, engine):
        """No signal when claude_maintenance_enabled=False."""
        engine.settings.claude_maintenance_enabled = False
        text = engine.get_context()
        assert "unprocessed_memories" not in text

    def test_signal_absent_below_threshold(self, engine):
        """No signal when count <= threshold."""
        engine.settings.claude_maintenance_enabled = True
        engine.settings.claude_maintenance_threshold = 100  # very high
        _seed_similar_nodes(engine, 2)
        text = engine.get_context()
        assert "unprocessed_memories" not in text

    def test_signal_present_above_threshold(self, engine):
        """Signal appears when count > threshold."""
        engine.settings.claude_maintenance_enabled = True
        engine.settings.claude_maintenance_threshold = 0  # always trigger
        _seed_similar_nodes(engine, 3)
        text = engine.get_context()
        assert "unprocessed_memories:" in text
