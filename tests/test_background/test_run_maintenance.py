"""Tests for the run_maintenance two-call protocol."""

from __future__ import annotations

import json

from ormah.engine.maintenance_signal import MAINTENANCE_DUE_SIGNAL
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
        _seed_similar_nodes(engine, 2)
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
        text = engine.get_whisper_context(prompt="how does Python indexing work")
        assert "maintenance_due" not in text

    def test_signal_absent_when_recently_run(self, engine):
        """No signal when maintenance was run within the interval."""
        from datetime import datetime, timezone

        engine.settings.claude_maintenance_enabled = True
        engine.settings.claude_maintenance_interval_hours = 24
        # Record a recent maintenance run
        now = datetime.now(timezone.utc).isoformat()
        engine.db.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_maintenance_run', ?)",
            (now,),
        )
        engine.db.conn.commit()
        text = engine.get_whisper_context(prompt="how does Python indexing work")
        assert "maintenance_due" not in text

    def test_signal_present_when_interval_elapsed(self, engine):
        """Signal appears when no maintenance has ever been run."""
        engine.settings.claude_maintenance_enabled = True
        engine.settings.claude_maintenance_interval_hours = 24
        engine.remember(CreateNodeRequest(
            content="User completed onboarding.",
            type=NodeType.preference,
            title="Onboarding complete",
            about_self=True,
        ))
        # Ensure no last_maintenance_run in meta (never ran)
        engine.db.conn.execute("DELETE FROM meta WHERE key = 'last_maintenance_run'")
        engine.db.conn.commit()
        text = engine.get_whisper_context(prompt="how does Python indexing work")
        assert MAINTENANCE_DUE_SIGNAL in text
        assert "continue the conversation without blocking the user" in text


class TestSelectClusterWithinBudget:
    """The per-node metadata overhead is ~77 chars for these fixtures; budgets below
    are chosen with that measured overhead in mind, so each case exercises the
    boundary it names."""

    def _node(self, nid: str, chars: int, content: str | None = None) -> dict:
        return {
            "id": nid,
            "title": f"t{nid}",
            "type": "fact",
            "space": "s",
            "content": content if content is not None else "x" * chars,
        }

    def _size(self, node: dict) -> int:
        return len(json.dumps(node, ensure_ascii=False))

    def test_cluster_within_budget_is_returned_whole(self):
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node(f"n{i}", 400) for i in range(5)]
        assert _select_cluster_within_budget(cluster, budget=24000, min_size=2) == cluster

    def test_oversized_cluster_is_trimmed(self, caplog):
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node(f"n{i}", 500) for i in range(4)]
        budget = self._size(cluster[0]) * 2 + 10  # exactly two nodes fit

        with caplog.at_level("INFO"):
            result = _select_cluster_within_budget(cluster, budget=budget, min_size=2)

        assert [n["id"] for n in result] == ["n0", "n1"]
        for node in result:
            assert len(node["content"]) == 500, "trim must never slice a node"
        assert "trimmed" in caplog.text

    def test_an_oversized_match_does_not_hide_the_smaller_ones(self, caplog):
        """The v3 defect: a strict prefix stopped here and lost seed + m2.

        Sizes: seed 177, m1 24_073, m2 173 serialized. `seed + m1` is 24_250,
        over budget; `seed + m2` is 350, well under. Stopping at m1 returns [].
        """
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node("seed", 100), self._node("m1", 24000), self._node("m2", 100)]

        with caplog.at_level("INFO"):
            result = _select_cluster_within_budget(cluster, budget=24000, min_size=2)

        assert [n["id"] for n in result] == ["seed", "m2"]
        assert "m1" in caplog.text, "the skipped node must be named in the log"

    def test_repeated_cycles_make_progress(self):
        """No starvation: the finder rebuilds the same ordered cluster every cycle,
        so a cluster that yields nothing once would yield nothing forever."""
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node("seed", 100), self._node("m1", 24000), self._node("m2", 100)]

        results = [
            _select_cluster_within_budget(cluster, budget=24000, min_size=2)
            for _ in range(3)
        ]

        assert all(r for r in results), "a cycle produced nothing — this cluster is starved"
        assert all([n["id"] for n in r] == ["seed", "m2"] for r in results)

    def test_a_kept_cluster_always_contains_the_seed(self):
        """The invariant next-fit violated: never consolidate without the seed."""
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node("seed", 600), self._node("m1", 400), self._node("m2", 400)]
        budget = self._size(cluster[0]) + self._size(cluster[1]) + 10

        result = _select_cluster_within_budget(cluster, budget=budget, min_size=2)

        assert result, "this budget fits two nodes; the cluster should survive"
        assert result[0]["id"] == "seed"

    def test_cluster_below_min_size_after_trim_is_dropped_with_warning(self, caplog):
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node(f"n{i}", 12001) for i in range(5)]

        with caplog.at_level("WARNING"):
            result = _select_cluster_within_budget(cluster, budget=24000, min_size=2)

        assert result == []
        assert caplog.records, "a dropped cluster must never be silent"

    def test_seed_larger_than_the_budget_drops_the_cluster(self, caplog):
        """An oversized seed must not claim the cluster and starve its matches —
        it drops the whole cluster, explicitly, with the seed named."""
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node("huge", 30000), self._node("a", 100), self._node("b", 100)]

        with caplog.at_level("WARNING"):
            result = _select_cluster_within_budget(cluster, budget=24000, min_size=2)

        assert result == []
        assert "huge" in caplog.text

    def test_budget_counts_serialized_size_not_raw_content(self, caplog):
        """3000 NUL chars are 3000 raw but 18_070 serialized — a len(content)
        budget would keep this cluster; the serialized budget must drop it."""
        from ormah.engine.memory_engine import _select_cluster_within_budget

        cluster = [self._node("esc", 0, content="\x00" * 3000), self._node("a", 400)]
        assert self._size(cluster[0]) > 5000 > len(cluster[0]["content"])

        with caplog.at_level("WARNING"):
            result = _select_cluster_within_budget(cluster, budget=5000, min_size=2)

        assert result == [], "the budget is measuring raw content, not the serialized node"


def _seed_long_nodes(engine, n: int = 2, chars: int = 600) -> dict[str, str]:
    """Create n nodes whose content is exactly `chars` long. Returns {id: content}.

    auto_link is disabled during remember() so the pairs stay unchecked and remain
    available as link candidates later.
    """
    base = "Python uses indentation to define code block scope. "
    seeded: dict[str, str] = {}
    orig_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        for i in range(n):
            content = (f"{i} " + base * 40)[:chars]
            assert len(content) == chars
            req = CreateNodeRequest(
                content=content,
                type=NodeType.fact,
                title=f"Python indentation {i}",
                space="testproject",
            )
            nid, _ = engine.remember(req)
            seeded[nid] = content
    finally:
        engine.settings.auto_link_similarity_threshold = orig_threshold
    return seeded


class TestConsolidationBatchFidelity:

    def test_consolidation_cluster_carries_full_content(self, engine):
        seeded = _seed_long_nodes(engine, n=2, chars=600)
        engine.settings.consolidation_cluster_threshold = 0.0
        engine.settings.consolidation_min_cluster_size = 2

        batches = engine.get_maintenance_batches()

        clusters = batches["consolidation_clusters"]
        assert clusters, "no consolidation cluster produced — the fixture is not exercising the batch"
        checked = 0
        for cluster in clusters:
            for node in cluster:
                if node["id"] in seeded:
                    assert node["content"] == seeded[node["id"]]
                    assert len(node["content"]) == 600
                    checked += 1
        assert checked >= 2, "seeded nodes never appeared in a cluster"

    def test_norm_truncates_screening_batches(self, engine, monkeypatch):
        """The over-fix guard: it must fail if _norm stops truncating screening.

        It monkeypatches the finder because the real one already slices to 400 in
        `_node_dict` (auto_linker.py:154), so asserting on its output would stay
        green even with _norm's limit removed. Both council peers found that.
        """
        import ormah.background.auto_linker as auto_linker

        long_node = {
            "id": "n1",
            "title": "long",
            "type": "fact",
            "space": "testproject",
            "content": "y" * 600,
        }
        other = dict(long_node, id="n2")
        monkeypatch.setattr(
            auto_linker,
            "_find_link_candidates",
            lambda engine, limit: [{"node_a": long_node, "node_b": other, "similarity": 0.9}],
        )

        batches = engine.get_maintenance_batches()

        assert batches["link_candidates"], "monkeypatched finder produced nothing"
        for candidate in batches["link_candidates"]:
            for node in (candidate["node_a"], candidate["node_b"]):
                assert len(node["content"]) == 400, "screening batches must stay truncated"

    def test_oversized_cluster_is_trimmed_in_the_batch(self, engine, monkeypatch):
        """A cluster over budget reaches the batch as a prefix, contents intact."""
        import json

        import ormah.background.consolidator as consolidator

        nodes = [
            {
                "id": f"n{i}",
                "title": f"node {i}",
                "type": "fact",
                "space": "testproject",
                "content": "z" * 6000,
            }
            for i in range(5)
        ]
        monkeypatch.setattr(
            consolidator, "_find_consolidation_clusters", lambda engine, limit: [nodes]
        )
        engine.settings.claude_maintenance_cluster_max_chars = 24000

        batches = engine.get_maintenance_batches()

        clusters = batches["consolidation_clusters"]
        assert len(clusters) == 1, "a prefix is one cluster, never several"
        kept = clusters[0]
        assert [n["id"] for n in kept] == ["n0", "n1", "n2"], (
            "three 6000-char nodes serialize to 18_258; a fourth reaches 24_344, over 24_000"
        )
        for node in kept:
            assert len(node["content"]) == 6000, "trim must never slice a node"
        assert len(json.dumps(kept, ensure_ascii=False)) <= 24000

    def test_worst_case_cardinality_stays_bounded(self, engine, monkeypatch):
        """Four max-size clusters of large nodes stay within `4 x budget`.

        With one prefix per cluster the bound is exactly `n_clusters * budget` —
        unlike v2's bin-packing, where the sub-cluster count was unbounded. Node
        size is 6000 chars so the trim actually fires (3 of 5 survive), and
        `assert clusters` keeps the test from passing by emitting nothing.
        """
        import json

        import ormah.background.consolidator as consolidator

        budget = engine.settings.claude_maintenance_cluster_max_chars
        max_nodes = engine.settings.consolidation_max_cluster_nodes
        big_clusters = [
            [
                {
                    "id": f"c{c}n{i}",
                    "title": f"node {c}-{i}",
                    "type": "fact",
                    "space": "testproject",
                    "content": "w" * 6000,
                }
                for i in range(max_nodes)
            ]
            for c in range(4)
        ]
        monkeypatch.setattr(
            consolidator, "_find_consolidation_clusters", lambda engine, limit: big_clusters
        )

        batches = engine.get_maintenance_batches()

        clusters = batches["consolidation_clusters"]
        assert clusters, "the trim dropped everything — the fixture proves nothing"
        assert len(clusters) == 4, "one prefix per cluster"
        for sub in clusters:
            assert len(json.dumps(sub, ensure_ascii=False)) <= budget

        payload = json.dumps(clusters, ensure_ascii=False)
        bound = 4 * budget + 4096
        assert len(payload) <= bound, (
            f"consolidation batch is unbounded: {len(payload)} chars, bound {bound}"
        )

    def test_consolidation_cluster_carries_type(self, engine):
        seeded = _seed_long_nodes(engine, n=2, chars=600)
        engine.settings.consolidation_cluster_threshold = 0.0
        engine.settings.consolidation_min_cluster_size = 2

        batches = engine.get_maintenance_batches()

        clusters = batches["consolidation_clusters"]
        assert clusters, "no consolidation cluster produced — the assertion below would never run"
        checked = 0
        for cluster in clusters:
            for node in cluster:
                if node["id"] in seeded:
                    assert node["type"] == "fact"
                    checked += 1
        assert checked >= 2, "seeded nodes never appeared in a cluster"
