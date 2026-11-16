"""Tests for the run_maintenance two-call protocol."""

from __future__ import annotations

import threading
import time

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
        assert counts == {
            "edges": 0, "merges": 0, "consolidations": 0, "skipped": 0, "errors": [],
        }


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


def _issued(*clusters: list[str], pairs: tuple[tuple[str, str], ...] = ()) -> dict:
    """Phase 1 batches that issued these node ids, shaped like get_maintenance_batches."""
    def node(nid: str) -> dict:
        return {"id": nid, "title": "", "type": "fact", "space": "", "content": ""}

    def candidates() -> list[dict]:
        return [{"node_a": node(a), "node_b": node(b), "similarity": 0.9} for a, b in pairs]

    return {
        "link_candidates": candidates(),
        "conflict_candidates": [],
        "merge_candidates": candidates(),
        "consolidation_clusters": [[node(n) for n in c] for c in clusters],
        "summary": "",
    }


def _mistype(node_id: str) -> str:
    """One-character transcription error in the last hex digit, as in the 2026-07-09 incident."""
    return node_id[:-1] + ("1" if node_id[-1] != "1" else "2")


def _tier(engine, node_id: str) -> str:
    return engine.db.conn.execute("SELECT tier FROM nodes WHERE id = ?", (node_id,)).fetchone()[0]


class TestApplyMaintenanceResultsRejects:

    def test_consolidation_with_an_id_outside_the_issued_batch_applies_nothing(self, engine):
        ids = _seed_similar_nodes(engine, 3)
        submitted = [ids[0], ids[1], _mistype(ids[2])]

        counts = engine.apply_maintenance_results(
            {"consolidations": [
                {"node_ids": submitted, "title": "Typo cluster", "content": "x", "type": "fact"},
            ]},
            issued=_issued(ids),
        )

        assert counts["consolidations"] == 0
        assert counts["errors"] == [{
            "kind": "consolidation",
            "ids": submitted,
            "reason": f"not in the issued batch: {submitted[2]}",
        }]
        assert engine.db.conn.execute(
            "SELECT 1 FROM nodes WHERE title = 'Typo cluster'"
        ).fetchone() is None
        assert "archival" not in [_tier(engine, i) for i in ids]

    def test_merge_naming_an_existing_node_outside_the_issued_batch_deletes_nothing(self, engine):
        a, b, stranger = _seed_similar_nodes(engine, 3)

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": a, "discard_id": stranger, "merged_content": "both"}]},
            issued=_issued(pairs=((a, b),)),
        )

        assert counts["merges"] == 0
        assert counts["errors"] == [{
            "kind": "merge", "ids": [a, stranger], "reason": f"not in the issued batch: {stranger}",
        }]
        assert engine.db.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE id IN (?, ?)", (a, stranger)
        ).fetchone()[0] == 2

    def test_edge_with_an_id_outside_the_issued_batch_records_nothing(self, engine):
        a, b = _seed_similar_nodes(engine, 2)
        typo = _mistype(b)

        counts = engine.apply_maintenance_results(
            {"edges": [{"node_a_id": a, "node_b_id": typo, "edge_type": "none", "reason": ""}]},
            issued=_issued(pairs=((a, b),)),
        )

        assert counts["skipped"] == 0
        assert counts["errors"] == [{
            "kind": "edge", "ids": [a, typo], "reason": f"not in the issued batch: {typo}",
        }]
        assert engine.db.conn.execute(
            "SELECT 1 FROM auto_link_checked WHERE node_a = ? OR node_b = ?", (typo, typo)
        ).fetchone() is None

    def test_short_ids_of_issued_nodes_are_in_the_batch(self, engine):
        # Phase 1 issues Full ids; a submitted id may be a unique id prefix.
        a, b = _seed_similar_nodes(engine, 2)

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": a[:8], "discard_id": b[:8]}]},
            issued=_issued(pairs=((a, b),)),
        )

        assert (counts["merges"], counts["errors"]) == (1, [])

    def test_items_inside_the_issued_batch_still_apply(self, engine):
        a, b, c = _seed_similar_nodes(engine, 3)

        counts = engine.apply_maintenance_results(
            {
                "edges": [{"node_a_id": a, "node_b_id": b, "edge_type": "supports", "reason": ""}],
                "consolidations": [
                    {"node_ids": [a, _mistype(c)], "title": "T", "content": "x", "type": "fact"},
                ],
            },
            issued=_issued([a, c], pairs=((a, b),)),
        )

        assert counts["edges"] == 1
        assert [e["kind"] for e in counts["errors"]] == ["consolidation"]

    def test_consolidation_naming_a_missing_node_applies_nothing(self, engine):
        # Without an issued batch, the 2026-07-09 typo created the node with 4 of 5 sources.
        ids = _seed_similar_nodes(engine, 2)
        submitted = [*ids, _mistype(ids[1])]

        counts = engine.apply_maintenance_results({"consolidations": [
            {"node_ids": submitted, "title": "Partial cluster", "content": "x", "type": "fact"},
        ]})

        assert counts["consolidations"] == 0
        assert counts["errors"] == [{
            "kind": "consolidation", "ids": submitted, "reason": f"node not found: {submitted[2]}",
        }]
        assert engine.db.conn.execute(
            "SELECT 1 FROM nodes WHERE title = 'Partial cluster'"
        ).fetchone() is None
        assert "archival" not in [_tier(engine, i) for i in ids]

    def test_merge_whose_node_an_earlier_merge_removed_is_an_error(self, engine):
        a, b = _seed_similar_nodes(engine, 2)
        merge = {"keep_id": a, "discard_id": b}

        counts = engine.apply_maintenance_results({"merges": [merge, dict(merge)]})

        gone = a if engine.graph.get_node(a) is None else b  # _pick_keeper chose the other
        assert counts["merges"] == 1
        assert counts["errors"] == [
            {"kind": "merge", "ids": [a, b], "reason": f"node not found: {gone}"},
        ]

    def test_apply_time_exception_is_reported(self, engine):
        a, b = _seed_similar_nodes(engine, 2)

        counts = engine.apply_maintenance_results(
            {"edges": [{"node_a_id": a, "node_b_id": b, "reason": "no edge_type"}]},
        )

        assert counts["edges"] == counts["skipped"] == 0
        assert counts["errors"] == [
            {"kind": "edge", "ids": [a, b], "reason": "KeyError: 'edge_type'"},
        ]


def test_maintenance_manager_checks_phase2_against_the_job_batches(engine):
    from ormah.background.maintenance_manager import MaintenanceManager

    a, b, stranger = _seed_similar_nodes(engine, 3)
    engine.get_maintenance_batches = lambda: _issued(pairs=((a, b),))
    manager = MaintenanceManager(engine)

    def wait_for(job_id: str, status: str) -> dict:
        deadline = time.monotonic() + 5
        while (state := manager.get_status(job_id))["status"] != status:
            assert time.monotonic() < deadline, state
            threading.Event().wait(0.01)
        return state

    job_id = manager.start_phase1()["job_id"]
    wait_for(job_id, "awaiting_results")
    manager.submit_results(
        {"edges": [{"node_a_id": a, "node_b_id": stranger, "edge_type": "supports"}]},
        job_id=job_id,
    )

    summary = wait_for(job_id, "completed")["apply_summary"]
    assert summary["edges"] == 0
    assert summary["errors"] == [{
        "kind": "edge", "ids": [a, stranger], "reason": f"not in the issued batch: {stranger}",
    }]


def test_maintenance_manager_rejects_every_item_when_the_job_issued_nothing(engine):
    # _run_phase2 for a job it no longer holds: no batches to check against, so nothing applies.
    from ormah.background.maintenance_manager import MaintenanceManager

    a, b = _seed_similar_nodes(engine, 2)
    MaintenanceManager(engine)._run_phase2(
        "no-such-job",
        {"edges": [{"node_a_id": a, "node_b_id": b, "edge_type": "supports"}]},
    )

    assert engine.db.conn.execute(
        "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?", (a, b)
    ).fetchone() is None
