"""Phase 2 of run_maintenance rejects a merge that the Phase 1 assignment does not support.

Worked example (2026-10-18): Phase 1 offered the merge pair (A, B). Node C was only a
consolidation cluster member. The agent submitted keep=A, discard=C,
merged_content="placeholder". The server applied it: one node was deleted and the
survivor's body became the literal "placeholder".
"""

from __future__ import annotations

import threading

import pytest

from ormah.background.maintenance_manager import MaintenanceManager
from ormah.models.node import CreateNodeRequest, NodeType

GOOD_CONTENT = (
    "The nightly export job writes one compressed archive per day to the shared bucket. "
    "Both source notes say this, so the merge keeps one copy of the fact."
)


def _seed(engine, *contents: str) -> list[str]:
    """Create one node per content and return their ids, with auto-link off."""
    ids = []
    orig = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        for i, content in enumerate(contents):
            nid, _ = engine.remember(
                CreateNodeRequest(content=content, type=NodeType.fact, title=f"node {i}")
            )
            ids.append(nid)
    finally:
        engine.settings.auto_link_similarity_threshold = orig
    return ids


def _ref(node_id: str) -> dict:
    return {"id": node_id, "title": "", "type": "fact", "space": "", "content": ""}


def _issued(*merge_pairs: tuple[str, str], cluster: tuple[str, ...] = ()) -> dict:
    return {
        "link_candidates": [],
        "conflict_candidates": [],
        "merge_candidates": [
            {"node_a": _ref(a), "node_b": _ref(b), "similarity": 0.84} for a, b in merge_pairs
        ],
        "consolidation_clusters": [[_ref(n) for n in cluster]] if cluster else [],
    }


def _state(engine, node_id: str) -> tuple[str, str] | None:
    node = engine.file_store.load(node_id)
    return None if node is None else (node.title, node.content)


@pytest.fixture
def three(engine):
    """keep and other form the offered pair; outsider is only in a cluster."""
    keep, other, outsider = _seed(
        engine,
        "The nightly export job writes a compressed archive to the shared bucket each day.",
        "The nightly export job uploads one archive per day to the shared storage bucket.",
        "The release checklist needs a signed tag and a changelog entry before a version "
        "is published to the package index.",
    )
    return keep, other, outsider


class TestMergeGate:

    def test_the_2026_10_18_merge_is_rejected_and_changes_nothing(self, engine, three):
        keep, other, outsider = three
        before = {n: _state(engine, n) for n in three}

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": outsider,
                         "merged_title": "nightly export writes one archive per day",
                         "merged_content": "placeholder"}]},
            issued=_issued((keep, other), cluster=(outsider,)),
        )

        assert counts["merges"] == 0
        assert counts["rejected"] == 1
        assert {n: _state(engine, n) for n in three} == before

    def test_ids_outside_the_offered_pair_are_rejected_even_with_good_content(
        self, engine, three
    ):
        keep, other, outsider = three
        before = {n: _state(engine, n) for n in three}

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": outsider,
                         "merged_title": "t", "merged_content": GOOD_CONTENT}]},
            issued=_issued((keep, other), cluster=(outsider,)),
        )

        assert (counts["merges"], counts["rejected"]) == (0, 1)
        assert {n: _state(engine, n) for n in three} == before

    @pytest.mark.parametrize(
        "content", ["", "   \n", "placeholder", "Placeholder", "  PLACEHOLDER\n"]
    )
    def test_empty_or_placeholder_content_is_rejected(self, engine, three, content):
        keep, other, _ = three
        before = (_state(engine, keep), _state(engine, other))

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": other,
                         "merged_title": "t", "merged_content": content}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (0, 1)
        assert (_state(engine, keep), _state(engine, other)) == before

    @pytest.mark.parametrize("content", ["    \n", "placeholder", "  Placeholder\n"])
    def test_empty_or_placeholder_content_is_rejected_when_no_shorter_than_a_source(
        self, engine, content
    ):
        # Four-character sources: the content above is not shorter than either, so
        # only the empty/placeholder rule can reject it.
        keep, other = _seed(engine, "Yes.", "Nope")
        before = (_state(engine, keep), _state(engine, other))

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": other,
                         "merged_title": "t", "merged_content": content}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (0, 1)
        assert (_state(engine, keep), _state(engine, other)) == before

    def test_content_shorter_than_the_shorter_source_is_rejected(self, engine, three):
        keep, other, _ = three
        shorter = min(len(engine.file_store.load(n).content) for n in (keep, other))
        before = (_state(engine, keep), _state(engine, other))

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": other,
                         "merged_title": "t", "merged_content": "x" * (shorter - 1)}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (0, 1)
        assert (_state(engine, keep), _state(engine, other)) == before

    def test_content_as_long_as_the_shorter_source_is_applied(self, engine, three):
        keep, other, _ = three
        shorter = min(len(engine.file_store.load(n).content) for n in (keep, other))

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": other,
                         "merged_title": "t", "merged_content": "x" * shorter}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (1, 0)

    def test_offered_pair_in_reverse_order_is_applied(self, engine, three):
        keep, other, _ = three

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": other, "discard_id": keep,
                         "merged_title": "merged", "merged_content": GOOD_CONTENT}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (1, 0)
        survivors = [n for n in (keep, other) if engine.file_store.load(n) is not None]
        assert len(survivors) == 1
        assert _state(engine, survivors[0]) == ("merged", GOOD_CONTENT)

    def test_offered_pair_given_by_short_ids_is_applied(self, engine, three):
        # A merge may name a node by a unique id prefix; Phase 1 issues Full ids.
        keep, other, _ = three

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep[:8], "discard_id": other[:8],
                         "merged_title": "merged", "merged_content": GOOD_CONTENT}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (1, 0)

    def test_a_rejected_merge_does_not_stop_the_rest_of_the_submission(self, engine):
        a, b, c, d = _seed(
            engine,
            "Alpha memory about the first topic, long enough to be a real source node.",
            "Beta memory about the first topic, long enough to be a real source node too.",
            "Gamma memory about the second topic, long enough to be a real source node.",
            "Delta memory about the second topic, long enough to be a real source node.",
        )

        counts = engine.apply_maintenance_results(
            {
                "edges": [{"node_a_id": c, "node_b_id": d, "edge_type": "supports",
                           "reason": "x"}],
                "merges": [
                    {"keep_id": a, "discard_id": c, "merged_title": "t",
                     "merged_content": "placeholder"},
                    {"keep_id": a, "discard_id": b, "merged_title": "first topic",
                     "merged_content": GOOD_CONTENT},
                ],
                "consolidations": [{"node_ids": [c, d], "title": "second topic",
                                    "content": "Gamma and Delta both cover the second topic.",
                                    "type": "fact"}],
            },
            issued=_issued((a, b)),
        )

        assert (
            counts["edges"], counts["merges"], counts["consolidations"], counts["rejected"]
        ) == (1, 1, 1, 1)
        assert engine.file_store.load(c) is not None

    def test_without_an_assignment_the_content_checks_still_apply(self, engine, three):
        keep, other, _ = three

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": other,
                         "merged_title": "t", "merged_content": "placeholder"}]},
        )

        assert (counts["merges"], counts["rejected"]) == (0, 1)
        assert engine.file_store.load(keep) is not None
        assert engine.file_store.load(other) is not None

    def test_a_merge_without_merged_content_keeps_todays_behavior(self, engine, three):
        keep, other, _ = three

        counts = engine.apply_maintenance_results(
            {"merges": [{"keep_id": keep, "discard_id": other}]},
            issued=_issued((keep, other)),
        )

        assert (counts["merges"], counts["rejected"]) == (1, 0)


class _ObservedManager(MaintenanceManager):
    """Expose phase completion as events, so the test waits on a condition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = threading.Event()
        self.finished = threading.Event()

    def _run_phase1(self, job_id):
        try:
            super()._run_phase1(job_id)
        finally:
            self.prepared.set()

    def _run_phase2(self, job_id, results):
        try:
            super()._run_phase2(job_id, results)
        finally:
            self.finished.set()


def test_phase2_checks_merges_against_the_batches_phase1_issued(engine, three, monkeypatch):
    from ormah.background import auto_linker, conflict_detector, consolidator, duplicate_merger

    keep, other, outsider = three
    pair = {"node_a": {"id": keep, "title": "a", "type": "fact", "space": None,
                       "content": "a"},
            "node_b": {"id": other, "title": "b", "type": "fact", "space": None,
                       "content": "b"},
            "similarity": 0.84}
    monkeypatch.setattr(auto_linker, "_find_link_candidates", lambda engine, limit: [])
    monkeypatch.setattr(conflict_detector, "_find_conflict_candidates", lambda engine, limit: [])
    monkeypatch.setattr(duplicate_merger, "_find_merge_candidates", lambda engine, limit: [pair])
    monkeypatch.setattr(consolidator, "_find_consolidation_clusters", lambda engine, limit: [])
    manager = _ObservedManager(engine)

    grant = manager.start_phase1()
    assert manager.prepared.wait(5)
    manager.submit_results(
        {"merges": [{"keep_id": keep, "discard_id": outsider, "merged_title": "t",
                     "merged_content": GOOD_CONTENT}]},
        job_id=grant["job_id"],
    )
    assert manager.finished.wait(5)

    status = manager.get_status(grant["job_id"])
    assert status["status"] == "completed"
    assert (status["apply_summary"]["merges"], status["apply_summary"]["rejected"]) == (0, 1)
    assert engine.file_store.load(outsider) is not None
