"""Every entry point canonicalizes a node reference to its Full id before acting (#280).

A Short id is the first 8 characters of a Full id. It is not unique, so it is only an
input alias: it must resolve to exactly one Full id before anything is written, and
everything written afterwards — edges, markdown connections, audit rows,
`auto_link_checked`, merge records, feedback rows — names the Full id.

Each entry point is checked three ways, using two memories that share the Short id
`deadbeef` and one memory whose Short id `cafef00d` is unique:
- the unique Short id works, and every side effect is recorded under the Full id;
- the ambiguous Short id changes nothing and says why;
- a Full id that no memory has is reported as not found.
"""

from __future__ import annotations

import json

import pytest

from ormah.models.node import (
    ConnectRequest,
    Connection,
    CreateNodeRequest,
    EdgeType,
    MemoryNode,
    NodeType,
    Tier,
    UpdateNodeRequest,
)
from ormah.store.file_store import UnresolvedNodeReference

AMBIGUOUS = "deadbeef"
UNIQUE = "cafef00d"
COLLIDER_A = f"{AMBIGUOUS}-0000-4000-8000-00000000000a"
COLLIDER_B = f"{AMBIGUOUS}-0000-4000-8000-00000000000b"
UNIQUE_ID = f"{UNIQUE}-0000-4000-8000-000000000001"
OTHER_ID = "0ddba11a-0000-4000-8000-000000000002"
ABSENT_ID = "abad1dea-0000-4000-8000-000000000003"


def _index(engine, node_id: str, title: str, content: str | None = None) -> MemoryNode:
    node = MemoryNode(
        id=node_id,
        type=NodeType.fact,
        tier=Tier.working,
        title=title,
        content=content or f"Content of {title}.",
        source="test",
    )
    engine.builder.index_single(engine.file_store.save(node))
    return node


@pytest.fixture
def nodes(engine):
    """Two colliders, one memory with a unique Short id, and one more to link to."""
    _index(engine, COLLIDER_A, "Collider A")
    _index(engine, COLLIDER_B, "Collider B")
    _index(engine, UNIQUE_ID, "Unique target")
    _index(engine, OTHER_ID, "Other memory")
    return engine


def _files(engine) -> dict[str, str]:
    """Every live node file and its text, to prove a refusal wrote nothing."""
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(engine.file_store.nodes_dir.glob("*.md"))
    }


def _edges(engine) -> list[tuple[str, str, str]]:
    rows = engine.db.conn.execute(
        "SELECT source_id, target_id, edge_type FROM edges ORDER BY source_id, target_id"
    ).fetchall()
    return [(r["source_id"], r["target_id"], r["edge_type"]) for r in rows]


def _audit(engine) -> list[tuple[str, str]]:
    rows = engine.db.conn.execute(
        "SELECT operation, node_id FROM audit_log ORDER BY id"
    ).fetchall()
    return [(r["operation"], r["node_id"]) for r in rows]


def _checked(engine) -> list[tuple[str, str]]:
    rows = engine.db.conn.execute(
        "SELECT node_a, node_b FROM auto_link_checked ORDER BY node_a, node_b"
    ).fetchall()
    return [(r["node_a"], r["node_b"]) for r in rows]


def _mark_checked(engine, a: str, b: str) -> None:
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'none', '2026-01-01T00:00:00+00:00')",
            tuple(sorted([a, b])),
        )


def _snapshot(engine):
    return (_files(engine), _edges(engine), _audit(engine), _checked(engine))


def _refusal(call) -> str:
    """A refusal is an exception, so no caller can read it as success."""
    with pytest.raises(UnresolvedNodeReference) as excinfo:
        call()
    return str(excinfo.value)


# --- The resolver ---------------------------------------------------------


class TestResolveNodeReference:
    def test_full_id_resolves_to_itself(self, nodes):
        assert nodes.resolve_node_reference(UNIQUE_ID).full_id == UNIQUE_ID

    def test_unique_short_id_resolves_to_its_full_id(self, nodes):
        resolution = nodes.resolve_node_reference(UNIQUE)
        assert resolution.full_id == UNIQUE_ID
        assert resolution.problem is None

    def test_longer_unique_prefix_resolves(self, nodes):
        assert nodes.resolve_node_reference(UNIQUE_ID[:13]).full_id == UNIQUE_ID

    def test_ambiguous_short_id_is_unresolved_and_names_every_candidate(self, nodes):
        resolution = nodes.resolve_node_reference(AMBIGUOUS)
        assert resolution.full_id is None
        assert not resolution.absent
        assert "ambiguous" in resolution.problem.lower()
        assert COLLIDER_A in resolution.problem and COLLIDER_B in resolution.problem

    def test_absent_full_id_is_absent_not_unresolved(self, nodes):
        resolution = nodes.resolve_node_reference(ABSENT_ID)
        assert resolution.full_id is None
        assert resolution.problem is None
        assert resolution.absent

    def test_prefix_shorter_than_a_short_id_is_refused_even_when_unique(self, nodes):
        resolution = nodes.resolve_node_reference(UNIQUE[:4])
        assert resolution.full_id is None
        assert resolution.problem is not None

    def test_empty_reference_is_refused(self, nodes):
        assert nodes.resolve_node_reference("").problem is not None

    def test_like_wildcards_are_not_prefix_patterns(self, nodes):
        resolution = nodes.resolve_node_reference("cafef00%")
        assert resolution.full_id is None

    def test_short_id_shared_with_an_unindexed_file_is_ambiguous(self, engine):
        """The index holds one node under the Short id, but a second file on disk shares
        it (saved, not yet indexed). The store has to agree before the alias resolves."""
        _index(engine, COLLIDER_A, "Indexed")
        engine.file_store.save(
            MemoryNode(
                id=COLLIDER_B, type=NodeType.fact, title="Indexed", content="Not yet indexed.",
                source="test",
            )
        )
        resolution = engine.resolve_node_reference(AMBIGUOUS)
        assert resolution.full_id is None
        assert resolution.problem is not None


# --- update_node ----------------------------------------------------------


class TestUpdateNode:
    def test_unique_short_id_records_everything_under_the_full_id(self, nodes):
        _mark_checked(nodes, UNIQUE_ID, OTHER_ID)

        result = nodes.update_node(UNIQUE, UpdateNodeRequest(content="New content."))

        assert result is not None and result.startswith("Updated")
        assert UNIQUE_ID in result
        assert nodes.file_store.load(UNIQUE_ID).content == "New content."
        assert _audit(nodes) == [("update", UNIQUE_ID)]
        # The checked pair names the Full id, so only a Full id delete can clear it.
        assert _checked(nodes) == []

    def test_ambiguous_short_id_changes_nothing(self, nodes):
        _mark_checked(nodes, COLLIDER_A, OTHER_ID)
        before = _snapshot(nodes)

        message = _refusal(
            lambda: nodes.update_node(AMBIGUOUS, UpdateNodeRequest(content="Overwritten."))
        )

        assert "ambiguous" in message.lower()
        assert COLLIDER_A in message and COLLIDER_B in message
        assert _snapshot(nodes) == before

    def test_absent_full_id_is_not_found(self, nodes):
        before = _snapshot(nodes)
        assert nodes.update_node(ABSENT_ID, UpdateNodeRequest(content="x")) is None
        assert _snapshot(nodes) == before

    def test_add_connections_by_unique_short_id_stores_the_full_id(self, nodes):
        nodes.update_node(
            OTHER_ID,
            UpdateNodeRequest(add_connections=[Connection(target=UNIQUE)]),
        )

        targets = [c.target for c in nodes.file_store.load(OTHER_ID).connections]
        assert targets == [UNIQUE_ID]
        assert (OTHER_ID, UNIQUE_ID, "related_to") in _edges(nodes)

    def test_add_connections_by_ambiguous_short_id_is_skipped_and_reported(self, nodes):
        before = _snapshot(nodes)

        result = nodes.update_node(
            OTHER_ID,
            UpdateNodeRequest(add_connections=[Connection(target=AMBIGUOUS)]),
        )

        assert "ambiguous" in result.lower()
        assert nodes.file_store.load(OTHER_ID).connections == []
        assert _snapshot(nodes) == before


# --- mark_outdated --------------------------------------------------------


class TestMarkOutdated:
    def test_unique_short_id_records_the_full_id(self, nodes):
        result = nodes.mark_outdated(UNIQUE, reason="superseded")

        assert result is not None and result.startswith("Marked outdated")
        assert UNIQUE_ID in result
        assert nodes.file_store.load(UNIQUE_ID).valid_until is not None
        assert _audit(nodes) == [("mark_outdated", UNIQUE_ID)]

    def test_ambiguous_short_id_changes_nothing(self, nodes):
        before = _snapshot(nodes)

        message = _refusal(lambda: nodes.mark_outdated(AMBIGUOUS, reason="wrong one?"))

        assert "ambiguous" in message.lower()
        assert _snapshot(nodes) == before
        assert nodes.file_store.load(COLLIDER_A).valid_until is None
        assert nodes.file_store.load(COLLIDER_B).valid_until is None

    def test_absent_full_id_is_not_found(self, nodes):
        before = _snapshot(nodes)
        assert nodes.mark_outdated(ABSENT_ID) is None
        assert _snapshot(nodes) == before


# --- delete_node ----------------------------------------------------------


class TestDeleteNode:
    def test_unique_short_id_records_the_full_id(self, nodes):
        _mark_checked(nodes, UNIQUE_ID, OTHER_ID)

        result = nodes.delete_node(UNIQUE)

        assert result.startswith("Deleted") and UNIQUE_ID in result
        assert _audit(nodes) == [("delete", UNIQUE_ID)]
        assert _checked(nodes) == []
        assert nodes.graph.get_node(UNIQUE_ID) is None

    def test_ambiguous_short_id_changes_nothing(self, nodes):
        before = _snapshot(nodes)
        message = _refusal(lambda: nodes.delete_node(AMBIGUOUS))
        assert "ambiguous" in message.lower()
        assert _snapshot(nodes) == before

    def test_absent_full_id_is_not_found(self, nodes):
        assert nodes.delete_node(ABSENT_ID) is None


# --- connect --------------------------------------------------------------


class TestConnect:
    def test_unique_short_ids_record_full_ids_in_edges_and_markdown(self, nodes):
        result = nodes.connect(
            ConnectRequest(source_id=UNIQUE, target_id=OTHER_ID[:8], edge=EdgeType.supports)
        )

        assert (UNIQUE_ID, OTHER_ID, "supports") in _edges(nodes)
        targets = [c.target for c in nodes.file_store.load(UNIQUE_ID).connections]
        assert targets == [OTHER_ID]
        assert UNIQUE_ID in result and OTHER_ID in result

    @pytest.mark.parametrize("side", ["source", "target"])
    def test_ambiguous_short_id_on_either_end_changes_nothing(self, nodes, side):
        before = _snapshot(nodes)
        req = (
            ConnectRequest(source_id=AMBIGUOUS, target_id=UNIQUE_ID)
            if side == "source"
            else ConnectRequest(source_id=UNIQUE_ID, target_id=AMBIGUOUS)
        )

        message = _refusal(lambda: nodes.connect(req))

        assert "ambiguous" in message.lower()
        assert _snapshot(nodes) == before

    @pytest.mark.parametrize("side", ["Source", "Target"])
    def test_absent_full_id_is_not_found(self, nodes, side):
        before = _snapshot(nodes)
        req = (
            ConnectRequest(source_id=ABSENT_ID, target_id=UNIQUE_ID)
            if side == "Source"
            else ConnectRequest(source_id=UNIQUE_ID, target_id=ABSENT_ID)
        )

        result = nodes.connect(req)

        assert result == f"{side} node {ABSENT_ID} not found."
        assert _snapshot(nodes) == before


# --- remember(links=...) --------------------------------------------------


class TestRememberLinks:
    def _remember(self, engine, target: str) -> tuple[str, str]:
        return engine.remember(
            CreateNodeRequest(
                content="A memory linked at creation.",
                type=NodeType.fact,
                title="Linked at creation",
                connections=[Connection(target=target)],
            )
        )

    def test_unique_short_id_link_is_stored_as_the_full_id(self, nodes):
        new_id, _ = self._remember(nodes, UNIQUE)

        targets = [c.target for c in nodes.file_store.load(new_id).connections]
        assert targets == [UNIQUE_ID]
        assert (new_id, UNIQUE_ID, "related_to") in _edges(nodes)

    def test_ambiguous_short_id_link_is_skipped_and_reported(self, nodes):
        new_id, text = self._remember(nodes, AMBIGUOUS)

        assert nodes.file_store.load(new_id).connections == []
        assert not [
            edge for edge in _edges(nodes)
            if new_id in edge[:2] and {COLLIDER_A, COLLIDER_B} & set(edge[:2])
        ]
        assert "ambiguous" in text.lower()

    def test_absent_full_id_link_is_skipped_and_reported(self, nodes):
        new_id, text = self._remember(nodes, ABSENT_ID)

        assert nodes.file_store.load(new_id).connections == []
        assert ABSENT_ID in text


# --- execute_merge / undo_merge ------------------------------------------


class TestMerge:
    def test_unique_short_ids_record_full_ids_and_report_them(self, nodes):
        result = nodes.execute_merge(UNIQUE, OTHER_ID[:8])

        assert result.startswith("Merged")
        row = nodes.db.conn.execute("SELECT * FROM merge_history").fetchone()
        assert {row["kept_node_id"], row["removed_node_id"]} == {UNIQUE_ID, OTHER_ID}
        assert UNIQUE_ID in result and OTHER_ID in result
        assert row["id"] in result

        undo = nodes.undo_merge(row["id"][:8])
        assert row["id"] in undo
        assert row["removed_node_id"] in undo

    @pytest.mark.parametrize("which", [0, 1])
    def test_ambiguous_short_id_merges_nothing(self, nodes, which):
        before = _snapshot(nodes)
        refs = [AMBIGUOUS, UNIQUE_ID]
        if which:
            refs.reverse()

        message = _refusal(lambda: nodes.execute_merge(*refs))

        assert "ambiguous" in message.lower()
        assert _snapshot(nodes) == before
        assert nodes.db.conn.execute("SELECT COUNT(*) FROM merge_history").fetchone()[0] == 0

    def test_absent_full_id_is_not_found(self, nodes):
        before = _snapshot(nodes)
        assert nodes.execute_merge(ABSENT_ID, UNIQUE_ID) == f"Node {ABSENT_ID} not found."
        assert _snapshot(nodes) == before

    def test_short_and_full_id_of_one_memory_is_not_a_self_merge(self, nodes):
        """Once canonical, both references are the same memory. Merging it into itself
        would soft-delete the only copy."""
        before = _snapshot(nodes)

        with pytest.raises(ValueError, match="both name the memory"):
            nodes.execute_merge(UNIQUE, UNIQUE_ID)

        assert _snapshot(nodes) == before
        assert nodes.file_store.load(UNIQUE_ID) is not None
        assert nodes.db.conn.execute("SELECT COUNT(*) FROM merge_history").fetchone()[0] == 0

    def test_ambiguous_merge_id_prefix_undoes_nothing(self, nodes):
        with nodes.db.transaction() as conn:
            for suffix in "ab":
                conn.execute(
                    "INSERT INTO merge_history (id, kept_node_id, removed_node_id, "
                    "removed_node_snapshot, original_edges, merged_at) "
                    "VALUES (?, ?, ?, '{}', '[]', '2026-01-01T00:00:00+00:00')",
                    (f"feedface-0000-4000-8000-00000000000{suffix}", UNIQUE_ID, OTHER_ID),
                )

        result = nodes.undo_merge("feedface")

        assert "ambiguous" in result.lower()
        undone = nodes.db.conn.execute(
            "SELECT COUNT(*) FROM merge_history WHERE undone_at IS NOT NULL"
        ).fetchone()[0]
        assert undone == 0


# --- apply_maintenance_results -------------------------------------------


class TestApplyMaintenanceResults:
    def test_edge_by_unique_short_ids_is_recorded_under_full_ids(self, nodes):
        counts = nodes.apply_maintenance_results({
            "edges": [{
                "node_a_id": UNIQUE, "node_b_id": OTHER_ID[:8],
                "edge_type": "supports", "reason": "r",
            }],
        })

        assert counts["edges"] == 1
        assert (UNIQUE_ID, OTHER_ID, "supports") in _edges(nodes)
        assert _checked(nodes) == [tuple(sorted([UNIQUE_ID, OTHER_ID]))]
        targets = [c.target for c in nodes.file_store.load(UNIQUE_ID).connections]
        assert targets == [OTHER_ID]

    def test_edge_by_ambiguous_short_id_is_not_applied(self, nodes):
        before = _snapshot(nodes)

        counts = nodes.apply_maintenance_results({
            "edges": [{
                "node_a_id": AMBIGUOUS, "node_b_id": UNIQUE_ID,
                "edge_type": "none", "reason": "r",
            }],
        })

        assert counts["edges"] == 0
        assert counts["unresolved"] == 1
        assert _snapshot(nodes) == before

    def test_merge_by_ambiguous_short_id_is_not_applied(self, nodes):
        before = _snapshot(nodes)

        counts = nodes.apply_maintenance_results({
            "merges": [{"keep_id": UNIQUE_ID, "discard_id": AMBIGUOUS}],
        })

        assert counts["merges"] == 0
        assert counts["unresolved"] == 1
        assert _snapshot(nodes) == before

    def test_self_merge_by_alias_is_not_applied_or_counted(self, nodes):
        counts = nodes.apply_maintenance_results({
            "merges": [{"keep_id": UNIQUE_ID, "discard_id": UNIQUE}],
        })

        assert counts["merges"] == 0
        assert nodes.file_store.load(UNIQUE_ID) is not None

    def test_merge_of_an_absent_node_is_not_counted(self, nodes):
        counts = nodes.apply_maintenance_results({
            "merges": [{"keep_id": UNIQUE_ID, "discard_id": ABSENT_ID}],
        })

        assert counts["merges"] == 0
        assert counts["unresolved"] == 1

    def test_consolidation_with_an_ambiguous_member_creates_nothing(self, nodes):
        before = _snapshot(nodes)

        counts = nodes.apply_maintenance_results({
            "consolidations": [{
                "node_ids": [UNIQUE_ID, AMBIGUOUS],
                "title": "Consolidated", "content": "Both together.",
            }],
        })

        assert counts["consolidations"] == 0
        assert counts["unresolved"] == 1
        assert _snapshot(nodes) == before

    def test_consolidation_by_unique_short_ids_links_full_ids(self, nodes):
        counts = nodes.apply_maintenance_results({
            "consolidations": [{
                "node_ids": [UNIQUE, OTHER_ID[:8]],
                "title": "Consolidated", "content": "Both together.",
            }],
        })

        assert counts["consolidations"] == 1
        # Asserted on the markdown, not the edges table: the consolidator's own
        # update_node(tier=archival) reindexes each source and drops the incoming
        # derived_from row until the next rebuild (a separate, pre-existing issue).
        [consolidated] = [
            n for n in nodes.file_store.list_all() if n.title == "Consolidated"
        ]
        derived = [
            c.target for c in consolidated.connections if c.edge == EdgeType.derived_from
        ]
        assert sorted(derived) == sorted([UNIQUE_ID, OTHER_ID])
        for source_id in (UNIQUE_ID, OTHER_ID):
            assert nodes.file_store.load(source_id).superseded_by == consolidated.id


# --- recall_node ----------------------------------------------------------


class TestRecallNode:
    def test_unique_short_id_recalls_and_logs_the_full_id(self, nodes):
        text = nodes.recall_node(UNIQUE)

        assert text is not None and UNIQUE_ID in text
        logged = {
            r["node_id"] for r in nodes.db.conn.execute("SELECT node_id FROM whisper_log")
        }
        assert UNIQUE_ID in logged
        assert UNIQUE not in logged

    def test_ambiguous_short_id_raises_and_logs_nothing(self, nodes):
        with pytest.raises(UnresolvedNodeReference):
            nodes.recall_node(AMBIGUOUS)
        assert nodes.db.conn.execute("SELECT COUNT(*) FROM whisper_log").fetchone()[0] == 0

    def test_absent_full_id_is_not_found(self, nodes):
        assert nodes.recall_node(ABSENT_ID) is None

    def test_neighbors_are_listed_by_full_id(self, nodes):
        nodes.connect(ConnectRequest(source_id=UNIQUE_ID, target_id=OTHER_ID))

        text = nodes.recall_node(UNIQUE_ID)

        assert f"(id: {OTHER_ID}" in text


# --- submit_feedback ------------------------------------------------------


def _log_whisper(engine, node_id: str) -> int:
    with engine.db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO whisper_log (session_id, space, prompt_hash, prompt_text, prompt_vec, "
            "node_id, score, was_injected, logged_at) "
            "VALUES ('s', NULL, 'h', 'prompt', X'', ?, 0.9, 1, '2026-01-01T00:00:00+00:00')",
            (node_id,),
        )
        return int(cursor.lastrowid)


class TestSubmitFeedback:
    def test_unique_short_id_records_the_full_id(self, nodes):
        _log_whisper(nodes, UNIQUE_ID)

        result = nodes.submit_feedback(UNIQUE, 1, "implicit")

        assert UNIQUE_ID in result
        rows = nodes.db.conn.execute("SELECT node_id FROM affinity").fetchall()
        assert [r["node_id"] for r in rows] == [UNIQUE_ID]
        signals = nodes.db.conn.execute("SELECT node_id FROM signals").fetchall()
        assert [r["node_id"] for r in signals] == [UNIQUE_ID]

    def test_short_id_ambiguous_among_memories_is_refused(self, nodes):
        """Only one collider was ever surfaced, so whisper_log alone sees one match. The
        Short id still names two memories, and the agent may mean the other."""
        _log_whisper(nodes, COLLIDER_A)

        result = nodes.submit_feedback(AMBIGUOUS, 1, "implicit")

        assert "ambiguous" in result.lower()
        assert nodes.db.conn.execute("SELECT COUNT(*) FROM affinity").fetchone()[0] == 0
        assert nodes.db.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 0

    def test_whisper_log_id_pins_the_event_even_for_a_colliding_short_id(self, nodes):
        log_id = _log_whisper(nodes, COLLIDER_A)

        result = nodes.submit_feedback(AMBIGUOUS, 1, "implicit", whisper_log_id=log_id)

        assert COLLIDER_A in result
        rows = nodes.db.conn.execute("SELECT node_id FROM affinity").fetchall()
        assert [r["node_id"] for r in rows] == [COLLIDER_A]

    def test_ambiguity_message_lists_full_ids(self, nodes):
        _log_whisper(nodes, COLLIDER_A)
        _log_whisper(nodes, COLLIDER_B)

        result = nodes.submit_feedback(AMBIGUOUS, 1, "implicit")

        assert COLLIDER_A in result and COLLIDER_B in result

    def test_mismatched_whisper_log_id_names_the_full_id(self, nodes):
        log_id = _log_whisper(nodes, UNIQUE_ID)

        result = nodes.submit_feedback(OTHER_ID, 1, "implicit", whisper_log_id=log_id)

        assert UNIQUE_ID in result

    def test_absent_full_id_is_not_found(self, nodes):
        result = nodes.submit_feedback(ABSENT_ID, 1, "implicit")
        assert "No whisper_log entry" in result


def test_decay_does_not_count_an_unreadable_node_as_demoted(nodes, monkeypatch, caplog):
    """update_node now raises for a file it cannot read. The decay run must skip that
    node, keep demoting the others, and not log it as demoted."""
    from ormah.background import decay_manager

    corrupt = nodes.file_store._find_file(COLLIDER_A)
    corrupt.write_text("this is not a node", encoding="utf-8")
    nodes.file_store._id_cache.clear()
    nodes.file_store._cache_built = False
    monkeypatch.setattr(decay_manager.lifecycle, "retrievability", lambda *a, **k: 0.0)

    with caplog.at_level("INFO", logger=decay_manager.logger.name):
        decay_manager.run_decay(nodes)

    tiers = {
        r["id"]: r["tier"] for r in nodes.db.conn.execute("SELECT id, tier FROM nodes")
    }
    assert tiers[UNIQUE_ID] == "archival"
    assert tiers[COLLIDER_A] == "working"
    assert "Decay manager failed" not in caplog.text


def test_audit_detail_survives_canonicalization(nodes):
    """The audit detail still lists the changed fields once the reference is canonical."""
    nodes.update_node(UNIQUE, UpdateNodeRequest(title="Renamed"))
    detail = nodes.db.conn.execute("SELECT detail FROM audit_log").fetchone()["detail"]
    assert json.loads(detail) == {"changed_fields": ["title"]}
