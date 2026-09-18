"""Reversible promotion: archival nodes return to working on confirmed use (#223)."""

from __future__ import annotations

from ormah.models.node import (
    ConnectRequest,
    CreateNodeRequest,
    EdgeType,
    MemoryNode,
    NodeType,
    Tier,
)


def _archive(engine, content: str, superseded_by: str | None = None) -> str:
    node_id, _ = engine.remember(CreateNodeRequest(content=content))
    node = engine.file_store.load(node_id)
    node.tier = Tier.archival
    node.superseded_by = superseded_by
    engine.builder.index_single(engine.file_store.save(node))
    return node_id


def test_recall_node_promotes_exactly_the_requested_node(engine):
    """The archival NEIGHBOUR is the point: without it, an implementation that
    promotes every id in whisper_log_ids passes — and recall_node DOES create
    whisper_log rows for neighbours, in _log_feedback_candidates."""
    a = _archive(engine, "the node actually recalled")
    b = _archive(engine, "a connected neighbour that must stay archival")
    engine.connect(ConnectRequest(source_id=a, target_id=b, edge=EdgeType.related_to))

    engine.recall_node(a)

    assert engine.file_store.load(a).tier is Tier.working
    assert engine.file_store.load(b).tier is Tier.archival


def test_a_generic_derived_from_target_still_promotes(engine):
    """derived_from is a general relationship; only marked sources are blocked.
    Testing only the marked node misses the block-everything bug the issue names.
    Both `plain` and `marked` are genuine derived_from targets (each has an
    incoming derived_from edge from a parent); only the superseded_by marker
    tells them apart, so a gate that keys off the edge instead of the marker
    would block both and fail this test.

    The marker must point at a node that really exists: a dangling marker no longer
    blocks (see test_a_dangling_superseded_by_no_longer_blocks_promotion)."""
    parent = _archive(engine, "a parent that derives both nodes below")
    plain = _archive(engine, "a derived_from target that was never superseded")
    replacement, _ = engine.remember(CreateNodeRequest(content="the consolidation node"))
    marked = _archive(engine, "a genuinely superseded source", superseded_by=replacement)
    engine.connect(ConnectRequest(source_id=parent, target_id=plain, edge=EdgeType.derived_from))
    engine.connect(ConnectRequest(source_id=parent, target_id=marked, edge=EdgeType.derived_from))

    engine._record_confirmed_use(plain)
    engine._record_confirmed_use(marked)

    assert engine.file_store.load(plain).tier is Tier.working
    assert engine.file_store.load(marked).tier is Tier.archival


def test_a_superseded_node_is_not_promoted_but_still_tracks_access(engine):
    """Blocking promotion must not silently swallow the access bookkeeping."""
    replacement, _ = engine.remember(CreateNodeRequest(content="the consolidation node"))
    marked = _archive(engine, "superseded but still read", superseded_by=replacement)
    before = engine.file_store.load(marked).access_count

    engine._record_confirmed_use(marked)

    after = engine.file_store.load(marked)
    assert after.tier is Tier.archival
    assert after.access_count == before + 1


def test_a_working_node_is_left_alone(engine):
    """promote() guards tier ordering; a working node must not be touched by the branch."""
    node_id, _ = engine.remember(CreateNodeRequest(content="already working"))

    engine._record_confirmed_use(node_id)

    assert engine.file_store.load(node_id).tier is Tier.working


def test_a_dangling_superseded_by_no_longer_blocks_promotion(engine):
    """Self-healing against permanent burial (#192 scenario): the consolidation node
    the marker points at was deleted, so nothing is left to represent this memory.
    The block is lifted on the next confirmed use — but the marker itself stays,
    because it is the provenance record, not the lock."""
    dangling = "00000000-0000-0000-0000-000000000000"
    marked = _archive(engine, "superseded by a node that no longer exists", superseded_by=dangling)

    engine._record_confirmed_use(marked)

    after = engine.file_store.load(marked)
    assert after.tier is Tier.working
    assert after.superseded_by == dangling


def test_a_live_superseded_by_still_blocks_promotion(engine):
    """The discriminator is whether the consolidation node is still loadable, pinned
    directly next to the dangling case: same marker field, opposite outcome."""
    replacement, _ = engine.remember(CreateNodeRequest(content="the live consolidation node"))
    marked = _archive(engine, "a genuinely superseded source", superseded_by=replacement)

    engine._record_confirmed_use(marked)

    assert engine.file_store.load(marked).tier is Tier.archival


def test_a_prefix_collision_is_not_mistaken_for_a_live_replacement(engine):
    """`load(marker) is not None` is not the same question as "the replacement exists".

    FileStore._find_file resolves a Node reference by Full id, not by the
    8-character Short id in the filename alone (#280): an absent Full id whose
    Short id collides with an unrelated node's filename must resolve to None, not
    to that unrelated node.  If it resolved to the collider instead, a dangling
    `superseded_by` marker would read as live, the block would never lift, and the
    memory would be buried forever — the failure the dangling-marker branch exists
    to prevent."""
    collider = MemoryNode(
        id="deadbeef-0000-0000-0000-00000000000b",
        type=NodeType.fact,
        content="an unrelated node that happens to share the first eight characters",
    )
    engine.builder.index_single(engine.file_store.save(collider))

    gone = "deadbeef-0000-0000-0000-00000000000a"
    assert engine.file_store.load(gone) is None, (
        "a dangling Full id whose Short id collides with a live node's filename "
        "must not resolve to that unrelated node"
    )

    marked = _archive(engine, "superseded by a node that no longer exists", superseded_by=gone)

    engine._record_confirmed_use(marked)

    after = engine.file_store.load(marked)
    assert after.tier is Tier.working
    assert after.superseded_by == gone


def test_promotion_is_written_to_the_index_too(engine):
    node_id = _archive(engine, "must land in SQL as well")

    engine._record_confirmed_use(node_id)

    row = engine.db.conn.execute(
        "SELECT tier FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    assert row["tier"] == "working"
