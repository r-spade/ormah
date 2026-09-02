"""Tests for index builder."""

import threading

from ormah.index.builder import IndexBuilder
from ormah.models.node import CreateNodeRequest, MemoryNode, NodeType


def test_full_rebuild(db, file_store):
    # Create some nodes on disk
    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact {i} for indexing.",
            title=f"Fact {i}",
        )
        file_store.save(node)

    builder = IndexBuilder(db, file_store)
    count = builder.full_rebuild()
    assert count == 3

    # Verify in DB
    rows = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    assert rows[0] == 3


def test_incremental_update(db, file_store):
    builder = IndexBuilder(db, file_store)

    # Initial build
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Original content.",
        title="Original",
    )
    file_store.save(node)
    builder.full_rebuild()

    # Add another node
    node2 = MemoryNode(
        type=NodeType.decision,
        source="agent:test",
        content="New decision.",
        title="Decision",
    )
    file_store.save(node2)

    added, updated = builder.incremental_update()
    assert added == 1
    assert updated == 0

    rows = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    assert rows[0] == 2


# --- lock order (0.14.7+ restore-exclusion lock) ---


def test_incremental_update_does_not_deadlock_against_a_memory_job(engine):
    """incremental_update must take L_mem BEFORE L_db, like every memory job.

    The restore-exclusion lock decorates 8 FileStore methods with the engine's own RLock
    (L_mem) -- MemoryEngine passes it in: FileStore(nodes_dir, self._memory_operation_lock).
    incremental_update opens the write txn (L_db) and only then calls file_store.list_paths /
    file_hash: L_db -> L_mem. Every @serialized_memory_job background job goes L_mem -> L_db.
    Opposite orders on two locks = deadlock, and index_updater runs every 60s.
    """
    engine.remember(CreateNodeRequest(
        content="indexed content", type=NodeType.fact, title="indexed content"))

    builder_reached_file_store = threading.Event()
    job_holds_mem = threading.Event()
    real_list_paths = engine.file_store.list_paths

    def instrumented_list_paths():
        # Before the fix this runs INSIDE the write txn: this thread holds L_db and is one call
        # away from taking L_mem. Let the memory job grab L_mem first, then reach for it.
        # After the fix this runs BEFORE the txn, so no L_db is held and nothing can cycle.
        builder_reached_file_store.set()
        job_holds_mem.wait(timeout=1.0)  # times out once this thread already holds L_mem
        return real_list_paths()

    engine.file_store.list_paths = instrumented_list_paths

    def memory_job():
        """What @serialized_memory_job + a write txn do on every background job: L_mem, L_db."""
        builder_reached_file_store.wait(timeout=5.0)
        with engine._memory_operation_lock:
            job_holds_mem.set()
            with engine.db.transaction():
                pass

    builder_thread = threading.Thread(target=engine.builder.incremental_update, daemon=True)
    job_thread = threading.Thread(target=memory_job, daemon=True)
    builder_thread.start()
    job_thread.start()
    builder_thread.join(timeout=10.0)
    job_thread.join(timeout=10.0)

    assert not builder_thread.is_alive(), "incremental_update held L_db while waiting for L_mem"
    assert not job_thread.is_alive(), "memory job held L_mem while waiting for L_db"


def test_builder_never_takes_file_lock_inside_write_transaction(engine):
    """All builder entry points must finish FileStore calls before taking L_db."""
    real_lock = engine._memory_operation_lock
    violations: list[int] = []

    class OrderProbe:
        def __enter__(self):
            tx_depth = getattr(engine.db._local, "tx_depth", 0)
            if tx_depth > 0:
                violations.append(tx_depth)
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    engine.file_store._operation_lock = OrderProbe()

    # Exercise full rebuild and single-file indexing.
    engine.builder.full_rebuild()
    single = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="single content",
        title="single",
    )
    single_path = engine.file_store.save(single)
    engine.builder.index_single(single_path)

    # Exercise every incremental branch: new, changed, then deleted.
    incremental = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="new content",
        title="incremental",
    )
    engine.file_store.save(incremental)
    assert engine.builder.incremental_update() == (1, 0)

    incremental.content = "changed content"
    engine.file_store.save(incremental)
    assert engine.builder.incremental_update() == (0, 1)

    engine.file_store.delete(incremental.id)
    assert engine.builder.incremental_update() == (0, 0)

    assert not violations, f"FileStore lock acquired inside db.transaction(): {violations}"


def test_reindex_preserves_incoming_edges(engine):
    """Reindexing a node must not destroy the edges that point AT it (#123).

    The connection A -> B lives in A's markdown file. Reindexing B has no access to that
    file and therefore no way to reconstruct the row, so it must not delete it.
    """
    from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType

    id_a, _ = engine.remember(
        CreateNodeRequest(content="A fact.", type=NodeType.fact), agent_id="t")
    id_b, _ = engine.remember(
        CreateNodeRequest(content="Another fact.", type=NodeType.fact), agent_id="t")

    node_a = engine.file_store.load(id_a)
    node_a.connections.append(
        Connection(target=id_b, edge=EdgeType.supports, weight=0.9)
    )
    engine.file_store.save(node_a)
    engine.builder.index_single(engine.file_store._path_for(node_a))

    def incoming():
        return engine.db.conn.execute(
            "SELECT source_id, edge_type, weight FROM edges WHERE target_id = ?", (id_b,)
        ).fetchall()

    assert len(incoming()) == 1, "sanity: the edge must exist before B is reindexed"

    # Reindex the TARGET — what the index updater does after any change to B's own file.
    node_b = engine.file_store.load(id_b)
    engine.builder.index_single(engine.file_store._path_for(node_b))

    rows = incoming()
    assert len(rows) == 1, "incoming edge destroyed by reindexing the target (#123)"
    assert rows[0]["source_id"] == id_a
    assert rows[0]["edge_type"] == "supports"
    assert rows[0]["weight"] == 0.9, "weight 0.9 (not the 0.5 default) proves this is A's row"


def test_touch_updated_does_not_drop_incoming_edges(engine):
    """The real-world trigger: file_hash changes, content fingerprint does not (#123).

    `_invalidate_checked_pairs` only fires when the CONTENT fingerprint changes, but the
    reindex fires on any file_hash change. `touch_updated()` moves only `updated`, so the
    edge dies while the cached pair verdict survives — and auto_linker, conflict_detector
    and duplicate_merger all skip a pair already recorded in `auto_link_checked`. Nothing
    ever recreates the edge; the loss stands until a full rebuild.

    Self-feeding: the auto-linker touches the node before saving, so creating any new link
    on a node destroys that node's own incoming edges. Any job that rewrites a node's
    markdown and calls `touch_updated()` — the auto-linker and the importance scorer both
    do — drives this same reindex path, so the loss compounds across the whole store
    rather than needing a user edit.
    """
    from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType

    id_a, _ = engine.remember(
        CreateNodeRequest(content="A fact.", type=NodeType.fact), agent_id="t")
    id_b, _ = engine.remember(
        CreateNodeRequest(content="Another fact.", type=NodeType.fact), agent_id="t")

    node_a = engine.file_store.load(id_a)
    node_a.connections.append(
        Connection(target=id_b, edge=EdgeType.supports, weight=0.7)
    )
    engine.file_store.save(node_a)
    engine.builder.index_single(engine.file_store._path_for(node_a))

    def incoming_count():
        return engine.db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE target_id = ?", (id_b,)
        ).fetchone()[0]

    assert incoming_count() == 1, "sanity: the edge must exist before the touch"

    # The only delta is `updated`: file_hash changes, content fingerprint does not.
    node_b = engine.file_store.load(id_b)
    node_b.touch_updated()
    engine.file_store.save(node_b)
    engine.builder.index_single(engine.file_store._path_for(node_b))

    assert incoming_count() == 1, "touch_updated() destroyed the incoming edge (#123)"


def test_incremental_update_preserves_incoming_edges(engine):
    """The path production actually takes: the 60s index updater (#123).

    `index_single` is not the production trigger. `incremental_update` is — it walks the
    store, sees B's file_hash changed, and calls `_clear_derived(node.id)` at builder.py:104
    (before #123, this called `_remove_node(id, keep_vectors=True)`), a DIFFERENT call site
    from the one index_single uses (:122). A fix applied only to index_single leaves this
    path destroying incoming edges once a minute.
    """
    from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType

    id_a, _ = engine.remember(
        CreateNodeRequest(content="A fact.", type=NodeType.fact), agent_id="t")
    id_b, _ = engine.remember(
        CreateNodeRequest(content="Another fact.", type=NodeType.fact), agent_id="t")

    node_a = engine.file_store.load(id_a)
    node_a.connections.append(
        Connection(target=id_b, edge=EdgeType.supports, weight=0.6)
    )
    engine.file_store.save(node_a)
    engine.builder.index_single(engine.file_store._path_for(node_a))

    def incoming():
        return engine.db.conn.execute(
            "SELECT source_id, weight FROM edges WHERE target_id = ?", (id_b,)
        ).fetchall()

    assert len(incoming()) == 1, "sanity: the edge must exist before the updater runs"

    # Change B's file so the updater sees a new file_hash, then run the REAL trigger.
    node_b = engine.file_store.load(id_b)
    node_b.touch_updated()
    engine.file_store.save(node_b)
    added, updated = engine.builder.incremental_update()

    assert updated == 1, "sanity: the updater must have seen B as changed"
    rows = incoming()
    assert len(rows) == 1, "incremental_update destroyed the incoming edge (#123)"
    assert rows[0]["source_id"] == id_a
    assert rows[0]["weight"] == 0.6, "weight 0.6 (not the 0.5 default) proves this is A's row"


def test_incremental_update_preserves_the_node_vector(engine):
    """incremental_update must not drop the `node_vectors` row (#123).

    `_clear_derived` takes `drop_vector` because `incremental_update` never re-embeds after
    it — unlike `index_single`, whose callers always call `_index_embedding` afterwards. If
    `incremental_update` ever passed `drop_vector=True`, the node would lose its embedding
    and stay unsearchable by similarity until the next startup backfill picks it up.
    """
    from ormah.models.node import CreateNodeRequest, NodeType

    node_id, _ = engine.remember(
        CreateNodeRequest(content="A fact to embed.", type=NodeType.fact), agent_id="t")

    def has_vector() -> bool:
        return engine.db.conn.execute(
            "SELECT 1 FROM node_vectors WHERE id = ?", (node_id,)
        ).fetchone() is not None

    assert has_vector(), "sanity: remember() must embed the node before the update runs"

    # Change the node's file so the updater sees a new file_hash and reindexes it.
    node = engine.file_store.load(node_id)
    node.touch_updated()
    engine.file_store.save(node)
    added, updated = engine.builder.incremental_update()

    assert updated == 1, "sanity: the updater must have seen the node as changed"
    assert has_vector(), "incremental_update dropped the node_vectors row"


def test_removing_a_node_still_drops_its_incoming_edges(engine):
    """When the file is really gone, incoming edges MUST die (the mirror of #123).

    `edges.target_id` is `REFERENCES nodes(id) ON DELETE CASCADE`: an edge pointing at a
    node that no longer exists is a foreign-key violation. A fix that simply never deleted
    incoming edges would pass every other test in this file and leave orphan rows behind.
    """
    from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType

    id_a, _ = engine.remember(
        CreateNodeRequest(content="A fact.", type=NodeType.fact), agent_id="t")
    id_b, _ = engine.remember(
        CreateNodeRequest(content="Another fact.", type=NodeType.fact), agent_id="t")

    node_a = engine.file_store.load(id_a)
    node_a.connections.append(
        Connection(target=id_b, edge=EdgeType.supports, weight=0.9)
    )
    engine.file_store.save(node_a)
    engine.builder.index_single(engine.file_store._path_for(node_a))

    assert engine.db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE target_id = ?", (id_b,)
    ).fetchone()[0] == 1, "sanity: the edge must exist before B's file is deleted"

    # B genuinely leaves the store: its markdown file is gone from disk.
    path_b = engine.file_store._path_for(engine.file_store.load(id_b))
    path_b.unlink()
    engine.builder.incremental_update()

    assert engine.db.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE id = ?", (id_b,)
    ).fetchone()[0] == 0, "the removed node must be gone from the index"
    assert engine.db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE target_id = ?", (id_b,)
    ).fetchone()[0] == 0, "orphan edge survived the removal of its target"


def test_reindex_keeps_the_incumbent_canonical_direction(engine):
    """When both files declare the same link, the incumbent row wins — stably.

    `_index_file_edges` skips inserting A -> B when the reverse B -> A already exists with
    the same edge type (builder.py:226-232). Before #123 was fixed, reindexing B destroyed both
    directions, so B's own declaration was reinserted and the surviving direction was
    whichever node happened to be reindexed last. Now the incumbent survives and B's
    declaration is skipped: deterministic, and NOT a regression.

    Auto-linking is suppressed around `remember()` (same idiom as
    `test_mutation_stamping.py`): "A fact." and "Another fact." are similar enough that the
    real encoder crosses `auto_link_similarity_threshold`, which would otherwise plant an
    unrelated `related_to` B -> A edge (persisted into B's own markdown, same as any other
    connection) before this test ever declares its own `supports` connection -- a confound
    unrelated to the canonicalisation mechanism under test.

    The weights are deliberately inverted against the outcome: A, indexed first, gets the
    LOWER weight (0.2) and B gets the HIGHER weight (0.9), so A's row surviving can only be
    explained by index order, not by a hypothetical "higher weight wins" rule -- there is no
    weight comparison anywhere in `_index_file_edges` or `_clear_derived`.
    """
    from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = engine.remember(
            CreateNodeRequest(content="A fact.", type=NodeType.fact), agent_id="t")
        id_b, _ = engine.remember(
            CreateNodeRequest(content="Another fact.", type=NodeType.fact), agent_id="t")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    node_a = engine.file_store.load(id_a)
    node_a.connections.append(
        Connection(target=id_b, edge=EdgeType.supports, weight=0.2)
    )
    engine.file_store.save(node_a)

    node_b = engine.file_store.load(id_b)
    node_b.connections.append(
        Connection(target=id_a, edge=EdgeType.supports, weight=0.9)
    )
    engine.file_store.save(node_b)

    # A is indexed first, so A -> B becomes the incumbent row.
    engine.builder.index_single(engine.file_store._path_for(node_a))
    engine.builder.index_single(engine.file_store._path_for(node_b))

    rows = engine.db.conn.execute(
        "SELECT source_id, weight FROM edges "
        "WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
        (id_a, id_b, id_b, id_a),
    ).fetchall()

    assert len(rows) == 1, "the pair must be represented by exactly one canonical row"
    assert rows[0]["source_id"] == id_a, "reindexing B flipped the canonical direction"
    assert rows[0]["weight"] == 0.2, "A's weight, not B's 0.9 — the incumbent's row is the one kept"
