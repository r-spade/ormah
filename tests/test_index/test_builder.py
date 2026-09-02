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
    file_hash: L_db -> L_mem. Every background job apply step goes L_mem -> L_db (#240).
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
        """What every background job's apply step does: L_mem, then a write txn (#240)."""
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


def test_no_background_job_takes_l_mem_inside_a_write_transaction(engine):
    """Cross-cutting net for #240: the inversion #207 fixed must not come back.

    Every job now acquires and releases L_mem repeatedly instead of once, so this
    is not redundant with test_builder_never_takes_file_lock_inside_write_transaction
    above — that test covers the builder's own entry points, not the seven jobs.
    """
    import json
    from datetime import datetime, timedelta, timezone
    from unittest.mock import patch

    from ormah.background.auto_cluster import run_auto_cluster
    from ormah.background.auto_linker import run_auto_linker
    from ormah.background.conflict_detector import run_conflict_detection
    from ormah.background.consolidator import run_consolidation
    from ormah.background.decay_manager import run_decay
    from ormah.background.duplicate_merger import run_duplicate_detection
    from ormah.background.importance_scorer import run_importance_scoring
    from ormah.models.node import (
        ConnectRequest,
        CreateNodeRequest,
        EdgeType,
        NodeType,
        UpdateNodeRequest,
    )

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

    engine._memory_operation_lock = OrderProbe()
    engine.file_store._operation_lock = engine._memory_operation_lock

    # Count L_mem acquisitions per job so a job that quietly finds zero candidates
    # (and so never reaches an apply step) fails loudly instead of the assertion
    # below passing vacuously for it. OrderProbe.__enter__ already runs on every
    # acquisition, so piggyback the count there instead of a second wrapper.
    acquisitions = {"n": 0}
    real_enter = OrderProbe.__enter__

    def counting_enter(self):
        acquisitions["n"] += 1
        return real_enter(self)

    OrderProbe.__enter__ = counting_enter

    # remember() auto-links highly-similar nodes at creation time via the same
    # similarity threshold the background auto_linker job uses -- if left on, the
    # 5 near-identical seed nodes below end up fully pairwise-linked before the job
    # ever runs, so it would find zero candidates. Disable it for the seeding only.
    real_auto_link = engine._auto_link_node
    engine._auto_link_node = lambda node: []
    ids = []
    for i in range(5):
        nid, _ = engine.remember(CreateNodeRequest(
            content=f"seed node {i} about project architecture", type=NodeType.fact,
            title=f"seed {i}"))
        ids.append(nid)
    engine._auto_link_node = real_auto_link

    # ids[0] <-> ids[4] manually connected so auto_cluster has a spaced neighbor to
    # vote from; ids[1..3] stay unconnected so auto_linker has real, not-already-
    # linked, above-threshold pairs to classify.
    engine.connect(ConnectRequest(
        source_id=ids[0], target_id=ids[4], edge=EdgeType.related_to, weight=1.0))
    # ids[4] is already space=None from creation -- nothing to set there.
    # ids[0]'s space must go through update_node, not a raw SQL UPDATE: the
    # markdown file (loaded by file_store) is the source of truth, and any later
    # engine.update_node call (decay's own demotion, right below) reloads from
    # the file and re-persists it -- silently reverting a DB-only `space` back
    # to None and starving auto_cluster of a spaced neighbor to vote from.
    engine.update_node(ids[0], UpdateNodeRequest(space="architecture"))
    # A bare SQL `datetime('now', '-30 days')` literal produces a naive,
    # space-separated string; decay_manager's anchor parse then mixes it with a
    # tz-aware `now`, raises TypeError, and silently skips the node -- vacuous.
    # Use the same tz-aware ISO string test_decay_manager.py's _make_stale uses.
    # last_accessed is read straight from the SQLite index by decay's own
    # candidate scan and revalidation, so a raw SQL UPDATE (unlike space above)
    # is exactly what's needed here -- and update_node's later reload/resave for
    # the tier demotion doesn't touch it since UpdateNodeRequest carries no
    # last_accessed field.
    stale_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ?, tier = 'working' WHERE id = ?",
        (stale_date, ids[0]))
    engine.db.conn.commit()

    engine.settings.llm_provider = "ollama"
    engine.settings.consolidation_min_cluster_size = 2
    # Cap the linker's source scan so it doesn't mark every pair as auto_link_checked
    # before conflict/duplicate detection run -- those two also skip already-checked
    # pairs, and with an unbounded scan the linker would starve them of candidates.
    engine.settings.auto_link_max_nodes_per_run = 2

    fake_link = json.dumps({"relationship": "supports", "reason": "same topic"})
    fake_conflict_true = json.dumps({
        "same_subject": True, "conflict": True, "type": "tension", "explanation": "x"})
    fake_conflict_false = json.dumps({
        "same_subject": True, "conflict": False, "type": "none", "explanation": "n/a"})
    fake_dup_true = json.dumps({
        "is_duplicate": True, "merged_title": "merged", "merged_content": "merged content"})
    fake_dup_false = json.dumps({"is_duplicate": False, "reason": "distinct"})
    fake_consolidate = json.dumps({
        "title": "merged", "summary": "merged content", "type": "fact"})

    # conflict_detection and duplicate_detection only reach their apply step (the
    # engine.memory_operation_at block) on a positive verdict -- a fixed "false"
    # response (as consolidation and auto_linker's "none"/"supports" pattern might
    # suggest) would leave those two jobs' apply steps completely unexercised. Flip
    # to positive for the first candidate of each type only, so exactly one real
    # apply happens per job instead of a cascade of merges/edges eating the fixture.
    done = {"conflict": False, "dup": False}

    def fake_llm(*args, **kwargs):
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")
        # Match on phrases unique to each job's prompt -- "contradict" alone also
        # appears inside the auto_linker prompt's "contradicts" edge-type option,
        # which would misroute every linker call into the conflict branch.
        if "duplicates that should be merged" in prompt:
            if not done["dup"]:
                done["dup"] = True
                return fake_dup_true
            return fake_dup_false
        if "contradict each other" in prompt:
            if not done["conflict"]:
                done["conflict"] = True
                return fake_conflict_true
            return fake_conflict_false
        if "consolidating a cluster" in prompt:
            return fake_consolidate
        return fake_link

    try:
        with patch("ormah.background.llm_client.llm_generate", side_effect=fake_llm):
            # decay gets its own check instead of the shared acquisitions-count guard:
            # run_decay unconditionally opens L_mem once at the top of every call to
            # clear stale proposals (decay_manager.py's one-time cleanup), before it
            # ever scans for demotion candidates. That means acquisitions["n"] > before
            # would hold even if decay found nothing to demote -- it doesn't prove the
            # apply step (the actual tier demotion) ran. Assert the observable effect
            # instead: the seeded stale node (ids[0]) really moved to archival.
            run_decay(engine)
            tier_row = engine.db.conn.execute(
                "SELECT tier FROM nodes WHERE id = ?", (ids[0],)).fetchone()
            assert tier_row is not None and tier_row["tier"] == "archival", (
                "decay never demoted the seeded stale node -- its apply step was "
                "never reached, so this net covers nothing for it"
            )

            for name, run_job in [
                ("importance_scoring", run_importance_scoring),
                ("auto_cluster", run_auto_cluster),
                ("auto_linker", run_auto_linker),
                ("conflict_detection", run_conflict_detection),
                ("duplicate_detection", run_duplicate_detection),
                ("consolidation", run_consolidation),
            ]:
                before = acquisitions["n"]
                run_job(engine)
                assert acquisitions["n"] > before, (
                    f"{name} never acquired L_mem -- its apply step was never "
                    "reached, so this net covers nothing for it"
                )
    finally:
        OrderProbe.__enter__ = real_enter

    assert not violations, f"L_mem acquired inside db.transaction(): depths {violations}"
