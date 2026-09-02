"""Integer lifecycle-model version replaces the boolean fsrs_migrated flag (#221)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from ormah.engine import memory_engine
from ormah.engine.memory_engine import LIFECYCLE_MODEL_VERSION
from ormah.models.node import CreateNodeRequest, MemoryNode, NodeType, Tier
from ormah.store.markdown import parse_node, serialize_node


def _version(engine) -> str | None:
    row = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'lifecycle_model_version'"
    ).fetchone()
    return row["value"] if row else None


def _make_node(engine) -> str:
    """A real node, so the seed has something to act on — an empty table would
    make every assertion below pass vacuously."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="A node the migration will or will not reseed",
        type=NodeType.fact,
        tier=Tier.working,
        title="Migration subject",
    ))
    return node_id


def _stability(engine, node_id: str) -> float:
    return engine.db.conn.execute(
        "SELECT stability FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()["stability"]


def test_the_version_constant_is_two():
    """1 = the legacy FSRS seed; 2 = bounded reinforcement."""
    assert LIFECYCLE_MODEL_VERSION == 2


def test_a_fresh_store_ends_at_the_current_version(engine):
    assert _version(engine) == "2"


def test_a_legacy_store_is_backfilled_without_reseeding(engine):
    """AC6: fsrs_migrated='1' means the seed already ran — record it as version 1."""
    node_id = _make_node(engine)
    # Simulate a store migrated by the old boolean flag: drop the new key, restore
    # the legacy one, and set a stability the seed would overwrite if it re-ran.
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'lifecycle_model_version'")
    engine.db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('fsrs_migrated', '1')"
    )
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 42.0, access_count = 5 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    engine._migrate_fsrs()

    assert _version(engine) == "2"
    assert _stability(engine, node_id) == 42.0, "the seed re-ran on an already-migrated store"


def test_an_unmigrated_store_still_gets_seeded(engine):
    node_id = _make_node(engine)
    # The seed now computes from the loaded Markdown, not the SQLite row
    # (council-pr, round 1, second pass: Codex medium@0.97) -- access_count has
    # to be real on disk, or the fixture proves nothing but its own staleness.
    node = engine.file_store.load(node_id)
    node.access_count = 5
    node.stability = 1.0
    node.last_review = None
    engine.file_store.save(node)
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'lifecycle_model_version'")
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'fsrs_migrated'")
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, access_count = 5 WHERE id = ?", (node_id,)
    )
    # A genuinely pre-FSRS store: no node has ever been seeded or reinforced.
    # Stated explicitly rather than assumed — this is what separates it from the
    # rebuilt-index case below, and the whole seed decision now rests on it.
    engine.db.conn.execute("UPDATE nodes SET last_review = NULL")
    engine.db.conn.commit()

    engine._migrate_fsrs()

    assert _version(engine) == "2"
    assert _stability(engine, node_id) == 10.0, "min(30, access_count * 2) was not applied"


def test_a_corrupt_version_fails_closed_as_migrated(engine):
    """Skipping a seed is inert; running one overwrites stability. Fail closed."""
    engine.db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('lifecycle_model_version', 'banana')"
    )
    engine.db.conn.commit()

    assert engine._lifecycle_model_version() == 1


def test_a_rebuilt_index_does_not_reseed_earned_stability(engine):
    """C3: SQLite is excluded from backups, so a restore arrives with no meta.

    The Markdown still carries stability and last_review, so the store is not a
    pre-FSRS one and must not be reseeded.

    SCOPE: this covers the EMPTY-meta case at the _migrate_fsrs level only —
    it deletes the keys by hand and never goes through full_rebuild or
    reload_restored_graph. The same-device restore, where meta used to survive
    the rebuild, is covered by test_a_restore_onto_an_existing_index_* below
    (#236).
    """
    node_id = _make_node(engine)
    now = datetime.now(timezone.utc)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, access_count = 7, last_review = ? WHERE id = ?",
        (now.isoformat(), node_id),
    )
    # Exactly what a fresh-device restore or a deleted index looks like.
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'lifecycle_model_version'")
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'fsrs_migrated'")
    engine.db.conn.commit()

    engine._migrate_fsrs()

    # Without the last_review guard the seed would write min(30, 7*2) = 14.0.
    assert _stability(engine, node_id) == 1.0
    assert _version(engine) == "2"


def test_the_legacy_flag_is_written_alongside_the_version(engine):
    """I1: a rollback to a binary that only knows fsrs_migrated must not reseed.

    SCOPE, stated so a green is not over-read (council round 3, C2): this proves
    the old binary will not RESEED. It does not prove rollback is safe. The old
    binary still writes stability with the unbounded formula while leaving
    lifecycle_model_version at '2', and on the next upgrade the early return
    trusts that stale marker. Downgrade is unsupported; see the scope note above.
    """
    row = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'fsrs_migrated'"
    ).fetchone()
    assert row is not None and row["value"] == "1"


def _unmigrated_meta(engine) -> None:
    """Put the store back to 'no lifecycle marker', the way full_rebuild does."""
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'lifecycle_model_version'")
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'fsrs_migrated'")
    engine.db.conn.commit()


def _write_lifecycle(engine, node_id: str, *, stability: float,
                     access_count: int, last_review) -> None:
    """Set a node's lifecycle fields in the DURABLE source, then reindex.

    Council round 3, Cursor F3: a fixture that only UPDATEs SQLite proves
    nothing about restore, because full_rebuild reindexes from Markdown and
    throws the DB row away. Everything these tests assert about a restored
    graph has to start on disk.
    """
    node = engine.file_store.load(node_id)
    node.stability = stability
    node.access_count = access_count
    node.last_review = last_review
    engine.file_store.save(node)
    engine.builder.full_rebuild()


def test_the_seed_never_overwrites_stability_it_did_not_produce(engine):
    """Codex F1: externally authored Markdown may carry a real stability with
    no last_review. The invariant that every Ormah writer stamps last_review
    says nothing about such a file, so eligibility must be decided per node."""
    node_id = _make_node(engine)
    _write_lifecycle(engine, node_id, stability=42.0, access_count=5, last_review=None)
    _unmigrated_meta(engine)

    engine._migrate_fsrs()

    assert _stability(engine, node_id) == 42.0, "the seed destroyed a stability it did not write"
    assert _version(engine) == "2"


def test_a_mixed_store_seeds_only_its_unreviewed_nodes(engine):
    """Codex F2: one migrated node must not suppress the seed for the rest."""
    reviewed = _make_node(engine)
    unreviewed = _make_node(engine)
    _write_lifecycle(engine, reviewed, stability=42.0, access_count=5,
                     last_review=datetime.now(timezone.utc))
    _write_lifecycle(engine, unreviewed, stability=1.0, access_count=5, last_review=None)
    _unmigrated_meta(engine)

    engine._migrate_fsrs()

    assert _stability(engine, reviewed) == 42.0, "an earned value was reseeded"
    assert _stability(engine, unreviewed) == 10.0, "a pre-FSRS node was skipped"


def test_a_node_with_no_usage_history_is_left_alone(engine):
    """access_count = 0 carries no signal, so the seed must not touch the node
    — not its stability, and not its last_review."""
    node_id = _make_node(engine)
    _write_lifecycle(engine, node_id, stability=1.0, access_count=0, last_review=None)
    _unmigrated_meta(engine)

    engine._migrate_fsrs()

    row = engine.db.conn.execute(
        "SELECT stability, last_review FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    assert row["stability"] == 1.0
    assert row["last_review"] is None, "the seed stamped a node it did not change"


def test_an_interrupted_seed_resumes_on_the_next_run(engine):
    """Codex F3: an interrupted seed leaves no version marker behind, and a
    subsequent run converges every eligible node on the correct stability.
    The seed formula is deterministic and the write idempotent, so this holds
    whether or not a node was already (re)written by the failed attempt."""
    first = _make_node(engine)
    second = _make_node(engine)
    for node_id in (first, second):
        _write_lifecycle(engine, node_id, stability=1.0, access_count=5, last_review=None)
    _unmigrated_meta(engine)

    real_save = engine.file_store.save
    saved: list[str] = []

    def save_once_then_fail(node):
        if saved:
            raise OSError("disk full")
        saved.append(node.id)
        return real_save(node)

    engine.file_store.save = save_once_then_fail
    with pytest.raises(OSError):
        engine._migrate_fsrs()
    engine.file_store.save = real_save

    assert _version(engine) is None, "a version was recorded over a failed seed"

    engine._migrate_fsrs()

    assert _stability(engine, first) == 10.0
    assert _stability(engine, second) == 10.0
    assert _version(engine) == "2"


def test_no_version_is_recorded_while_a_file_is_missing_from_the_index(engine):
    """Council round 2 (Codex F2) / round 3 (Cursor F1): recording version 2
    over a graph that did not fully index strands every node that only lands on
    a later incremental pass. The check has to live here, because startup() and
    BackupService.rebuild_index call this method without consulting the builder."""
    node_id = _make_node(engine)
    _write_lifecycle(engine, node_id, stability=1.0, access_count=5, last_review=None)
    _unmigrated_meta(engine)
    # A file on disk that never made it into the index — exactly the shape
    # full_rebuild leaves behind when a file fails to hash, parse, or index.
    (engine.file_store.nodes_dir / "broken.md").write_text("not: [valid", encoding="utf-8")

    engine._migrate_fsrs()

    assert _version(engine) is None, "a version was recorded over an incomplete graph"
    assert _stability(engine, node_id) == 10.0, "indexed nodes were not seeded"

    (engine.file_store.nodes_dir / "broken.md").unlink()
    engine._migrate_fsrs()

    assert _version(engine) == "2"


def _rewrite_as_pre_fsrs(engine, node_id: str, access_count: int) -> None:
    """Rewrite one node's Markdown the way a pre-FSRS store would have it:
    no last_review, default stability, but a real usage history."""
    node = engine.file_store.load(node_id)
    node.last_review = None
    node.stability = 1.0
    node.access_count = access_count
    engine.file_store.save(node)


def test_a_restore_onto_an_existing_index_seeds_pre_fsrs_nodes(engine):
    """#236: the store was already migrated (version 2 in meta), then the
    Markdown is replaced by pre-FSRS nodes and the graph is reloaded. The
    surviving marker must not short-circuit the seed."""
    node_id = _make_node(engine)
    assert _version(engine) == "2"
    _rewrite_as_pre_fsrs(engine, node_id, access_count=5)

    engine.reload_restored_graph()

    assert _stability(engine, node_id) == 10.0, "min(30, access_count * 2) was not applied"
    assert _version(engine) == "2"
    assert engine.file_store.load(node_id).stability == 10.0, "Markdown was not reseeded"


def test_a_restore_onto_an_existing_index_keeps_earned_stability(engine):
    """The inverse of the case above: a store that carries last_review is not
    pre-FSRS, so reloading must leave earned stability alone and simply
    re-record the version."""
    node_id = _make_node(engine)
    node = engine.file_store.load(node_id)
    node.stability = 42.0
    node.access_count = 5
    node.last_review = datetime.now(timezone.utc)
    engine.file_store.save(node)

    engine.reload_restored_graph()

    assert _stability(engine, node_id) == 42.0, "the seed re-ran on a migrated store"
    assert _version(engine) == "2"


def test_an_admin_rebuild_re_records_the_version(engine):
    """rebuild_index is also reachable from the admin route; it must leave the
    store at the current version, not with the keys wiped by full_rebuild."""
    _make_node(engine)

    engine.rebuild_index()

    assert _version(engine) == "2"


def test_rebuild_index_holds_the_memory_lock_across_the_seeds_file_write(engine):
    """#236: rebuild_index must hold _memory_operation_lock for its whole body.

    _seed_stability_from_access_count writes to file_store.save() from INSIDE
    engine.db.transaction() -- the inverse of the order every other memory job
    uses (memory lock, then db transaction; see _record_confirmed_use's
    docstring). @_serialized_memory_operation on rebuild_index is what turns
    that inverted db -> file order back into memory -> db -> file, the same
    order the rest of the engine relies on to avoid two operations taking the
    two locks in opposite sequences and deadlocking. Remove the decorator and
    the seed's file_store.save() runs with no outer lock held at all.

    A reentrant lock is always acquirable from the thread that already holds
    it, so testing "is the lock held" from the test's own thread would pass
    vacuously regardless of whether rebuild_index acquired it. The only
    reliable signal is a non-blocking acquire attempt from a different
    thread: it succeeds iff nobody holds the lock.
    """
    node_id = _make_node(engine)
    _write_lifecycle(engine, node_id, stability=1.0, access_count=5, last_review=None)
    _unmigrated_meta(engine)

    real_save = engine.file_store.save
    calls: list[bool] = []

    def probe_lock_held_by_another_thread() -> bool:
        result: list[bool] = []

        def attempt():
            acquired = engine._memory_operation_lock.acquire(blocking=False)
            if acquired:
                engine._memory_operation_lock.release()
            result.append(not acquired)

        probe = threading.Thread(target=attempt)
        probe.start()
        probe.join(timeout=5.0)
        assert not probe.is_alive(), "lock probe thread did not finish"
        return result[0]

    def save_and_record_lock_state(node):
        calls.append(probe_lock_held_by_another_thread())
        return real_save(node)

    engine.file_store.save = save_and_record_lock_state
    try:
        engine.rebuild_index()
    finally:
        engine.file_store.save = real_save

    assert calls, "file_store.save was never reached -- the test proves nothing"
    assert all(calls), (
        "rebuild_index's memory lock was not held while the seed wrote to "
        "file_store.save() -- the db-transaction-then-file-write order is "
        "no longer wrapped in the engine's memory -> db -> file sequence"
    )


def test_a_node_indexed_after_the_migration_is_still_seeded(engine):
    """#236, council round 4 (Codex, high @ 0.98): the store-wide marker must
    not gate the per-node seed. A pre-FSRS Markdown node dropped into nodes/ by
    an external tool and picked up by incremental_update arrives AFTER the store
    already records version 2. Under the old early return it kept stability 1.0
    forever, so decay treated a well-used memory as brand new."""
    assert _version(engine) == "2"

    latecomer = MemoryNode(
        type=NodeType.fact,
        title="Latecomer",
        content="Used five times before FSRS existed",
        access_count=5,
        stability=1.0,
        last_review=None,
    )
    engine.file_store.save(latecomer)
    engine.builder.incremental_update()

    engine._migrate_fsrs()

    assert _stability(engine, latecomer.id) == 10.0, (
        "the store-wide version marker short-circuited the per-node seed"
    )
    assert engine.file_store.load(latecomer.id).stability == 10.0, "Markdown was not reseeded"
    assert _version(engine) == "2", "the marker should be left exactly as it was"


def test_equal_counts_with_different_membership_withhold_the_version(engine):
    """#236, council round 4 (Codex, high @ 0.97): len(list_paths()) ==
    COUNT(*) proves cardinality, not membership. An indexed file replaced
    externally by a different node while the process was stopped leaves both
    sides at the same count while the id sets diverge. Recording the version
    over that graph asserts a completed migration that never happened."""
    node_id = _make_node(engine)
    _unmigrated_meta(engine)

    # Swap one indexed file for an unindexed stranger: one file out, one file
    # in, so the counts stay identical and only the membership changes.
    target = next(
        path for path in engine.file_store.list_paths()
        if parse_node(path.read_text(encoding="utf-8")).id == node_id
    )
    target.unlink()
    stranger = MemoryNode(
        type=NodeType.fact,
        title="Stranger",
        content="On disk, never indexed",
    )
    engine.file_store.save(stranger)

    on_disk = len(engine.file_store.list_paths())
    indexed = engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert on_disk == indexed, "the fixture must keep the counts equal, or it proves nothing"

    engine._migrate_fsrs()

    assert _version(engine) is None, (
        "a version was recorded over a graph whose membership does not match the index"
    )


def test_two_files_sharing_one_id_withhold_the_version(engine):
    """#236, council round 5 (Codex, high @ 0.99): a set of parsed ids discards
    multiplicity. Two Markdown files carrying the same node.id collapse to one
    entry while SQLite holds one row, so bare set equality calls the graph
    complete -- a case the cardinality check this task replaces used to reject."""
    node_id = _make_node(engine)
    _unmigrated_meta(engine)

    # The store already holds the Self node, so assert against the baseline
    # rather than absolute counts.
    indexed = engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert len(engine.file_store.list_paths()) == indexed, (
        "the fixture must start from a balanced graph, or it proves nothing"
    )

    # A duplicate the way an external tool makes one: same id, different title,
    # therefore a different filename, written straight to disk so FileStore's
    # _path_for cannot reuse the existing file.
    original = next(
        path for path in engine.file_store.list_paths()
        if parse_node(path.read_text(encoding="utf-8")).id == node_id
    )
    node = parse_node(original.read_text(encoding="utf-8"))
    node.title = "Renamed by an external tool"
    duplicate = original.parent / f"fact_renamed-by-an-external-tool_{node.short_id}.md"
    duplicate.write_text(serialize_node(node), encoding="utf-8")

    assert len(engine.file_store.list_paths()) == indexed + 1, (
        "the duplicate must add one file and no row, or it proves nothing"
    )
    assert engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == indexed

    engine._migrate_fsrs()

    assert _version(engine) is None, (
        "a version was recorded over a graph holding two files for one id"
    )


def test_a_store_at_the_current_version_never_parses_the_graph(engine, monkeypatch):
    """#236, council round 5 (Cursor, medium @ 0.93): the early return must sit
    above the completeness guard. Nothing else pins that ordering -- the
    latecomer test stays green either way -- and losing it puts a full-store
    Markdown parse on every startup(), synchronous in the lifespan."""
    assert _version(engine) == "2"

    # Count calls rather than raise: _graph_is_fully_indexed wraps the parse in
    # `except Exception`, which swallows AssertionError and fails closed, so a
    # raising sentinel is invisible to the test and pins nothing.
    calls: list[str] = []
    real_parse = memory_engine.parse_node

    def _spy(text):
        calls.append(text)
        return real_parse(text)

    monkeypatch.setattr(memory_engine, "parse_node", _spy)

    engine._migrate_fsrs()

    assert not calls, (
        "a store at the current version parsed the graph: the early return no "
        "longer sits above the completeness guard, so every startup() now pays "
        "a full-store Markdown parse"
    )
    assert _version(engine) == "2"


def test_a_stale_index_row_does_not_overwrite_newer_markdown(engine):
    """council-pr, round 1 (Codex high@0.99, Cursor high@0.90, converged
    independently): eligibility comes from SQLite; the seed used to write to
    the loaded Markdown unconditionally. If the index goes stale relative to
    disk -- the server was stopped while an external tool wrote a real
    stability into a node the index still shows as pre-FSRS -- startup() with
    a non-empty index never refreshes before _migrate_fsrs() runs, so the
    stale row destroyed the newer, earned value."""
    node_id = _make_node(engine)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, last_review = NULL, access_count = 5 "
        "WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    # An external edit lands on disk with a real earned value while the index
    # still reflects the pre-FSRS shape above.
    node = engine.file_store.load(node_id)
    node.stability = 42.0
    node.last_review = None
    engine.file_store.save(node)

    engine._migrate_fsrs()

    on_disk = engine.file_store.load(node_id)
    assert on_disk.stability == 42.0, (
        "a stale index row destroyed a newer Markdown value"
    )
    # The DB row is left stale at 1.0 here -- reconciling it to the disk value
    # is the index refresh's job (incremental_update / index_updater), not the
    # seed's. What must not happen is the seed fabricating a *third* value on
    # either side, e.g. min(30, access_count * 2) = 10.0.
    assert _stability(engine, node_id) != 10.0, (
        "the seed wrote its own formula's value into the DB over a node it "
        "correctly left alone on disk"
    )


def test_stale_index_access_count_does_not_drive_the_formula(engine):
    """council-pr, round 1, second pass (Codex medium@0.97): the earlier
    re-check protects stability/last_review from being overwritten, but the
    seed still computed its VALUE from the stale SQLite row's access_count.
    An external tool that advances access_count on Markdown without touching
    stability/last_review leaves the row eligible while the formula's input
    is already wrong -- the write must come from the loaded node's own
    access_count, not the row that only decided the node was worth loading."""
    node_id = _make_node(engine)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, last_review = NULL, access_count = 5 "
        "WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    node = engine.file_store.load(node_id)
    node.access_count = 10
    # Set the pre-FSRS shape on disk explicitly rather than relying on the
    # model default: fsrs_initial_stability is a settable knob, and a store
    # whose remember() seeds a non-1.0 initial stability would make this
    # fixture silently prove nothing.
    node.stability = 1.0
    node.last_review = None
    engine.file_store.save(node)

    engine._migrate_fsrs()

    on_disk = engine.file_store.load(node_id)
    assert on_disk.stability == 20.0, (
        f"seeded from the stale SQLite access_count (would give 10.0) instead "
        f"of the disk's own (20.0): got {on_disk.stability}"
    )
