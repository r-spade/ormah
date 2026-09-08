"""Regression tests for issue #29: a Node reference must resolve only to the
node it actually names, never to another node whose Short id happens to
collide with it.

FileStore._find_file resolves a Node reference (Full id or Short id) through
a glob on the 8-char Short id suffix in the filename and accepts the first
match without ever opening the file to confirm its Full id. Two nodes whose
ids share the same first 8 hex characters (same "-" group) therefore collide:
whichever file the glob happens to return wins, silently, and the wrong path
gets cached under the requested id.

Every assertion here goes through the FileStore public surface (save / load
/ delete / soft_delete) except two declared exceptions: whether an absent id
was left out of `_id_cache` is not externally observable, so that one case
pokes `store._id_cache` directly — same pattern as
test_file_cache.py::test_stale_cache_entry_recovers — and the store exposes no
reader for tombstones, so the soft-delete case reads the `deleted/` file itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import frontmatter

from ormah.models.node import MemoryNode, NodeType, Tier
from ormah.store.markdown import serialize_node


COLLIDING_SHORT_ID = "deadbeef"


def _collider(suffix: str, title: str, content: str) -> MemoryNode:
    """A node whose id is forced to start with the shared Short id prefix."""
    return MemoryNode(
        id=f"{COLLIDING_SHORT_ID}-0000-0000-0000-00000000000{suffix}",
        type=NodeType.fact,
        tier=Tier.working,
        title=title,
        content=content,
        source="test",
    )


def test_deleting_one_of_a_colliding_pair_does_not_resurrect_it_as_the_other(file_store):
    """Save two nodes that share a Short id, soft-delete the first, then ask for
    it by its own Full id: it must come back None, not the survivor's node.

    Before the fix, once the first node's file is moved out of nodes/, the
    Short-id glob for its id has only the second node's file left to match
    and returns that instead — the first node reads as still alive, as the
    wrong node.
    """
    first = _collider("a", "Collision A", "first colliding node")
    second = _collider("b", "Collision B", "second colliding node")
    file_store.save(first)
    file_store.save(second)

    assert file_store.soft_delete(first.id) is True
    file_store.save(second)  # second stays live and freshly written on disk

    assert file_store.load(first.id) is None

    deleted_dir = file_store.nodes_dir.parent / "deleted"
    tombstones = sorted(deleted_dir.glob("*.md"))
    assert len(tombstones) == 1
    meta = frontmatter.loads(tombstones[0].read_text(encoding="utf-8")).metadata
    assert meta["id"] == first.id
    assert meta.get("deleted_at") is not None


def test_absent_full_id_sharing_a_short_id_with_a_live_node_resolves_to_none(file_store):
    """An id that was never saved must not resolve just because some other,
    unrelated node's file happens to share its Short id — and the miss must
    not be cached, so a second lookup is not silently served the wrong node
    either."""
    live = _collider("a", "Live node", "the only node actually on disk")
    file_store.save(live)

    absent_id = f"{COLLIDING_SHORT_ID}-ffff-ffff-ffff-ffffffffffff"

    assert file_store.load(absent_id) is None
    assert absent_id not in file_store._id_cache  # declared exception: not observable via load()

    assert file_store.load(absent_id) is None


def test_short_id_matching_exactly_one_node_resolves_via_load(file_store):
    """The everyday path: a Short id with a single match still works."""
    node = MemoryNode(
        type=NodeType.fact,
        tier=Tier.working,
        title="Solo node",
        content="the only node with this short id",
        source="test",
    )
    file_store.save(node)

    loaded = file_store.load(node.short_id)

    assert loaded is not None
    assert loaded.id == node.id


def test_a_unique_short_id_stops_resolving_once_a_collider_is_born(file_store):
    """A Short id is never a cache key, so a second node sharing it turns a
    previously unique reference ambiguous on the very next call.

    This is the transition the original bug lived in: `_find_file` used to cache
    the *requested* key, so the first lookup's answer outlived the arrival of the
    collider and kept naming the first node forever.
    """
    first = _collider("a", "First node", "the only node with this short id, at first")
    file_store.save(first)

    assert file_store.load(COLLIDING_SHORT_ID).id == first.id
    assert COLLIDING_SHORT_ID not in file_store._id_cache  # declared exception, as above

    file_store.save(_collider("b", "Second node", "the collider that arrives later"))

    assert file_store.load(COLLIDING_SHORT_ID) is None


def test_an_ambiguous_short_id_resolves_again_once_only_one_node_is_left(file_store):
    """The inverse transition: ambiguity is recomputed, never remembered.

    A negative cache would keep answering None after the pair became a single
    node, losing a resolution that is valid again.
    """
    first = _collider("a", "Collision A", "first colliding node")
    second = _collider("b", "Collision B", "second colliding node")
    file_store.save(first)
    file_store.save(second)

    assert file_store.load(COLLIDING_SHORT_ID) is None

    assert file_store.delete(first.id) is True

    survivor = file_store.load(COLLIDING_SHORT_ID)
    assert survivor is not None
    assert survivor.id == second.id


def test_a_read_failure_while_deleting_does_not_leave_the_id_resolvable(file_store, monkeypatch):
    """`delete` must drop the cache entry even when the file will not read.

    The cache is keyed by Full id, but the key cannot come from re-reading the
    file: a transient OSError there used to be swallowed, leaving the entry
    pointing at a path `delete` then unlinks. `_path_for` is deterministic, so
    the next node with the same type, title and Short id lands on that exact
    path — and the existence-only cache hit hands the deleted id the new node.
    """
    doomed = _collider("a", "Same title", "the node being deleted")
    file_store.save(doomed)
    file_store.load(doomed.id)  # warm the cache so the entry exists to go stale
    assert file_store._id_cache[doomed.id].exists()

    real_read_text = Path.read_text

    def read_text_failing_once(self, *args, **kwargs):
        if self.name == file_store._id_cache[doomed.id].name:
            raise OSError("transient read failure")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text_failing_once)
    assert file_store.delete(doomed.id) is True
    monkeypatch.undo()

    # Same type, same title, same Short id — so _path_for regenerates the freed path.
    successor = _collider("b", "Same title", "an unrelated node reusing the filename")
    file_store.save(successor)

    assert file_store.load(doomed.id) is None


def test_short_id_matching_two_nodes_resolves_to_none_and_warns(file_store, caplog):
    """An ambiguous Short id resolves to nothing (ADR-0007), loudly."""
    first = _collider("a", "Collision A", "first colliding node")
    second = _collider("b", "Collision B", "second colliding node")
    file_store.save(first)
    file_store.save(second)

    with caplog.at_level(logging.WARNING, logger="ormah.store.file_store"):
        result = file_store.load(COLLIDING_SHORT_ID)

    assert result is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(COLLIDING_SHORT_ID in r.getMessage() for r in warnings)


def test_full_id_resolves_to_its_own_node_despite_a_short_id_collision(file_store):
    """A Full id is unambiguous even when its Short id is not: load() must
    return the exact node asked for, and delete()/soft_delete() must act on
    that node's file only, leaving the collider untouched."""
    target = _collider("a", "Target node", "the node actually being asked for")
    collider = _collider("b", "Collider node", "an unrelated node sharing the short id")
    file_store.save(target)
    file_store.save(collider)

    loaded = file_store.load(target.id)
    assert loaded is not None
    assert loaded.id == target.id
    assert loaded.content == target.content

    assert file_store.delete(target.id) is True
    assert file_store.load(target.id) is None
    survivor = file_store.load(collider.id)
    assert survivor is not None
    assert survivor.id == collider.id

    # soft_delete must show the same discrimination, on a fresh colliding pair.
    target2 = _collider("c", "Target node 2", "second node actually being asked for")
    collider2 = _collider("d", "Collider node 2", "second unrelated collider")
    file_store.save(target2)
    file_store.save(collider2)

    assert file_store.soft_delete(target2.id) is True
    survivor2 = file_store.load(collider2.id)
    assert survivor2 is not None
    assert survivor2.id == collider2.id


def test_cache_hit_stays_validated_by_existence_only_watcher_hole_is_accepted(file_store):
    """ACCEPTED, documented behaviour — not a bug, not this ticket's territory.

    Once a Full id is cached, a subsequent lookup trusts the cached path the
    moment `Path.exists()` is true; it does not re-open and re-parse the
    file to confirm the id still matches. If something outside FileStore
    (the watcher) replaces the file behind its back, the cache keeps serving
    the old path and load() returns whatever is now written there. Guarding
    against that belongs to the watcher, not to `_find_file` — this test
    pins the hole so nobody "fixes" it here by accident.
    """
    original = MemoryNode(
        type=NodeType.fact,
        tier=Tier.working,
        title="Original node",
        content="before the watcher clobbers it",
        source="test",
    )
    path = file_store.save(original)
    warmed = file_store.load(original.id)  # populates _id_cache[original.id] = path
    assert warmed is not None and warmed.id == original.id

    replacement = MemoryNode(
        type=NodeType.fact,
        tier=Tier.working,
        title="Replacement node",
        content="written straight to disk, bypassing the store",
        source="test",
    )
    path.write_text(serialize_node(replacement), encoding="utf-8")

    served = file_store.load(original.id)

    assert served is not None
    assert served.id == replacement.id
