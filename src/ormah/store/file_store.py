"""File-based CRUD for memory nodes stored as markdown files."""

from __future__ import annotations

import hashlib
import itertools
from functools import wraps
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from slugify import slugify

from ormah.models.node import MemoryNode
from ormah.store.markdown import parse_node, serialize_node

logger = logging.getLogger(__name__)

# ponytail: bounded retry, not a lock. Each attempt costs one lost race against another
# FileStore over the same directory; 50 reports rather than spins if that ever repeats.
_PUBLISH_ATTEMPTS = 50


def _numbered_names(prefix: str, short_id: str):
    """``<prefix>-2_<short_id>.md``, ``-3``, ... without end."""
    for attempt in itertools.count(2):
        yield f"{prefix}-{attempt}_{short_id}.md"


class UnresolvedNodeReference(LookupError):
    """The store cannot tell which node a reference names, nor that none does.

    Raised for an ambiguous Short id, and for a reference whose only candidate files
    will not parse. Neither is absence, so a caller about to mutate state on "not
    found" must not treat it as one.
    """


def _serialized_store_operation(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._operation_lock:
            return method(self, *args, **kwargs)

    return locked


class FileStore:
    """Manages memory node files on disk.

    Maintains an in-memory ``Full id → Path`` cache so that lookups are O(1)
    after the first scan, instead of falling back to an O(N) grep over
    every markdown file. Only a Full id is ever a key, and only once the file
    at that path has confirmed it: a Short id lookup re-resolves on each call,
    and a miss caches nothing.
    """

    def __init__(self, nodes_dir: Path, operation_lock=None) -> None:
        self.nodes_dir = nodes_dir
        self._operation_lock = operation_lock or threading.RLock()
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        # Full id -> Path cache, populated lazily on first miss
        self._id_cache: dict[str, Path] = {}
        self._cache_built = False

    @_serialized_store_operation
    def save(self, node: MemoryNode) -> Path:
        """Write a node to disk atomically. Returns the file path.

        Writes to a temporary file in the same directory, then publishes it under a
        name no other node holds (`_publish`). This prevents partial/corrupt files if
        the process crashes mid-write, and a live node being replaced by a different
        one that raced for the same filename.
        """
        text = serialize_node(node)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.nodes_dir), suffix=".tmp", prefix=".ormah_"
        )
        closed = False
        try:
            os.write(fd, text.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            closed = True
            path = self._publish(node, tmp)
        except BaseException:
            if not closed:
                os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # Update cache
        self._id_cache[node.id] = path
        return path

    @_serialized_store_operation
    def load(self, node_id: str) -> MemoryNode | None:
        """Load a node by ID. Returns None if not found, or if the reference does not
        resolve to one node (see `resolve`)."""
        path = self._find_file(node_id)
        if path is None:
            return None
        return self._load_path(path)

    @_serialized_store_operation
    def resolve(self, node_id: str) -> MemoryNode | None:
        """Like `load`, but only a confirmed absence returns None.

        Raises `UnresolvedNodeReference` where `load` would fold "could not tell" into
        None. A caller that removes index rows on None needs the difference; the
        background jobs keep `load`, whose nullable contract they branch on.
        """
        path = self._locate(node_id)
        if path is None:
            return None
        return self._load_path(path)

    @_serialized_store_operation
    def delete(self, node_id: str) -> bool:
        """Delete a node file. Returns True if deleted."""
        path = self._find_file(node_id)
        if path is None:
            return False
        self._forget(path)  # by path: eviction must not depend on reading the file
        path.unlink()
        return True

    @_serialized_store_operation
    def soft_delete(self, node_id: str) -> bool:
        """Move a node file to the deleted/ directory, stamping `deleted_at`
        into its frontmatter first so tombstones carry their deletion time
        (sync merge ordering depends on it). Returns True if moved.
        """
        path = self._find_file(node_id)
        if path is None:
            return False
        node: MemoryNode | None = None
        try:
            node = self._load_path(path)
            node.deleted_at = datetime.now(timezone.utc)
            # save() writes atomically (tmp + os.replace) onto the existing
            # path, so an interruption can never truncate the live node file.
            self.save(node)
        except Exception:
            node = None  # nothing proves which node this file is
            logger.warning(
                "soft_delete: could not stamp deleted_at on %s; moving as-is", path
            )
        self._forget(path)  # by path: eviction must not depend on reading the file
        self._bury(path, node)
        return True

    @_serialized_store_operation
    def list_all(self) -> list[MemoryNode]:
        """Load all nodes from disk."""
        nodes = []
        for path in sorted(self.nodes_dir.glob("*.md")):
            try:
                nodes.append(self._load_path(path))
            except Exception:
                continue  # skip malformed files
        return nodes

    @_serialized_store_operation
    def list_paths(self) -> list[Path]:
        """List all markdown file paths."""
        return sorted(self.nodes_dir.glob("*.md"))

    @_serialized_store_operation
    def file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file's contents."""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    @_serialized_store_operation
    def touch_access(self, node_id: str) -> MemoryNode | None:
        """Update last_accessed and access_count. Returns updated node."""
        node = self.load(node_id)
        if node is None:
            return None
        node.last_accessed = datetime.now(timezone.utc)
        node.access_count += 1
        self.save(node)
        return node

    def _publish(self, node: MemoryNode, tmp: str) -> Path:
        """Move the staged file `tmp` onto this node's path, and return that path.

        `_path_for` proves only that a candidate was free *when it looked*. Inside one
        process that is enough: the only store that writes the live directory is the
        server's `MemoryEngine` one, and its lock serializes every save. The other
        stores in the codebase only read (`backup.rebuild_index`) or work on another
        directory (`cloud.restore`). The gap is a second process writing the same
        directory — two servers started against one store — whose lock the first
        cannot see. Between that look and this write it can take the name, and
        `os.replace` would drop a live node while both saves report success.

        `os.link` is an atomic create-if-absent: it fails with EEXIST instead of
        clobbering. On EEXIST the name is gone, so ask `_path_for` again — it sees the
        new file and widens past it.
        `os.replace` stays for the one case it is right: this node's own file, and
        `_holds_node` reads that off the file. Asking `_find_file` instead would accept
        a cache hit, which is validated by existence alone — blind to another store
        having deleted this node and given the freed name to a colliding one.

        Known limit: the update is still check-then-act.
        Nothing binds the name to the inode between `_holds_node` and `os.replace`, so
        another store that deletes this node and publishes a colliding one inside that
        window loses its node. Closing it takes a cross-process lock the store has never
        had; `test_update_survives_a_steal_between_identity_read_and_replace` is the
        strict xfail that records it. The same second process also defeats the
        existence-validated cache hit in `delete`, `soft_delete` and `load`, which
        can then act on a colliding node that took the freed name.
        """
        for _ in range(_PUBLISH_ATTEMPTS):
            path = self._path_for(node)
            try:
                os.link(tmp, str(path))
            except FileExistsError:
                if self._holds_node(path, node.id):
                    os.replace(tmp, str(path))  # our own file: an update, not a clobber
                    return path
                # Someone else holds the name, and a cache entry may still point here.
                # Drop it, or `_path_for` keeps handing back this path until the bound
                # runs out instead of widening past the new file.
                self._id_cache.pop(node.id, None)
                continue
            os.unlink(tmp)
            return path
        raise OSError(
            f"could not reserve a filename for node {node.id} in "
            f"{_PUBLISH_ATTEMPTS} attempts"
        )

    def _holds_node(self, path: Path, node_id: str) -> bool:
        """Whether the file at ``path`` names ``node_id`` as its own Full id.

        Read off the file, never out of `_id_cache`: a cache hit is validated by
        existence alone, so it cannot tell this node's file from a colliding node that
        took the name after a delete — and the caller is about to overwrite whatever is
        there. A file that will not parse names nothing, so it does not own the name.
        An OSError propagates, as it does in `_find_file`: not knowing is not permission
        to overwrite.
        """
        try:
            return self._load_path(path).id == node_id
        except OSError:
            raise
        except Exception:
            return False

    def _bury(self, path: Path, node: MemoryNode | None) -> Path:
        """Move the node file at ``path`` into deleted/, and return where it landed.

        A tombstone keeps the node's filename, and that name is not unique: once a
        node is buried, a colliding node can take the same name in nodes/, and a
        blind rename of its file would replace the first tombstone. So the move
        follows `_publish`: walk the node's candidate names and claim one with
        `os.link`, which fails with EEXIST instead of clobbering. A name held by
        this node's own earlier tombstone (read off the file) is replaced — one
        tombstone per node. A file that did not parse names no node, so it never
        replaces anything and takes a numbered name instead.
        """
        deleted_dir = self.nodes_dir.parent / "deleted"
        deleted_dir.mkdir(parents=True, exist_ok=True)
        if node is not None:
            names = self._names_for(node)
        else:
            prefix, _, short_id = path.stem.rpartition("_")
            names = itertools.chain([path.name], _numbered_names(prefix, short_id))
        for name in names:
            dest = deleted_dir / name
            try:
                os.link(path, dest)
            except FileExistsError:
                if node is not None and self._holds_node(dest, node.id):
                    os.replace(path, dest)  # this node's own tombstone
                    return dest
                continue
            os.unlink(path)
            return dest
        raise AssertionError("unreachable: the numbered names never run out")

    def _names_for(self, node: MemoryNode):
        """The filenames a node may take, in order: `_path_for` walks them in nodes/,
        `_bury` in deleted/.

        The Short id is not unique, so type, slug and Short id can all coincide and
        hand a new node the path of a live one — the save would replace its content
        while the filename kept advertising the old title. So after the plain name,
        the slug widens with the next groups of the Full id. The Short id stays the
        last element of the name: the lookup globs on that suffix, and a name that
        dropped it would hide one of two colliding nodes from the ambiguity check
        instead of reporting the clash. Once the Full id is exhausted, the names are
        numbered.
        """
        slug = slugify(node.title or node.content[:60], max_length=40)
        prefix = f"{node.type.value}_{slug}"
        parts = node.id.split("-")
        for width in range(len(parts)):
            extra = "-".join(parts[1 : width + 1])
            widened = f"{prefix}-{extra}" if extra else prefix
            yield f"{widened}_{node.short_id}.md"
        yield from _numbered_names(prefix, node.short_id)

    def _path_for(self, node: MemoryNode) -> Path:
        """Compute the file path for a node, reusing existing file if present.

        The path is a *candidate*, not a reservation: only `_publish` decides. See its
        docstring for why the existence check here cannot be the last word.
        """
        existing = self._find_file(node.id)
        if existing:
            return existing
        # `_find_file` above already returned the node's own file, so any name taken
        # here holds a different node — or a file the lookup did not confirm
        # (unparseable, or renamed behind the store's back). Skip it, never overwrite.
        for name in self._names_for(node):
            path = self.nodes_dir / name
            if not path.exists():
                return path
        raise AssertionError("unreachable: the numbered names never run out")

    def _forget(self, path: Path) -> None:
        """Drop every cache entry naming ``path``.

        The cache is keyed by Full id, but a caller may have addressed the node by
        its Short id, so the key cannot be read off the reference. Reading it off
        the file instead would make eviction depend on that read succeeding: a
        transient OSError leaves the entry pointing at a path the caller is about
        to unlink, and the next node with the same type, title and Short id lands on
        exactly that path — `_path_for` hands out the canonical name whenever it is
        free, and the unlink just freed it — inheriting the entry through the
        existence-only cache hit. Comparing paths needs no file at all.
        """
        for key in [k for k, v in self._id_cache.items() if v == path]:
            del self._id_cache[key]

    def _find_file(self, node_id: str) -> Path | None:
        """`_locate`, with an unresolved reference logged and folded into None."""
        try:
            return self._locate(node_id)
        except UnresolvedNodeReference as exc:
            logger.warning("%s Resolving to nothing.", exc)
            return None

    def _locate(self, node_id: str) -> Path | None:
        """Find the file for a Node reference — a Full id, or the 8-character Short id
        the Whisper showed the agent.

        Lookup order:
        1. In-memory cache, keyed by Full id (O(1)), validated by file existence
        2. Glob on the Short id suffix, accepting a candidate only once the Full id
           in its own frontmatter confirms the reference
        3. Full cache rebuild from disk (one-time O(N), then O(1) forever)

        The glob narrows; it never decides. A filename carries the Short id, which is
        not unique, so the first match may be a stranger's memory — the file has to
        state its Full id before the store hands it back. An ambiguous
        Short id raises `UnresolvedNodeReference`; `_find_file` turns that into None
        with a warning, because the load contract is nullable and the background jobs
        branch only on "is it None" — raising there would trade silent corruption for
        a crashed sleep cycle.

        None therefore means *confirmed absent*, never *could not tell*. A file that
        will not parse confirms nothing and is skipped, matching `list_all` and
        `_build_cache` — but when it was the only candidate, or the reference is a
        bare Short id, the reference is unresolved, not absent: that file may be this
        very node, or a second node sharing the Short id. An OSError
        propagates, because a caller that reads it as absence goes on to mutate state
        the file still contradicts.

        A cache hit stays validated by file existence alone. Re-parsing on every hit
        would destroy the O(1) the cache exists for. A file replaced behind the
        store's back is the watcher's territory, and is accepted here.
        """
        # 1. Cache hit
        cached = self._id_cache.get(node_id)
        if cached is not None:
            if cached.exists():
                return cached
            # Stale entry — remove and fall through
            del self._id_cache[node_id]

        # 2. Glob on the Short id, then confirm each candidate against its own Full id.
        #    Width stays at 8: filenames end in the 8-character Short id, so any other
        #    width would force a full directory scan to serve a caller that does not exist.
        short_id = node_id.split("-")[0]
        unparseable = False
        if len(short_id) == 8:
            confirmed: list[tuple[str, Path]] = []
            for path in sorted(self.nodes_dir.glob(f"*_{short_id}.md")):
                try:
                    candidate = self._load_path(path)
                except OSError:
                    # Not knowing is not absence. Swallowing this would resolve an
                    # existing node to None, and `MemoryEngine.delete_node` reads
                    # None as absence: it drops the index row and reports success
                    # while the file survives to be reindexed. Before this lookup
                    # opened candidates at all, the error surfaced from `load`.
                    raise
                except Exception:
                    unparseable = True
                    continue  # a file that will not parse confirms nothing
                # A bare Short id has no dashes, so the split above is a no-op and
                # it equals itself — that is what tells the two lookups apart.
                if candidate.id == node_id or (
                    node_id == short_id and candidate.short_id == short_id
                ):
                    confirmed.append((candidate.id, path))
            if unparseable and node_id == short_id:
                # A bare Short id is unique only if every file sharing it is known:
                # the one that will not parse may be a second node with this Short id.
                raise UnresolvedNodeReference(
                    f"Node reference {node_id} is a Short id shared with a file "
                    "that will not parse."
                )
            if len(confirmed) == 1:
                full_id, found = confirmed[0]
                # Keyed by Full id, and written only after the file confirmed it.
                self._id_cache[full_id] = found
                return found
            if len(confirmed) > 1:
                raise UnresolvedNodeReference(
                    f"Node reference {node_id} is an ambiguous Short id: "
                    f"{len(confirmed)} nodes share it."
                )

        # 3. Build full cache once if not already done
        if not self._cache_built:
            self._build_cache()
            cached = self._id_cache.get(node_id)
            if cached is not None and cached.exists():
                return cached

        if unparseable:
            raise UnresolvedNodeReference(
                f"Node reference {node_id} matches only files that will not parse."
            )
        return None

    def _build_cache(self) -> None:
        """Scan all markdown files and populate the id→path cache."""
        count = 0
        for path in self.nodes_dir.glob("*.md"):
            try:
                node = self._load_path(path)
                self._id_cache[node.id] = path
                count += 1
            except Exception:
                continue
        self._cache_built = True
        if count:
            logger.debug("FileStore cache built: %d nodes", count)

    def _load_path(self, path: Path) -> MemoryNode:
        text = path.read_text(encoding="utf-8")
        return parse_node(text)
