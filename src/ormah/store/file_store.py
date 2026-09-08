"""File-based CRUD for memory nodes stored as markdown files."""

from __future__ import annotations

import hashlib
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

        Writes to a temporary file in the same directory, then uses
        ``os.replace()`` to atomically swap it into place. This prevents
        partial/corrupt files if the process crashes mid-write.
        """
        path = self._path_for(node)
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
            os.replace(tmp, str(path))
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
        """Load a node by ID. Returns None if not found."""
        path = self._find_file(node_id)
        if path is None:
            return None
        return self._load_path(path)

    @_serialized_store_operation
    def delete(self, node_id: str) -> bool:
        """Delete a node file. Returns True if deleted."""
        path = self._find_file(node_id)
        if path is None:
            return False
        self._forget(path)  # while the file is still there to name its Full id
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
        try:
            node = self._load_path(path)
            node.deleted_at = datetime.now(timezone.utc)
            # save() writes atomically (tmp + os.replace) onto the existing
            # path, so an interruption can never truncate the live node file.
            self.save(node)
        except Exception:
            logger.warning(
                "soft_delete: could not stamp deleted_at on %s; moving as-is", path
            )
        deleted_dir = self.nodes_dir.parent / "deleted"
        deleted_dir.mkdir(parents=True, exist_ok=True)
        dest = deleted_dir / path.name
        self._forget(path)  # while the file is still there to name its Full id
        path.rename(dest)
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

    def _path_for(self, node: MemoryNode) -> Path:
        """Compute the file path for a node, reusing existing file if present."""
        existing = self._find_file(node.id)
        if existing:
            return existing
        slug = slugify(node.title or node.content[:60], max_length=40)
        filename = f"{node.type.value}_{slug}_{node.short_id}.md"
        return self.nodes_dir / filename

    def _forget(self, path: Path) -> None:
        """Drop the cache entry for the node stored at ``path``.

        The cache is keyed by Full id, but a caller may have addressed the node by
        its Short id, so the key cannot be read off the reference — only off the
        file. Call this while the file is still on disk. A file that will not parse
        leaves nothing behind that the next lookup's existence check will not clear.
        """
        try:
            self._id_cache.pop(self._load_path(path).id, None)
        except Exception:
            pass

    def _find_file(self, node_id: str) -> Path | None:
        """Find the file for a Node reference — a Full id, or the 8-character Short id
        the Whisper showed the agent.

        Lookup order:
        1. In-memory cache, keyed by Full id (O(1)), validated by file existence
        2. Glob on the Short id suffix, accepting a candidate only once the Full id
           in its own frontmatter confirms the reference
        3. Full cache rebuild from disk (one-time O(N), then O(1) forever)

        The glob narrows; it never decides. A filename carries the Short id, which is
        not unique, so the first match may be a stranger's memory — the file has to
        state its Full id before the store hands it back (ADR-0007). An ambiguous
        Short id resolves to nothing, with a warning: the load contract is nullable
        and the background jobs branch only on "is it None", so raising would trade
        silent corruption for a crashed sleep cycle.

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
        if len(short_id) == 8:
            confirmed: list[tuple[str, Path]] = []
            for path in sorted(self.nodes_dir.glob(f"*_{short_id}.md")):
                try:
                    candidate = self._load_path(path)
                except Exception:
                    continue  # a file that will not parse confirms nothing
                # A bare Short id has no dashes, so the split above is a no-op and
                # it equals itself — that is what tells the two lookups apart.
                if candidate.id == node_id or (
                    node_id == short_id and candidate.short_id == short_id
                ):
                    confirmed.append((candidate.id, path))
            if len(confirmed) == 1:
                full_id, found = confirmed[0]
                # Keyed by Full id, and written only after the file confirmed it.
                self._id_cache[full_id] = found
                return found
            if len(confirmed) > 1:
                logger.warning(
                    "Node reference %s is an ambiguous Short id: %d nodes share it. "
                    "Resolving to nothing rather than to an arbitrary one of them.",
                    node_id,
                    len(confirmed),
                )
                return None

        # 3. Build full cache once if not already done
        if not self._cache_built:
            self._build_cache()
            cached = self._id_cache.get(node_id)
            if cached is not None and cached.exists():
                return cached

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
