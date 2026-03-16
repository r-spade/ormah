"""File-based CRUD for memory nodes stored as markdown files."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from slugify import slugify

from ormah.models.node import MemoryNode
from ormah.store.markdown import parse_node, serialize_node

logger = logging.getLogger(__name__)


class FileStore:
    """Manages memory node files on disk.

    Maintains an in-memory ``id → Path`` cache so that lookups are O(1)
    after the first scan, instead of falling back to an O(N) grep over
    every markdown file.
    """

    def __init__(self, nodes_dir: Path) -> None:
        self.nodes_dir = nodes_dir
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        # id -> Path cache, populated lazily on first miss
        self._id_cache: dict[str, Path] = {}
        self._cache_built = False

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

    def load(self, node_id: str) -> MemoryNode | None:
        """Load a node by ID. Returns None if not found."""
        path = self._find_file(node_id)
        if path is None:
            return None
        return self._load_path(path)

    def delete(self, node_id: str) -> bool:
        """Delete a node file. Returns True if deleted."""
        path = self._find_file(node_id)
        if path is None:
            return False
        path.unlink()
        self._id_cache.pop(node_id, None)
        return True

    def soft_delete(self, node_id: str) -> bool:
        """Move a node file to the deleted/ directory. Returns True if moved."""
        path = self._find_file(node_id)
        if path is None:
            return False
        deleted_dir = self.nodes_dir.parent / "deleted"
        deleted_dir.mkdir(parents=True, exist_ok=True)
        dest = deleted_dir / path.name
        path.rename(dest)
        self._id_cache.pop(node_id, None)
        return True

    def list_all(self) -> list[MemoryNode]:
        """Load all nodes from disk."""
        nodes = []
        for path in sorted(self.nodes_dir.glob("*.md")):
            try:
                nodes.append(self._load_path(path))
            except Exception:
                continue  # skip malformed files
        return nodes

    def list_paths(self) -> list[Path]:
        """List all markdown file paths."""
        return sorted(self.nodes_dir.glob("*.md"))

    def file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file's contents."""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

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

    def _find_file(self, node_id: str) -> Path | None:
        """Find the file for a given node ID.

        Lookup order:
        1. In-memory cache (O(1))
        2. Glob on short_id suffix (fast for single file)
        3. Full cache rebuild from disk (one-time O(N), then O(1) forever)
        """
        # 1. Cache hit
        cached = self._id_cache.get(node_id)
        if cached is not None:
            if cached.exists():
                return cached
            # Stale entry — remove and fall through
            del self._id_cache[node_id]

        # 2. Glob on short_id
        short_id = node_id.split("-")[0]
        matches = list(self.nodes_dir.glob(f"*_{short_id}.md"))
        if matches:
            self._id_cache[node_id] = matches[0]
            return matches[0]

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
