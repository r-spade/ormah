# Task 04: file_store tombstone — `deleted_at`, `list_deleted`, `purge`

`soft_delete` already moves a node file to `deleted/`. Make it stamp `deleted_at` into the
moved file's frontmatter (self-contained purge clock), and add `list_deleted()` + `purge()`
so Phase B can find and hard-remove expired tombstones.

**Files:**
- Modify: `src/ormah/store/file_store.py`
- Test: `tests/test_store/test_soft_delete_tombstone.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_store/test_soft_delete_tombstone.py`:

```python
from __future__ import annotations

import frontmatter

from ormah.models.node import MemoryNode, NodeType
from ormah.store.file_store import FileStore


def _store(tmp_path):
    return FileStore(tmp_path / "nodes")


def _make(store, content="x"):
    node = MemoryNode(type=NodeType.fact, content=content, title=content)
    store.save(node)
    return node


def test_soft_delete_stamps_deleted_at(tmp_path):
    store = _store(tmp_path)
    node = _make(store)

    assert store.soft_delete(node.id) is True

    deleted_dir = tmp_path / "deleted"
    files = list(deleted_dir.glob("*.md"))
    assert len(files) == 1
    meta = frontmatter.loads(files[0].read_text(encoding="utf-8")).metadata
    assert "deleted_at" in meta and meta["deleted_at"]


def test_list_deleted_returns_id_and_deleted_at(tmp_path):
    store = _store(tmp_path)
    node = _make(store)
    store.soft_delete(node.id)

    entries = store.list_deleted()
    assert len(entries) == 1
    node_id, deleted_at, path = entries[0]
    assert node_id == node.id
    assert deleted_at is not None
    assert path.exists()


def test_purge_removes_tombstone(tmp_path):
    store = _store(tmp_path)
    node = _make(store)
    store.soft_delete(node.id)

    assert store.purge(node.id) is True
    assert store.list_deleted() == []
    assert store.purge(node.id) is False  # idempotent: already gone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_store/test_soft_delete_tombstone.py -v`
Expected: FAIL (`deleted_at` not stamped; `list_deleted`/`purge` missing).

- [ ] **Step 3: Implement in `file_store.py`**

Add `import frontmatter` at the top (with the other imports). `os` and `tempfile` are already
imported in `file_store.py`.

Replace the existing `soft_delete` method body with an **atomic** tombstone write that mirrors
`save()` (council R3 H6: the old `path.rename` was atomic; an in-place `write_text`+`unlink` is
not — a crash mid-write could truncate the tombstone):

```python
    def soft_delete(self, node_id: str) -> bool:
        """Move a node file to the deleted/ directory, stamping deleted_at atomically.

        The deleted_at tombstone in the moved file's frontmatter is the purge
        clock (#28): self-contained, so it survives backup/restore and mtime resets.
        Written via tmp + fsync + os.replace so a crash never leaves a partial tombstone.
        """
        path = self._find_file(node_id)
        if path is None:
            return False
        deleted_dir = self.nodes_dir.parent / "deleted"
        deleted_dir.mkdir(parents=True, exist_ok=True)

        post = frontmatter.loads(path.read_text(encoding="utf-8"))
        post["deleted_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        text = frontmatter.dumps(post)
        dest = deleted_dir / path.name

        fd, tmp = tempfile.mkstemp(dir=str(deleted_dir), suffix=".tmp", prefix=".ormah_")
        closed = False
        try:
            data = text.encode("utf-8")
            written = 0
            while written < len(data):     # write-all: os.write may short-write (council R4 H9)
                written += os.write(fd, data[written:])
            os.fsync(fd)
            os.close(fd)
            closed = True
            os.replace(tmp, str(dest))   # atomic publish of the tombstone
        except BaseException:
            if not closed:
                os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        path.unlink()                    # remove the original only after dest is durable

        self._id_cache.pop(node_id, None)
        return True
```

Add two new methods after `soft_delete`:

```python
    def list_deleted(self) -> list[tuple[str, str | None, Path]]:
        """List tombstones in deleted/ as (node_id, deleted_at, path)."""
        deleted_dir = self.nodes_dir.parent / "deleted"
        if not deleted_dir.exists():
            return []
        out: list[tuple[str, str | None, Path]] = []
        for path in sorted(deleted_dir.glob("*.md")):
            try:
                meta = frontmatter.loads(path.read_text(encoding="utf-8")).metadata
            except Exception:
                continue  # skip unreadable tombstone
            node_id = meta.get("id")
            if node_id:
                out.append((node_id, meta.get("deleted_at"), path))
        return out

    def purge(self, node_id: str) -> bool:
        """Hard-delete a tombstone from deleted/. Returns True if removed."""
        deleted_dir = self.nodes_dir.parent / "deleted"
        if not deleted_dir.exists():
            return False
        for node_id_found, _deleted_at, path in self.list_deleted():
            if node_id_found == node_id:
                path.unlink()
                return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_store/test_soft_delete_tombstone.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Regression — manual delete path still works**

Run: `.venv/bin/python -m pytest tests/test_engine -k delete -v`
Expected: PASS (engine.delete_node calls soft_delete; the stamp is additive).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/store/file_store.py tests/test_store/test_soft_delete_tombstone.py
git add src/ormah/store/file_store.py tests/test_store/test_soft_delete_tombstone.py
git commit -m "feat(store): stamp deleted_at on soft_delete; add list_deleted/purge (#28)"
```
