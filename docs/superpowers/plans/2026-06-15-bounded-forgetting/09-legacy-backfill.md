# Task 09: Lazy, atomic legacy `archived_at` file backfill

**Depends on:** Tasks 02, 03, 05.

Legacy archival nodes (demoted before `archived_at` existed) have no `archived_at` in their
**files**. Task 02's migration only backfilled the derived index — wiped on the next
`full_rebuild`. This task durably stamps the **source files**, once, the safe way (council R2 C4):

- **Lazy + one-time**, guarded by a `meta` flag — runs at the top of `run_forgetting` only when
  the feature is enabled, never inside the SQLite migration transaction (avoids the #18
  startup-contention class Cursor flagged).
- **Atomic** — reuses `file_store.save` (tmp + fsync + `os.replace`), never an in-place
  `Path.write_text`, so a crash/disk-full can never truncate a source memory.
- **Fail-open, retry-safe** — a per-file failure is logged and counted; the `done` flag is set
  only when zero files failed, so a transient error retries next run instead of being silently
  marked complete.

The one-time guard also defines the semantics for `remember(tier=archival)` (council L2): every
archival file present at the *first* enabled run is treated as legacy and stamped; archival
nodes created *after* that stay `archived_at=NULL` → protected. The first run is O(archival) file
loads — acceptable for a one-time, opt-in pass; documented as such.

**Files:**
- Modify: `src/ormah/background/forgetting_manager.py`
- Test: `tests/test_background/test_legacy_backfill.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_background/test_legacy_backfill.py`:

```python
from __future__ import annotations

from ormah.background.forgetting_manager import _BACKFILL_META_KEY, run_forgetting
from ormah.models.node import CreateNodeRequest, NodeType, Tier


def _legacy_archival(engine, content="legacy"):
    """A node whose FILE lacks archived_at (remember(tier=archival) never stamps it)."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content=content, type=NodeType.fact, tier=Tier.archival, title=content))
    assert engine.file_store.load(node_id).archived_at is None
    return node_id


def _meta_done(engine):
    row = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key=?", (_BACKFILL_META_KEY,)).fetchone()
    return row is not None


def test_backfill_stamps_legacy_files_and_survives_rebuild(engine):
    engine.settings.deletion_enabled = True
    node_id = _legacy_archival(engine)

    run_forgetting(engine)

    assert engine.file_store.load(node_id).archived_at is not None  # file stamped (durable)
    engine.builder.full_rebuild()
    row = engine.db.conn.execute(
        "SELECT archived_at FROM nodes WHERE id=?", (node_id,)).fetchone()
    assert row["archived_at"] is not None  # survives rebuild
    assert _meta_done(engine) is True


def test_backfill_skipped_when_disabled(engine):
    node_id = _legacy_archival(engine)
    run_forgetting(engine)  # deletion_enabled defaults to False
    assert engine.file_store.load(node_id).archived_at is None
    assert _meta_done(engine) is False


def test_backfill_write_failure_preserves_file_and_retries(engine, monkeypatch):
    engine.settings.deletion_enabled = True
    node_id = _legacy_archival(engine)
    original = engine.file_store.load(node_id)

    def boom(node):
        raise OSError("disk full")
    monkeypatch.setattr(engine.file_store, "save", boom)

    run_forgetting(engine)  # must not raise; file untouched; not marked done

    reloaded = engine.file_store.load(node_id)
    assert reloaded is not None and reloaded.archived_at is None  # original intact
    assert _meta_done(engine) is False  # transient failure → retry next run
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_background/test_legacy_backfill.py -v`
Expected: FAIL (`_BACKFILL_META_KEY` / backfill not implemented).

- [ ] **Step 3: Wire the backfill into `run_forgetting`**

In `src/ormah/background/forgetting_manager.py`, inside `run_forgetting`, add the call right
after `now = ...` and before `_run_gate_phase`:

```python
        _backfill_legacy_archived_at(engine)
```

- [ ] **Step 4: Implement the backfill**

Append to `forgetting_manager.py`:

```python
_BACKFILL_META_KEY = "archived_at_legacy_backfill_done"


def _backfill_legacy_archived_at(engine) -> int:
    """One-time: stamp archived_at into legacy archival files lacking it (atomic via file_store).

    Runs once (guarded by a meta flag), outside any migration transaction. Uses the atomic
    file_store.save so a write failure can never truncate a source memory; the done-flag is set
    only on a fully clean pass, so transient failures retry next run.
    """
    done = engine.db.conn.execute(
        "SELECT 1 FROM meta WHERE key = ?", (_BACKFILL_META_KEY,)
    ).fetchone()
    if done:
        return 0

    stamped, skipped = 0, 0
    # Enumerate the SOURCE FILES on disk, not the derived index (council R3 H8): an archival
    # file absent from the index would otherwise be missed forever once `done` is set.
    for node in engine.file_store.list_all():
        if node.tier != Tier.archival:
            continue
        irow = engine.db.conn.execute(
            "SELECT archived_at FROM nodes WHERE id = ?", (node.id,)
        ).fetchone()
        index_has = irow is not None and irow["archived_at"] is not None
        if node.archived_at is not None and index_has:
            continue  # file + index already consistent
        try:
            if node.archived_at is None:
                node.archived_at = node.updated  # best proxy for the legacy demotion time
                path = engine.file_store.save(node)   # atomic tmp + fsync + os.replace
            else:
                path = engine.file_store._find_file(node.id)  # file ok, index lagged
            engine.builder.index_single(path)         # stamp/repair the index
            stamped += 1
        except Exception:
            skipped += 1
            logger.warning("archived_at backfill skipped node %s", node.id)

    # Only mark done on a fully clean pass — a save-ok/index-fail node is re-checked next run
    # (its file has archived_at but the index does not, so the loop re-indexes it).
    if skipped == 0:
        with engine.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')",
                (_BACKFILL_META_KEY,),
            )
    if stamped or skipped:
        logger.info(
            "Forgetting legacy backfill: stamped %d, skipped %d", stamped, skipped
        )
    return stamped
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_background/test_legacy_backfill.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Regression — the rest of the forgetting suite still passes**

Run: `.venv/bin/python -m pytest tests/test_background/test_forgetting_manager.py -v`
Expected: PASS. (Note: `_make_eligible`/`_make_archival_recent` set `archived_at` explicitly, so
they remain eligible/scored after the backfill runs; the backfill only touches files where
`archived_at` is None.)

- [ ] **Step 7: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/background/forgetting_manager.py tests/test_background/test_legacy_backfill.py
git add src/ormah/background/forgetting_manager.py tests/test_background/test_legacy_backfill.py
git commit -m "feat(background): lazy atomic legacy archived_at file backfill (#28)"
```
