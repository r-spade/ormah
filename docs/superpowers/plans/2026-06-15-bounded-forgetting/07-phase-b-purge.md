# Task 07: Phase B — hard-purge expired tombstones

**Depends on:** Tasks 04, 05.

After the soft-delete phases, scan `deleted/` and hard-remove any tombstone whose `deleted_at`
is older than `deletion_retention_days`, logging each purge to `audit_log` (`operation='purge'`).
Tombstones with no/unparseable `deleted_at` are kept (fail-safe).

**Files:**
- Modify: `src/ormah/background/forgetting_manager.py`
- Test: `tests/test_background/test_forgetting_manager.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_background/test_forgetting_manager.py`:

```python
import frontmatter


def _backdate_tombstone(engine, node_id, days):
    when = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")
    for nid, _da, path in engine.file_store.list_deleted():
        if nid == node_id:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
            post["deleted_at"] = when
            path.write_text(frontmatter.dumps(post), encoding="utf-8")
            return
    raise AssertionError(f"tombstone for {node_id} not found")


def _tombstone_ids(engine):
    return {nid for nid, _da, _p in engine.file_store.list_deleted()}


def test_expired_tombstone_is_purged_and_audited(engine):
    _enable(engine)
    node_id, _ = engine.remember(CreateNodeRequest(
        content="bye", type=NodeType.fact, tier=Tier.working, title="bye"))
    engine.delete_node(node_id)              # soft-delete → deleted/
    _backdate_tombstone(engine, node_id, days=60)  # retention is 30

    run_forgetting(engine)

    assert node_id not in _tombstone_ids(engine)
    audited = engine.db.conn.execute(
        "SELECT 1 FROM audit_log WHERE operation='purge' AND node_id=?", (node_id,)
    ).fetchone()
    assert audited is not None


def test_tombstone_within_window_is_kept(engine):
    _enable(engine)
    node_id, _ = engine.remember(CreateNodeRequest(
        content="recent", type=NodeType.fact, tier=Tier.working, title="recent"))
    engine.delete_node(node_id)
    _backdate_tombstone(engine, node_id, days=5)  # inside the 30-day window

    run_forgetting(engine)

    assert node_id in _tombstone_ids(engine)


def test_purge_skipped_when_disabled(engine):
    # master switch OFF: even an old tombstone is not purged
    node_id, _ = engine.remember(CreateNodeRequest(
        content="keep", type=NodeType.fact, tier=Tier.working, title="keep"))
    engine.delete_node(node_id)
    _backdate_tombstone(engine, node_id, days=60)

    run_forgetting(engine)  # deletion_enabled defaults to False

    assert node_id in _tombstone_ids(engine)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_background/test_forgetting_manager.py -k purge -v`
Expected: FAIL (`test_purge_skipped_when_disabled` passes already; the other two fail — expired tombstone survives).

- [ ] **Step 3: Wire Phase B into `run_forgetting`**

In `src/ormah/background/forgetting_manager.py`, replace the line:

```python
        # Task 07 inserts Phase B (hard-purge) here.
```

with:

```python
        _run_purge(engine, now)
```

- [ ] **Step 4: Implement the purge**

Append to `forgetting_manager.py`:

```python
def _run_purge(engine, now: datetime) -> int:
    """Hard-purge tombstones whose deleted_at is past the retention window."""
    s = engine.settings
    cutoff = now - timedelta(days=s.deletion_retention_days)
    purged = 0
    for node_id, deleted_at, _path in engine.file_store.list_deleted():
        if not deleted_at:
            continue  # no clock → keep (fail-safe)
        try:
            ts = datetime.fromisoformat(deleted_at)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts > cutoff:
            continue  # still inside the reversibility window
        if engine.file_store.purge(node_id):
            engine._write_audit_log(operation="purge", node_id=node_id)
            purged += 1
    if purged:
        logger.info("Forgetting hard-purged %d expired tombstones", purged)
    return purged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_background/test_forgetting_manager.py -v`
Expected: PASS (full suite for this module).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/background/forgetting_manager.py tests/test_background/test_forgetting_manager.py
git add src/ormah/background/forgetting_manager.py tests/test_background/test_forgetting_manager.py
git commit -m "feat(background): phase B hard-purge of expired tombstones (#28)"
```
