# Task 05: Forgetting manager — Phase A gates → soft-delete

**Depends on:** Tasks 01, 02, 03, 04.

Create the new job module. Phase A selects `archival` candidates, applies the §1 gates split
into two predicates, and soft-deletes the eligible ones via `engine.delete_node`. The whole job
is a no-op when `deletion_enabled=False`.

## Council R1 design decisions baked in

- **Single source of protection (`_is_protected`).** The §1 gates split into **protections**
  (hard "never delete": self node, `archived_at IS NULL`, importance ≥ threshold, positive
  feedback, hub/strong-edge, degree > max) and **staleness signals** (`archived_at` old,
  `last_accessed` old, `R < floor`). Phase A requires *not protected AND stale*. Task 06's cap
  reuses **the exact same `_is_protected`** — so the cap can never delete a protected node
  (council C1). `archived_at IS NULL` counts as protected, closing the `remember(tier=archival)`
  hole (council H3).
- **Atomic delete-if-eligible (council C2 + R2 C3).** Per-node revalidation alone leaves a
  TOCTOU gap (re-check and delete are separate ops). The fix uses a new
  `engine.delete_node_guarded(node_id, guard)`: the guard re-checks eligibility **inside the
  same `db.transaction()`** that removes the node. `Database.transaction()` is `BEGIN IMMEDIATE`
  and holds a process lock for its whole duration, so every concurrent write
  (`submit_feedback`, `connect`, `update_node` promotion) is serialized — it either commits
  before the guard reads (guard aborts) or blocks until the node is already gone. No global lock
  around `recall` (that would reintroduce the #18/#19 contention PR#19 fixed).
- **Robust success check (council L1).** `delete_node` returns `str | None`; treat success only
  when the message starts with `"Deleted"`.

**Files:**
- Modify: `src/ormah/engine/memory_engine.py` (add `delete_node_guarded`)
- Create: `src/ormah/background/forgetting_manager.py`
- Test: `tests/test_background/test_forgetting_manager.py`, `tests/test_engine/test_delete_guarded.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_background/test_forgetting_manager.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ormah.background.forgetting_manager import run_forgetting
from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType, Tier


def _exists(engine, node_id) -> bool:
    row = engine.db.conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row is not None


def _enable(engine):
    engine.settings.deletion_enabled = True


def _make_eligible(engine, content="dead weight", days=200):
    """Create an archival node eligible in BOTH file and index (the guard reads the file)."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content=content, type=NodeType.fact, tier=Tier.archival, title=content))
    old = datetime.now(timezone.utc) - timedelta(days=days)
    node = engine.file_store.load(node_id)
    node.importance = 0.1
    node.stability = 1.0
    node.last_review = old
    node.last_accessed = old
    node.archived_at = old
    path = engine.file_store.save(node)        # source of truth
    engine.builder.index_single(path)          # keep the index in sync
    engine.db.conn.commit()
    return node_id


def test_master_switch_off_is_noop(engine):
    node_id = _make_eligible(engine)
    run_forgetting(engine)  # deletion_enabled defaults to False
    assert _exists(engine, node_id) is True


def test_fully_eligible_node_is_soft_deleted(engine):
    _enable(engine)
    node_id = _make_eligible(engine)
    run_forgetting(engine)
    assert _exists(engine, node_id) is False


def test_idempotent_second_run_deletes_nothing(engine):
    _enable(engine)
    node_id = _make_eligible(engine)
    run_forgetting(engine)
    run_forgetting(engine)
    assert _exists(engine, node_id) is False


# --- conjunction matrix: passing all gates deletes; breaking exactly one keeps (council M1) ---

def _break(engine, node_id, gate):
    now = datetime.now(timezone.utc)
    recent = now.isoformat()
    if gate == "tier":
        engine.db.conn.execute("UPDATE nodes SET tier='working' WHERE id=?", (node_id,))
    elif gate == "archived_recent":
        engine.db.conn.execute("UPDATE nodes SET archived_at=? WHERE id=?", (recent, node_id))
    elif gate == "accessed_recent":
        engine.db.conn.execute("UPDATE nodes SET last_accessed=? WHERE id=?", (recent, node_id))
    elif gate == "retrievable":   # high stability ⇒ R well above floor
        engine.db.conn.execute("UPDATE nodes SET stability=100000.0 WHERE id=?", (node_id,))
    elif gate == "importance":
        engine.db.conn.execute("UPDATE nodes SET importance=0.9 WHERE id=?", (node_id,))
    elif gate == "archived_null":
        engine.db.conn.execute("UPDATE nodes SET archived_at=NULL WHERE id=?", (node_id,))
    elif gate == "feedback":
        engine.db.conn.execute(
            "INSERT INTO affinity (prompt_vec, node_id, signal, source, confirmed_at, session_id) "
            "VALUES (?, ?, 1, 'explicit', ?, 's1')", (b"\x00", node_id, recent))
    engine.db.conn.commit()


@pytest.mark.parametrize("gate", [
    "tier", "archived_recent", "accessed_recent", "retrievable",
    "importance", "archived_null", "feedback",
])
def test_breaking_one_gate_keeps_node(engine, gate):
    _enable(engine)
    node_id = _make_eligible(engine)
    _break(engine, node_id, gate)
    run_forgetting(engine)
    assert _exists(engine, node_id) is True, f"gate={gate} should have protected the node"


def test_strong_edge_protects_both_nodes(engine):
    _enable(engine)
    a = _make_eligible(engine, content="hub a")
    b = _make_eligible(engine, content="hub b")
    engine.connect(ConnectRequest(source_id=a, target_id=b, edge=EdgeType.related_to, weight=0.9))
    run_forgetting(engine)
    assert _exists(engine, a) is True and _exists(engine, b) is True


def test_user_node_never_deleted(engine):
    _enable(engine)
    uid = engine.user_node_id
    assert uid is not None
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET tier='archival', importance=0.1, stability=1.0, "
        "last_review=?, last_accessed=?, archived_at=? WHERE id=?",
        (old, old, old, uid))
    engine.db.conn.commit()
    run_forgetting(engine)
    assert _exists(engine, uid) is True


def test_guard_reads_file_over_stale_index(engine):
    """Cross-path race (council R3 C5): a promotion writes the FILE before the index.

    The pre-filter (index) still sees archival+stale and selects the node, but the hybrid guard
    reads the source file (tier=working) and aborts. Fails with an index-only guard.
    """
    _enable(engine)
    node_id = _make_eligible(engine)
    node = engine.file_store.load(node_id)
    node.tier = Tier.working
    engine.file_store.save(node)  # file promoted; index intentionally NOT updated
    run_forgetting(engine)
    assert _exists(engine, node_id) is True  # guard saw the fresh file → no deletion
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_background/test_forgetting_manager.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Add `delete_node_guarded` to the engine (atomic delete-if-eligible)**

First write the guard test — create `tests/test_engine/test_delete_guarded.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from ormah.models.node import CreateNodeRequest, NodeType, Tier


def _archival(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="g", type=NodeType.fact, tier=Tier.archival, title="g"))
    return node_id


def _exists(engine, node_id):
    return engine.db.conn.execute(
        "SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone() is not None


def test_guard_false_aborts_deletion(engine):
    node_id = _archival(engine)
    res = engine.delete_node_guarded(node_id, lambda conn: False)
    assert res is None
    assert _exists(engine, node_id) is True


def test_guard_true_deletes(engine):
    node_id = _archival(engine)
    res = engine.delete_node_guarded(node_id, lambda conn: True)
    assert res is not None and res.startswith("Deleted")
    assert _exists(engine, node_id) is False


def test_guard_observes_writes_in_same_transaction(engine):
    """A +feedback row inserted inside the guard's txn is visible to the guard's recheck."""
    node_id = _archival(engine)

    def guard(conn):
        conn.execute(
            "INSERT INTO affinity (prompt_vec, node_id, signal, source, confirmed_at, session_id) "
            "VALUES (?, ?, 1, 'explicit', ?, 's1')",
            (b"\x00", node_id, datetime.now(timezone.utc).isoformat()))
        row = conn.execute(
            "SELECT 1 FROM affinity WHERE node_id=? AND signal>0 LIMIT 1", (node_id,)
        ).fetchone()
        return row is None  # protected (has feedback) → guard returns False → abort

    res = engine.delete_node_guarded(node_id, guard)
    assert res is None
    assert _exists(engine, node_id) is True


def test_guard_never_deletes_user_node(engine):
    res = engine.delete_node_guarded(engine.user_node_id, lambda conn: True)
    assert res is None
```

Run: `.venv/bin/python -m pytest tests/test_engine/test_delete_guarded.py -v` → FAIL (method missing).

Add the method to `src/ormah/engine/memory_engine.py`, right after `delete_node`:

```python
    def delete_node_guarded(self, node_id: str, guard) -> str | None:
        """Soft-delete a node only if ``guard(conn)`` still holds inside the write txn.

        ``Database.transaction`` is BEGIN IMMEDIATE and holds the cross-thread write lock for
        its whole duration, so the guard's recheck and the index removal are atomic against any
        concurrent feedback/connect/promotion — closing the forgetting TOCTOU race (#28) without
        serializing the recall hot path.
        """
        if node_id == self.user_node_id:
            return None
        full_node = self.file_store.load(node_id)
        if full_node is None:
            return None
        title = full_node.title or full_node.content[:60]
        snapshot = json.dumps(full_node.model_dump(mode="json"))
        node_type = full_node.type.value

        with self.db.transaction() as conn:
            if not guard(conn):
                return None  # state changed since selection — abort atomically
            # Move the file FIRST (atomic), then remove from the index — both inside the txn
            # (council R3 H7). If we crash after the move but before COMMIT, the index still
            # references a now-missing file: load() returns None and the next full_rebuild drops
            # the dangling row. The reverse order would resurrect the node on rebuild.
            if not self.file_store.soft_delete(node_id):
                return None  # file already gone — index untouched, nothing to do
            self.builder._remove_node(node_id)
            conn.execute(
                "DELETE FROM auto_link_checked WHERE node_a = ? OR node_b = ?",
                (node_id, node_id),
            )
            conn.execute(
                "INSERT INTO audit_log (operation, node_id, node_snapshot, detail, performed_at) "
                "VALUES ('delete', ?, ?, ?, ?)",
                (node_id, snapshot, json.dumps({"reason": "bounded_forgetting"}),
                 datetime.now(timezone.utc).isoformat()),
            )

        return f"Deleted [{node_type}]: {title}\nID: {node_id}"
```

`json`, `datetime`, `timezone` are already imported in this module. Run the test again → PASS.

- [ ] **Step 4: Implement the module**

Create `src/ormah/background/forgetting_manager.py`:

```python
"""Bounded forgetting (#28): delete dead-weight archival nodes via conjunction gates.

Two phases per run, both behind the master switch ``deletion_enabled`` (default OFF):
  A. apply §1 gates → soft-delete eligible archival nodes (+ §3 cap backstop, task 06);
  B. hard-purge tombstones past the retention window (task 07).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def run_forgetting(engine) -> None:
    """Soft-delete dead-weight archival nodes, then purge expired tombstones."""
    if not engine.settings.deletion_enabled:
        return  # opt-in; the graveyard is untouched until explicitly armed
    try:
        now = datetime.now(timezone.utc)
        _run_gate_phase(engine, now)
        # Task 06 inserts the §3 cap backstop here.
        # Task 07 inserts Phase B (hard-purge) here.
    except Exception as e:
        logger.warning("Forgetting manager failed: %s", e)


def _run_gate_phase(engine, now: datetime) -> int:
    """Phase A: soft-delete archival nodes that are not protected AND are stale."""
    s = engine.settings
    candidates = [
        row["id"] for row in _archival_rows(engine)
        if not _is_protected(engine, row, now) and _is_stale_eligible(s, row, now)
    ]
    deleted = 0
    for node_id in candidates:
        if engine.delete_node_guarded(node_id, _eligibility_guard(engine, node_id, now)):
            deleted += 1
    if deleted:
        logger.info("Forgetting soft-deleted %d archival nodes", deleted)
    return deleted


def _hybrid_row(engine, node_id: str, conn):
    """A fresh gate row reading volatile fields from the SOURCE FILE, not the lagging index.

    Council R3 C5: mutators (`update_node`, `_touch_access`) write the markdown file BEFORE the
    index, so a guard reading only the index can see stale tier/last_accessed and delete a node
    mid-promotion. The file is authoritative for tier / last_accessed / archived_at / FSRS fields.
    `importance` stays index-authoritative (importance_scorer writes the index directly), and
    affinity/edges are read via conn (serialized by BEGIN IMMEDIATE). Returns None if the file is
    gone or no longer archival.
    """
    node = engine.file_store.load(node_id)
    if node is None or node.tier != Tier.archival:
        return None
    irow = conn.execute("SELECT importance FROM nodes WHERE id = ?", (node_id,)).fetchone()
    importance = irow["importance"] if irow and irow["importance"] is not None else node.importance
    return {
        "id": node.id,
        "importance": importance,
        "stability": node.stability,
        "last_review": node.last_review.isoformat() if node.last_review else None,
        "last_accessed": node.last_accessed.isoformat(),
        "archived_at": node.archived_at.isoformat() if node.archived_at else None,
    }


def _eligibility_guard(engine, node_id: str, now: datetime):
    """Build a guard(conn) that re-validates the gates from the source file inside the txn."""
    s = engine.settings

    def guard(conn) -> bool:
        row = _hybrid_row(engine, node_id, conn)
        if row is None:
            return False  # promoted / recalled-out / gone since selection
        return not _is_protected(engine, row, now) and _is_stale_eligible(s, row, now)

    return guard


# --- shared gate predicates -------------------------------------------------

_ROW_COLS = "id, importance, stability, last_review, last_accessed, archived_at"


def _archival_rows(engine):
    return engine.db.conn.execute(
        f"SELECT {_ROW_COLS} FROM nodes WHERE tier = 'archival'"
    ).fetchall()


def _evaluate_protection(engine, row, now: datetime) -> tuple[bool, int]:
    """§1 hard protections — single source of truth, computing connectivity at most once.

    Returns ``(protected, degree)``. ``protected`` True means NEVER delete (Phase A and the cap
    both honor it). ``degree`` is returned so the cap's forget-score never recomputes it (H5).
    Cheap protections short-circuit before the edge query, so Phase A stays cheap for the common
    high-importance / feedback cases.
    """
    s = engine.settings
    if row["id"] == getattr(engine, "user_node_id", None):
        return True, 0                                # gate #7: self node
    if row["archived_at"] is None:
        return True, 0                                # un-aged (e.g. remember(tier=archival))
    importance = row["importance"] if row["importance"] is not None else 0.5
    if importance >= s.decay_importance_threshold:
        return True, 0                                # gate #4: high importance
    if _has_positive_feedback(engine, row["id"]):
        return True, 0                                # gate #5: ever positively useful
    degree, max_weight = _connectivity(engine, row["id"])
    if degree > s.deletion_max_degree or max_weight >= s.deletion_strong_edge_weight:
        return True, degree                           # gate #6: hub / strong edge
    return False, degree


def _is_protected(engine, row, now: datetime) -> bool:
    return _evaluate_protection(engine, row, now)[0]


def _is_stale_eligible(s, row, now: datetime) -> bool:
    """§1 staleness signals — sustained dead weight. NOT protections (the cap skips these)."""
    cutoff = now - timedelta(days=s.deletion_min_archival_days)
    if _parse_dt(row["archived_at"]) > cutoff:
        return False                                  # gate #2: in graveyard long enough
    if _parse_dt(row["last_accessed"]) > cutoff:
        return False                                  # gate #2: not re-accessed
    if _retrievability(row, now) >= s.deletion_retrievability_floor:
        return False                                  # gate #3: R below the floor
    return True


def _retrievability(row, now: datetime) -> float:
    """FSRS retrievability R = exp(-days_since_anchor / stability). 1.0 if uncomputable."""
    stability = row["stability"] if row["stability"] else 1.0
    anchor_str = row["last_review"] or row["last_accessed"]
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except (ValueError, TypeError):
        return 1.0
    days_since = max((now - _aware(anchor)).total_seconds() / 86400, 0.001)
    return math.exp(-days_since / stability)


def _has_positive_feedback(engine, node_id: str) -> bool:
    row = engine.db.conn.execute(
        "SELECT 1 FROM affinity WHERE node_id = ? AND signal > 0 LIMIT 1", (node_id,)
    ).fetchone()
    return row is not None


def _connectivity(engine, node_id: str) -> tuple[int, float]:
    row = engine.db.conn.execute(
        "SELECT COUNT(*) AS degree, COALESCE(MAX(weight), 0) AS max_w "
        "FROM edges WHERE source_id = ? OR target_id = ?",
        (node_id, node_id),
    ).fetchone()
    return row["degree"], row["max_w"]


def _parse_dt(value: str) -> datetime:
    return _aware(datetime.fromisoformat(value))


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine/test_delete_guarded.py tests/test_background/test_forgetting_manager.py -v`
Expected: PASS (guard tests + all forgetting tests, including the 7-gate conjunction matrix).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/engine/memory_engine.py src/ormah/background/forgetting_manager.py tests/test_background/test_forgetting_manager.py tests/test_engine/test_delete_guarded.py
git add src/ormah/engine/memory_engine.py src/ormah/background/forgetting_manager.py tests/test_background/test_forgetting_manager.py tests/test_engine/test_delete_guarded.py
git commit -m "feat(background): forgetting gates + atomic guarded soft-delete (#28)"
```
