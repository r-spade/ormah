# Task 03: Stamp `archived_at` on entering archival, clear on leaving

**Depends on:** Task 02.

The single chokepoint for tier changes is `MemoryEngine.update_node`. Stamp `archived_at`
every time a node *enters* the archival tier (old != archival, new == archival), and **clear**
it when the node *leaves* archival (promoted back to working/core). Stamping on the transition
(not on `is None`) is what makes the graveyard clock honest across an archival→working→archival
lifecycle — a re-archived node must not look months old (council R1, Codex H1).

**Files:**
- Modify: `src/ormah/engine/memory_engine.py` (`update_node`)
- Test: `tests/test_engine/test_archived_at.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine/test_archived_at.py`:

```python
from __future__ import annotations

from ormah.models.node import CreateNodeRequest, NodeType, Tier, UpdateNodeRequest


def _archived_at(engine, node_id):
    row = engine.db.conn.execute(
        "SELECT archived_at FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return row["archived_at"]


def test_demotion_to_archival_stamps_archived_at(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="demote me", type=NodeType.fact, tier=Tier.working, title="d"))
    assert _archived_at(engine, node_id) is None

    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))

    assert _archived_at(engine, node_id) is not None
    assert engine.file_store.load(node_id).archived_at is not None  # source of truth


def test_non_archival_update_does_not_stamp(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="rename me", type=NodeType.fact, tier=Tier.working, title="r"))
    engine.update_node(node_id, UpdateNodeRequest(title="renamed"))
    assert _archived_at(engine, node_id) is None


def test_metadata_edit_while_archival_keeps_archived_at(engine):
    """A metadata edit (no tier change) must not move the clock."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="x", type=NodeType.fact, tier=Tier.working, title="x"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    first = _archived_at(engine, node_id)
    engine.update_node(node_id, UpdateNodeRequest(title="x2"))  # no tier change
    assert _archived_at(engine, node_id) == first


def test_leaving_archival_clears_archived_at(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="y", type=NodeType.fact, tier=Tier.working, title="y"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    assert _archived_at(engine, node_id) is not None
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.working))  # promoted out
    assert _archived_at(engine, node_id) is None


def test_re_entering_archival_restamps_fresh(engine):
    """archival → working → archival must reset the clock, not keep the old one."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="z", type=NodeType.fact, tier=Tier.working, title="z"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.working))   # clears
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))  # re-stamps
    assert _archived_at(engine, node_id) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine/test_archived_at.py -v`
Expected: FAIL (`archived_at` not stamped/cleared on tier transitions).

- [ ] **Step 3: Implement the transition logic**

In `src/ormah/engine/memory_engine.py` `update_node`, replace the existing tier-apply line:

```python
        if req.tier is not None:
            node.tier = req.tier
```

with:

```python
        if req.tier is not None:
            old_tier = node.tier
            node.tier = req.tier
            if req.tier == Tier.archival and old_tier != Tier.archival:
                node.archived_at = datetime.now(timezone.utc)  # entered the graveyard
            elif req.tier != Tier.archival and old_tier == Tier.archival:
                node.archived_at = None  # left archival → reset the graveyard clock
```

`Tier` and `datetime`/`timezone` are already imported in this module (used in `update_node`).
Verify `Tier` is in the `from ormah.models.node import ...` line; if missing, add it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine/test_archived_at.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Regression — decay still works and now stamps**

Run: `.venv/bin/python -m pytest tests/test_background/test_decay_manager.py -v`
Expected: PASS (decay demotes via update_node; archived_at set as a side effect).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/engine/memory_engine.py tests/test_engine/test_archived_at.py
git add src/ormah/engine/memory_engine.py tests/test_engine/test_archived_at.py
git commit -m "feat(engine): stamp archived_at on entering archival, clear on leaving (#28)"
```
