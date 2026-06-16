# Task 06: §3 cap backstop (forget-score eviction)

**Depends on:** Task 05.

After Phase A, if `archival_soft_cap > 0` and the archival count still exceeds it, evict the
worst nodes by a composite forget-score, worst-first, down to the cap — **respecting the exact
same `_is_protected` set as Phase A** (council C1). The cap deliberately ignores the *staleness*
signals (it may evict a node younger than the graveyard window — that is the point of a
backstop), but it never touches a **protected** node: self, `archived_at IS NULL`, high
importance, positive feedback, hub/strong-edge, degree > max. If the cap cannot reach the
target without violating a protection, it **accepts the overflow** (spec §3).

**Files:**
- Modify: `src/ormah/background/forgetting_manager.py`
- Test: `tests/test_background/test_forgetting_manager.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_background/test_forgetting_manager.py`:

```python
def _make_archival_recent(engine, content, archived_days, importance=0.1):
    """Archival node NOT gate-stale (recent access), eligible in BOTH file and index."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content=content, type=NodeType.fact, tier=Tier.archival, title=content))
    recent = datetime.now(timezone.utc) - timedelta(days=3)
    archived = datetime.now(timezone.utc) - timedelta(days=archived_days)
    node = engine.file_store.load(node_id)
    node.importance = importance
    node.stability = 1.0
    node.last_review = recent
    node.last_accessed = recent
    node.archived_at = archived
    path = engine.file_store.save(node)
    engine.builder.index_single(path)
    engine.db.conn.commit()
    return node_id


def _archival_count(engine):
    return engine.db.conn.execute(
        "SELECT COUNT(*) AS c FROM nodes WHERE tier='archival'").fetchone()["c"]


def test_cap_evicts_worst_down_to_cap(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 2
    ids = [_make_archival_recent(engine, f"n{i}", archived_days=age)
           for i, age in enumerate([300, 250, 200, 150, 50])]
    run_forgetting(engine)  # none are gate-stale → Phase A deletes nothing
    assert _archival_count(engine) == 2
    assert _exists(engine, ids[4]) is True   # youngest (least forgettable) survives
    assert _exists(engine, ids[3]) is True
    assert _exists(engine, ids[0]) is False  # oldest evicted


def test_cap_never_evicts_protected_high_importance(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    old_imp = _make_archival_recent(engine, "old important", archived_days=400, importance=0.9)
    mid = _make_archival_recent(engine, "mid", archived_days=200)
    young = _make_archival_recent(engine, "young", archived_days=20)
    run_forgetting(engine)
    assert _exists(engine, old_imp) is True   # protected by importance despite worst age
    assert _exists(engine, young) is True
    assert _exists(engine, mid) is False      # evicted to approach the cap


def test_cap_accepts_overflow_when_only_protected_remain(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    a = _make_archival_recent(engine, "fa", archived_days=300, importance=0.9)
    b = _make_archival_recent(engine, "fb", archived_days=250, importance=0.9)
    run_forgetting(engine)
    assert _archival_count(engine) == 2  # both protected → cap exceeded, nothing deleted


def test_cap_protects_feedback_node(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    fb = _make_archival_recent(engine, "fb node", archived_days=400)
    other = _make_archival_recent(engine, "other", archived_days=30)
    engine.db.conn.execute(
        "INSERT INTO affinity (prompt_vec, node_id, signal, source, confirmed_at, session_id) "
        "VALUES (?, ?, 1, 'explicit', ?, 's1')",
        (b"\x00", fb, datetime.now(timezone.utc).isoformat()))
    engine.db.conn.commit()
    run_forgetting(engine)
    assert _exists(engine, fb) is True       # feedback protects even the worst-scored
    assert _exists(engine, other) is False


def test_cap_disabled_by_default_zero(engine):
    _enable(engine)
    ids = [_make_archival_recent(engine, f"z{i}", archived_days=300) for i in range(4)]
    run_forgetting(engine)
    assert _archival_count(engine) == 4  # cap off → no eviction


def test_cap_protects_strong_edge_hub(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    a = _make_archival_recent(engine, "hub a", archived_days=400)
    b = _make_archival_recent(engine, "hub b", archived_days=380)
    filler = _make_archival_recent(engine, "filler", archived_days=20)
    engine.connect(ConnectRequest(source_id=a, target_id=b, edge=EdgeType.related_to, weight=0.9))
    run_forgetting(engine)
    assert _exists(engine, a) is True and _exists(engine, b) is True  # hub protected in cap
    assert _exists(engine, filler) is False


def test_cap_never_evicts_user_node(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 0  # force overflow regardless of count below
    uid = engine.user_node_id
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET tier='archival', importance=0.1, stability=1.0, "
        "last_review=?, last_accessed=?, archived_at=? WHERE id=?",
        (old, old, old, uid))
    engine.db.conn.commit()
    engine.settings.archival_soft_cap = 1
    _make_archival_recent(engine, "other", archived_days=30)
    run_forgetting(engine)
    assert _exists(engine, uid) is True  # self node never evicted by the cap
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_background/test_forgetting_manager.py -k cap -v`
Expected: FAIL (cap not wired; `test_cap_disabled_by_default_zero` already passes).

- [ ] **Step 3: Wire the cap into `run_forgetting`**

In `src/ormah/background/forgetting_manager.py`, replace the line:

```python
        # Task 06 inserts the §3 cap backstop here.
```

with:

```python
        _run_cap_backstop(engine, now)
```

- [ ] **Step 4: Implement the cap + forget-score**

Append these functions to `forgetting_manager.py`:

```python
def _run_cap_backstop(engine, now: datetime) -> int:
    """Evict worst-first by forget-score down to archival_soft_cap, respecting protections."""
    s = engine.settings
    if s.archival_soft_cap <= 0:
        return 0
    rows = _archival_rows(engine)
    overflow = len(rows) - s.archival_soft_cap
    if overflow <= 0:
        return 0

    scored: list[tuple[float, str]] = []
    for row in rows:
        protected, degree = _evaluate_protection(engine, row, now)  # connectivity once (H5)
        if protected:
            continue  # same hard protections as Phase A — accept overflow over deleting these
        scored.append((_forget_score(row, now, degree), row["id"]))

    scored.sort(reverse=True)  # highest forget-score (worst) first
    evicted = 0
    for _score, node_id in scored[:overflow]:
        # atomic delete-if-still-unprotected (council R2 C3) — staleness not required for the cap
        if engine.delete_node_guarded(node_id, _cap_guard(engine, node_id, now)):
            evicted += 1
    if evicted:
        logger.info("Forgetting cap backstop evicted %d archival nodes", evicted)
    return evicted


def _cap_guard(engine, node_id: str, now: datetime):
    def guard(conn) -> bool:
        row = _hybrid_row(engine, node_id, conn)  # source-of-truth recheck (council R3 C5)
        return row is not None and not _is_protected(engine, row, now)

    return guard


def _forget_score(row, now: datetime, degree: int) -> float:
    """Composite worst-first score: low R × low importance × age × low connectivity.

    Candidates already exclude protected nodes, so positive feedback never reaches here.
    """
    r = _retrievability(row, now)
    importance = row["importance"] if row["importance"] is not None else 0.5
    # archived_at is guaranteed non-null (NULL ⇒ protected), so age is well defined.
    age_days = max((now - _parse_dt(row["archived_at"])).total_seconds() / 86400, 0.0)
    return (1.0 - r) * (1.0 - importance) * age_days * (1.0 / (1 + degree))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_background/test_forgetting_manager.py -v`
Expected: PASS (all tests, including the Task 05 set).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/background/forgetting_manager.py tests/test_background/test_forgetting_manager.py
git add src/ormah/background/forgetting_manager.py tests/test_background/test_forgetting_manager.py
git commit -m "feat(background): cap backstop forget-score eviction respecting protections (#28)"
```
