# Incremental auto-linker cursor — Implementation Plan (v2, post-council)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_auto_linker` (and the shared `_find_link_candidates` preview) process only nodes newer than a persisted **internal change-sequence** cursor, turning the per-run O(n²) full scan into O(batch·n), draining the historical backlog gradually, without missing nodes from reindex/import or transient LLM failures.

**Architecture:** A monotonic `nodes.seq` integer, bumped at the single content-write site (`Builder._index_file_nodes_only`), and an integer watermark in `meta`. `run_auto_linker` selects a bounded batch `WHERE seq > watermark ORDER BY seq ASC`, runs the unchanged inner logic, and advances the watermark only to the last **fully-resolved** node. `_find_link_candidates` reads the same window without advancing. Absent watermark = 0, so the backlog drains oldest-first.

**Tech Stack:** Python 3.11, SQLite, pytest (`asyncio_mode=auto`), existing `engine` fixture.

**Spec:** `docs/superpowers/specs/2026-06-15-auto-linker-incremental-cursor-design.md` (v2)
**Branch:** `perf/auto-linker-incremental` (base 0.11.0)
**Council:** v1 rejected — this plan addresses crit#1 (LLM None skip), crit#2 (domain-timestamp cursor → internal seq), imp#3 (bounded recall stated), imp#4 (max_edges test), minor#5 (index).

## File structure

- Modify `src/ormah/config.py` — setting `auto_link_max_nodes_per_run`.
- Modify `src/ormah/index/schema.sql` — `seq` column + index for fresh DBs.
- Modify `src/ormah/index/db.py` — `_migrate()` adds `seq` + index + backfill for existing DBs.
- Modify `src/ormah/index/builder.py` — bump `seq` after the content write.
- Modify `src/ormah/background/auto_linker.py` — watermark helpers, incremental select, rewrite `run_auto_linker` (with crit#1) and `_find_link_candidates`.
- Modify `tests/test_background/test_auto_linker.py` + `tests/test_index/` — tests.

Watermark: `meta.auto_link_watermark` = integer `seq` (text), absent ⇒ `0`.

---

### Task 1: Config setting

**Files:** Modify `src/ormah/config.py` (beside `auto_link_*`); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing test**

```python
def test_max_nodes_per_run_default(engine):
    assert engine.settings.auto_link_max_nodes_per_run == 500
```

- [ ] **Step 2:** `pytest tests/test_background/test_auto_linker.py::test_max_nodes_per_run_default -v` → FAIL (AttributeError).
- [ ] **Step 3: Add setting** in `config.py` beside other `auto_link_*`:

```python
    auto_link_max_nodes_per_run: int = 500  # cursor batch: nodes scanned per run
```

- [ ] **Step 4:** Re-run → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(config): add auto_link_max_nodes_per_run (#26)"`

---

### Task 2: `nodes.seq` schema + migration + backfill

**Files:** Modify `src/ormah/index/schema.sql`; Modify `src/ormah/index/db.py` (`_migrate`, ~L102-116); Test `tests/test_index/test_migration_seq.py` (create).

- [ ] **Step 1: Failing test**

```python
def test_seq_column_backfilled_by_created(engine):
    """Existing nodes get a monotonic seq ordered by created ASC."""
    # _create_pair inserts two nodes via the builder
    from tests.test_background.test_auto_linker import _create_pair
    id_a, id_b = _create_pair(engine)
    rows = engine.db.conn.execute(
        "SELECT id, seq FROM nodes WHERE id IN (?, ?) ORDER BY seq", (id_a, id_b)
    ).fetchall()
    seqs = [r["seq"] for r in rows]
    assert all(s > 0 for s in seqs)
    assert seqs[0] != seqs[1]  # unique, monotonic
```

- [ ] **Step 2:** `pytest tests/test_index/test_migration_seq.py -v` → FAIL (no `seq` column).
- [ ] **Step 3: Schema + migration.** In `schema.sql`, add to the `nodes` table definition (after `file_hash`):

```sql
    seq INTEGER NOT NULL DEFAULT 0
```

and after the existing `idx_nodes_*` indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_nodes_seq ON nodes(seq);
```

In `db.py` `_migrate()`, after the `enrichment_migrations` loop:

```python
            if "seq" not in node_cols:
                conn.execute("ALTER TABLE nodes ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_seq ON nodes(seq)")
                # Backfill existing rows: oldest (created ASC) gets the lowest seq,
                # so the historical backlog drains oldest-first.
                rows = conn.execute(
                    "SELECT id FROM nodes ORDER BY created ASC, rowid ASC"
                ).fetchall()
                for i, row in enumerate(rows, start=1):
                    conn.execute("UPDATE nodes SET seq = ? WHERE id = ?", (i, row[0]))
                # Initialize the durable monotonic counter past the backfilled max,
                # so future writes always allocate seq above any current watermark.
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_seq_next', ?)",
                    (str(len(rows) + 1),),
                )
```

- [ ] **Step 4:** Re-run → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(index): add monotonic nodes.seq column + backfill (#26)"`

---

### Task 3: Builder bumps `seq` on content write

**Files:** Modify `src/ormah/index/builder.py` (`_index_file_nodes_only`, after the `INSERT OR REPLACE` at ~L110-139); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing tests**

```python
def test_seq_bumped_on_rewrite(engine):
    """Re-writing a node's content bumps its seq to the head (crit#2 mechanism)."""
    from ormah.models.node import UpdateNodeRequest
    id_a, id_b = _create_pair(engine)
    seq_before = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    max_before = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]
    engine.update_node(id_a, UpdateNodeRequest(content="rewritten content"))
    seq_after = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    assert seq_after > seq_before
    assert seq_after > max_before  # landed at the head


def test_metadata_update_does_not_bump_seq(engine):
    """A direct metadata UPDATE (not via the builder) must not change seq."""
    id_a, _ = _create_pair(engine)
    before = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    with engine.db.transaction() as conn:
        conn.execute("UPDATE nodes SET access_count = access_count + 1 WHERE id=?", (id_a,))
    after = engine.db.conn.execute("SELECT seq FROM nodes WHERE id=?", (id_a,)).fetchone()["seq"]
    assert after == before
```

- [ ] **Step 2:** Run → FAIL (`test_seq_bumped_on_rewrite`: seq unchanged after rewrite).
- [ ] **Step 3: Bump seq** in `builder.py`, immediately after the `INSERT OR REPLACE INTO nodes (...)` `conn.execute(...)` block (before the Tags loop):

```python
        # Durable monotonic change-sequence (council v2 crit#1): allocate the next seq from
        # meta.node_seq_next — never decreases, independent of current rows, unlike MAX(seq)+1
        # which is non-monotonic across INSERT OR REPLACE. Every content (re)write lands the node
        # at the head, so reindex/import/restore re-enter the delta regardless of frontmatter
        # timestamps. Metadata-only UPDATEs elsewhere do not pass through here.
        row = conn.execute("SELECT value FROM meta WHERE key = 'node_seq_next'").fetchone()
        next_seq = int(row[0]) if row else 1
        conn.execute("UPDATE nodes SET seq = ? WHERE id = ?", (next_seq, node.id))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_seq_next', ?)",
            (str(next_seq + 1),),
        )
```

- [ ] **Step 4:** Run → PASS (both tests).
- [ ] **Step 5: Commit** `git commit -am "feat(index): bump nodes.seq on content write in builder (#26)"`

---

### Task 4: Watermark helpers (integer seq)

**Files:** Modify `src/ormah/background/auto_linker.py` (top, after imports); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing test**

```python
def test_watermark_roundtrip(engine):
    from ormah.background.auto_linker import _get_watermark, _set_watermark
    assert _get_watermark(engine.db.conn) == 0
    _set_watermark(engine, 42)
    assert _get_watermark(engine.db.conn) == 42
```

- [ ] **Step 2:** Run → FAIL (ImportError).
- [ ] **Step 3: Implement** in `auto_linker.py` (`import json` already present):

```python
_WATERMARK_KEY = "auto_link_watermark"


def _get_watermark(conn) -> int:
    """Return the seq of the last fully-processed node, or 0 if unset."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (_WATERMARK_KEY,)).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (ValueError, TypeError):
        return 0


def _set_watermark(engine, seq: int) -> None:
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (_WATERMARK_KEY, str(seq)),
        )
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(background): integer seq watermark helpers (#26)"`

---

### Task 5: Incremental node select by seq

**Files:** Modify `src/ormah/background/auto_linker.py`; Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing test**

```python
def test_select_nodes_after_seq(engine):
    from ormah.background.auto_linker import _select_nodes_after
    id_a, id_b = _create_pair(engine)
    rows = _select_nodes_after(engine.db.conn, 0, limit=10)
    assert {id_a, id_b} <= {r["id"] for r in rows}
    last = rows[-1]
    rows2 = _select_nodes_after(engine.db.conn, last["seq"], limit=10)
    assert all(r["id"] != last["id"] for r in rows2)
    assert len(_select_nodes_after(engine.db.conn, 0, limit=1)) == 1
```

- [ ] **Step 2:** Run → FAIL (ImportError).
- [ ] **Step 3: Implement**

```python
def _select_nodes_after(conn, watermark: int, limit: int) -> list:
    """Nodes with seq strictly greater than the watermark, ascending, bounded."""
    return conn.execute(
        "SELECT id, content, title, type, space, seq FROM nodes "
        "WHERE seq > ? ORDER BY seq ASC LIMIT ?",
        (watermark, limit),
    ).fetchall()
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(background): incremental seq node select (#26)"`

---

### Task 5b: `error` classification for invalid LLM output (poison-node guard, council v2 crit#2)

**Files:** Modify `src/ormah/background/auto_linker.py` (`_llm_classify_link` ~L105-117, `_apply_edge` ~L251); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing test**

```python
def test_invalid_llm_output_records_error_not_none(engine):
    """Malformed LLM JSON → recorded as result='error' (no edge), so the node resolves."""
    id_a, id_b = _create_pair(engine)
    engine.settings.llm_provider = "ollama"; engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    with patch(_LLM_PATCH, return_value="not valid json"):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)
    assert len(_edges_between(engine, id_a, id_b)) == 0  # no edge
    pair = tuple(sorted([id_a, id_b]))
    row = engine.db.conn.execute(
        "SELECT result FROM auto_link_checked WHERE node_a=? AND node_b=?", pair
    ).fetchone()
    assert row is not None and row["result"] == "error"
```

- [ ] **Step 2:** Run → FAIL (invalid JSON currently returns None → no record).
- [ ] **Step 3a:** In `_llm_classify_link`, return an `error` classification (not `None`) for invalid/unexpected output; keep `None` only for an unavailable LLM:

```python
    raw = llm_generate(settings, prompt, json_mode=True)
    if raw is None:
        return None  # LLM UNAVAILABLE — transient; caller leaves the node unresolved

    try:
        result = json.loads(raw)
        if "relationship" not in result:
            return {"relationship": "error", "reason": "missing relationship field"}
        result["relationship"] = normalize_link_type(result["relationship"])
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for link classification")
        return {"relationship": "error", "reason": "invalid LLM output"}
```

- [ ] **Step 3b:** In `_apply_edge`, treat `error` like `none` — record the checked pair but create no edge. Change both guards from `if edge_type != "none":` to:

```python
        if edge_type not in ("none", "error"):
```

(both the `edges` INSERT block and the markdown-connection block).

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(background): classify invalid LLM output as error, not None (#26)"`

---

### Task 6: `run_auto_linker` drives the cursor (with crit#1)

**Files:** Modify `src/ormah/background/auto_linker.py` (`run_auto_linker`); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing tests**

```python
def test_run_advances_watermark(engine):
    from ormah.background.auto_linker import run_auto_linker, _get_watermark, _select_nodes_after
    _create_pair(engine)
    engine.settings.llm_provider = "ollama"; engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=json.dumps({"relationship": "none", "reason": "x"})):
        run_auto_linker(engine)
    last = _select_nodes_after(engine.db.conn, 0, limit=100)[-1]
    assert _get_watermark(engine.db.conn) == last["seq"]


def test_llm_none_does_not_advance_past_node(engine):
    """crit#1: a transient None must not let the watermark pass the node."""
    from ormah.background.auto_linker import run_auto_linker, _get_watermark
    _create_pair(engine)
    engine.settings.llm_provider = "ollama"; engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    with patch(_LLM_PATCH, return_value=None):
        run_auto_linker(engine)
    # No node fully resolved → watermark stays at 0
    assert _get_watermark(engine.db.conn) == 0
    # Next run with the LLM healthy re-evaluates the pair
    mock_llm = MagicMock(return_value=json.dumps({"relationship": "supports", "reason": "x"}))
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)
    assert mock_llm.call_count >= 1


def test_max_edges_does_not_skip_interrupted_node(engine):
    """imp#4: max_edges mid-run must not advance the watermark past unprocessed nodes."""
    from ormah.background.auto_linker import run_auto_linker, _get_watermark, _select_nodes_after
    # three mutually-similar nodes
    _create_pair(engine, title_a="A", content_a="shared topic alpha", title_b="B", content_b="shared topic alpha beta")
    _create_pair(engine, title_a="C", content_a="shared topic alpha gamma", title_b="D", content_b="shared topic alpha delta")
    engine.settings.llm_provider = "ollama"; engine.settings.auto_link_similarity_threshold = 0.0
    engine.settings.auto_link_max_edges_per_run = 1
    _reset_adapter()
    rows = _select_nodes_after(engine.db.conn, 0, limit=100)
    with patch(_LLM_PATCH, return_value=json.dumps({"relationship": "supports", "reason": "x"})):
        run_auto_linker(engine)
    wm = _get_watermark(engine.db.conn)
    assert wm < rows[-1]["seq"]  # did not reach the last node
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Rewrite `run_auto_linker`**

```python
def run_auto_linker(engine) -> None:
    """Incrementally link nodes with seq above the watermark; advance only past
    fully-resolved nodes."""
    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore

        settings = engine.settings
        if not settings.llm_enabled:
            logger.debug("Auto-linker skipped: LLM not enabled")
            return

        encoder = get_encoder(settings)
        vec_store = VectorStore(engine.db)
        conn = engine.db.conn
        threshold = settings.auto_link_similarity_threshold
        cross_space_penalty = settings.auto_link_cross_space_penalty
        max_edges = settings.auto_link_max_edges_per_run

        watermark = _get_watermark(conn)
        nodes = _select_nodes_after(conn, watermark, settings.auto_link_max_nodes_per_run)

        created = 0
        last_complete: int | None = None

        for node in nodes:
            if created >= max_edges:
                break  # batch budget spent; do not advance past this node

            node_resolved = True
            text = f"{node['title'] or ''} {node['content']}".strip()
            if text:
                query_vec = encoder.encode(text)
                similar = vec_store.search(query_vec, limit=6)

                for match in similar:
                    if created >= max_edges:
                        node_resolved = False  # interrupted mid-node
                        break
                    if match["id"] == node["id"]:
                        continue

                    similarity = match["similarity"]
                    other_space = conn.execute(
                        "SELECT space FROM nodes WHERE id = ?", (match["id"],)
                    ).fetchone()
                    if other_space is not None and (node["space"] or "") != (other_space["space"] or ""):
                        similarity -= cross_space_penalty
                    if similarity < threshold:
                        continue

                    pair = tuple(sorted([node["id"], match["id"]]))
                    if conn.execute(
                        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
                    ).fetchone():
                        continue
                    if conn.execute(
                        "SELECT 1 FROM edges WHERE (source_id = ? AND target_id = ?) "
                        "OR (source_id = ? AND target_id = ?)",
                        (node["id"], match["id"], match["id"], node["id"]),
                    ).fetchone():
                        continue
                    other = conn.execute(
                        "SELECT title, content, type, space FROM nodes WHERE id = ?",
                        (match["id"],),
                    ).fetchone()
                    if other is None:
                        continue

                    llm_result = _llm_classify_link(settings, node, other)
                    if llm_result is None:
                        # LLM UNAVAILABLE (raw None) — transient. Leave node unresolved so the
                        # watermark does not pass it. Not a poison: if the LLM is down, EVERY node
                        # is unresolved, so the whole run waits — no single node blocks others.
                        node_resolved = False
                        continue
                    relationship = llm_result["relationship"]  # may be 'error' (invalid output)
                    _apply_edge(
                        engine, node["id"], match["id"], relationship,
                        llm_result.get("reason", ""), similarity,
                    )
                    # 'error' (poison content) is recorded in auto_link_checked by _apply_edge
                    # and the node still counts as resolved → watermark advances (council v2 crit#2).
                    if relationship not in ("none", "error"):
                        created += 1

            if not node_resolved:
                break  # crit#1/imp#4: stop; watermark stays at the last fully-resolved node
            last_complete = node["seq"]

        if last_complete is not None:
            _set_watermark(engine, last_complete)
        if created:
            logger.info("Auto-linker created %d edges", created)

    except Exception as e:
        logger.warning("Auto-linker failed: %s", e)
```

- [ ] **Step 4:** Run the full file → PASS, including existing `test_llm_*` / `test_checked_pairs_*`.
- [ ] **Step 5: Commit** `git commit -am "feat(background): drive auto_linker by seq cursor, retry on LLM failure (#26)"`

---

### Task 7: `_find_link_candidates` shares the incremental scan

**Files:** Modify `src/ormah/background/auto_linker.py` (`_find_link_candidates`); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing test**

```python
def test_find_candidates_uses_window_without_advancing(engine):
    from ormah.background.auto_linker import _find_link_candidates, _get_watermark
    _create_pair(engine)
    engine.settings.auto_link_similarity_threshold = 0.0
    before = _get_watermark(engine.db.conn)
    cands = _find_link_candidates(engine, limit=8)
    assert all("node_a" in c and "node_b" in c and "similarity" in c for c in cands)
    assert _get_watermark(engine.db.conn) == before  # preview never advances the cursor
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Replace the node fetch** in `_find_link_candidates` (remove the `ORDER BY RANDOM()` line; the candidate-collection loop bounded by `limit` is unchanged):

```python
        conn = engine.db.conn
        watermark = _get_watermark(conn)
        nodes = _select_nodes_after(conn, watermark, settings.auto_link_max_nodes_per_run)
```

- [ ] **Step 4:** Run → PASS; existing `TestFindLinkCandidates` / `test_run_maintenance` stay green.
- [ ] **Step 5: Commit** `git commit -am "feat(background): share seq scan in _find_link_candidates (#26)"`

---

### Task 7b: `full_rebuild` resets the watermark (council v2 crit#1)

**Files:** Modify `src/ormah/index/builder.py` (`full_rebuild`, ~L23-31); Test `tests/test_background/test_auto_linker.py`.

- [ ] **Step 1: Failing test**

```python
def test_full_rebuild_resets_watermark(engine):
    """A mass reindex must not leave a stale watermark hiding the whole store."""
    from ormah.background.auto_linker import _set_watermark, _get_watermark
    _create_pair(engine)
    _set_watermark(engine, 99999)
    engine.builder.full_rebuild()
    assert _get_watermark(engine.db.conn) == 0
```

- [ ] **Step 2:** Run → FAIL (watermark stays 99999).
- [ ] **Step 3:** In `Builder.full_rebuild`, after it has deleted/reindexed the nodes (so the reset is not clobbered), clear the watermark — the single point all rebuild paths funnel through (startup FTS migration, `rebuild_index`/`POST /rebuild`, restore):

```python
        # Mass reindex re-allocates seq from the durable counter; clear the watermark so the
        # rebuilt store is reprocessed even if the counter was also reset (wiped meta).
        self.db.conn.execute("DELETE FROM meta WHERE key = 'auto_link_watermark'")
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(index): reset auto_link watermark on full_rebuild (#26)"`

---

### Task 8: Full suite + lint

- [ ] **Step 1:** `pytest tests/test_background/ tests/test_index/ -v` → all green.
- [ ] **Step 2:** `ruff check src/ tests/` → clean.
- [ ] **Step 3 (optional smoke, real store):** restart `.venv/bin/ormah server`; after a maintenance run, confirm `meta.auto_link_watermark` advanced and equals a real `seq`:
  `sqlite3 ~/.local/share/ormah/memory/index.db "SELECT (SELECT value FROM meta WHERE key='auto_link_watermark') AS wm, (SELECT MAX(seq) FROM nodes) AS maxseq"`

---

## Self-review

- **Council coverage:** crit#1 (LLM None) → Task 6 `node_resolved`/`test_llm_none_does_not_advance_past_node` ✓; crit#2 (domain-timestamp → internal seq) → Tasks 2/3 + `test_seq_bumped_on_rewrite` ✓; imp#3 (bounded recall) → spec "Correctness" updated ✓; imp#4 (max_edges test) → `test_max_edges_does_not_skip_interrupted_node` ✓; minor#5 (index) → `idx_nodes_seq` in Task 2 ✓.
- **Placeholder scan:** none — every code step shows full code.
- **Type consistency:** `_get_watermark(conn) -> int`, `_set_watermark(engine, seq: int)`, `_select_nodes_after(conn, watermark: int, limit)` used identically in Tasks 6 and 7; `auto_link_max_nodes_per_run` consistent; `seq` column consistent across Tasks 2/3/5/6.
- **Out of scope (unchanged):** LLM edge-quality, ANN (#25), revisiting bounded recall, embedding reuse.
