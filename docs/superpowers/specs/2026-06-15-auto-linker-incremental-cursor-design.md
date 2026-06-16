# Auto-linker incremental cursor (#26)

**Date:** 2026-06-15
**Branch:** `perf/auto-linker-incremental` (off 0.11.0)
**Status:** revised after council review (v2) — pending re-review
**Issue:** upstream #26 — `auto_linker` is O(n²): full-table scan × brute-force vector search × LLM-per-candidate every run.

## Problem

`run_auto_linker` (`src/ormah/background/auto_linker.py`) scans **every** node each run and,
per node, re-encodes its text, runs a brute-force `vec_store.search` (itself O(n)), and calls
the LLM per surviving candidate. Total cost is **O(n²)** compute + O(candidates) LLM calls,
repeated every `auto_link_interval_minutes` (default 1440 = 24h). `_find_link_candidates`
(the maintenance-protocol preview, via `get_maintenance_batches` → `maintenance_manager`)
has the **same shape**, additionally sorting with `ORDER BY RANDOM()`.

`auto_link_checked` already skips already-evaluated pairs, but only *after* the expensive
encode+search — so the scan cost is unbounded in `n` regardless. On the real store
(8.355 nodes) most of each run is spent re-encoding and re-searching nodes that yield
nothing (4.776 nodes — 57% — have never produced an evaluated pair).

## Approach (revised: internal change-sequence cursor)

> **Council change (crit. #2):** an earlier draft used a `(updated, id)` watermark over the
> domain timestamp. But `created`/`updated` come from the markdown frontmatter
> (`builder.py:128`), so historical imports, restores, and index rebuilds insert rows with
> timestamps **behind** the watermark — permanently invisible to the scan. Replaced with an
> internal monotonic change-sequence that is bumped wherever node *content* is (re)written.

A single mechanism serves both **backfill** (drain the historical backlog) and **incremental**
(only new/changed nodes), differing only in the watermark's initial value.

### State & config
- **New column `nodes.seq INTEGER NOT NULL DEFAULT 0`** + index `idx_nodes_seq`. A monotonic
  change-sequence, **not** a domain timestamp.
- **`seq` is allocated from a durable monotonic counter** `meta.node_seq_next` (never
  decreases, independent of the current rows) at the single content-write site
  `Builder._index_file_nodes_only` (`src/ormah/index/builder.py:103`), through which `remember`,
  the file-watcher, **and reindex/import** all pass. After the `INSERT OR REPLACE INTO nodes`,
  read+bump the counter and set the node's `seq` to it. So every (re)write of content — including
  reindex of an old markdown file — lands the node at the **head** of the sequence, always above
  any watermark. Metadata-only `UPDATE nodes SET ...` (access_count, importance, tier) deliberately
  do **not** bump `seq` (they don't change the embedding, so they must not trigger re-linking).
  > **Council v2 (crit #1):** an earlier draft used `MAX(seq)+1`, which is not monotonic across
  > `INSERT OR REPLACE` (the deleted row's seq is recomputed) — a durable counter fixes this.
- **`full_rebuild()` resets the watermark to 0** (`builder.py:23` — the single point all rebuild
  paths funnel through: startup FTS migration, `rebuild_index`/`POST /rebuild`, restore). Defensive:
  the durable counter already keeps rebuilt nodes above the watermark, but the reset also covers a
  wiped-meta rebuild. (Council v2.)
- **`meta.auto_link_watermark`** — the integer `seq` of the last fully-processed node, stored
  as text. Absent ⇒ `0` ⇒ backlog drains from the lowest seq. (`meta` already holds
  `last_maintenance_run`.)
- **New setting `auto_link_max_nodes_per_run: int = 500`** — bounds the scan (outer loop).
  `auto_link_max_edges_per_run` (500) stays as a secondary write guard.
- **Migration** in `Database._migrate()` (idempotent `ALTER TABLE`, same pattern as
  `edges.reason` at `db.py:100`): add the column + index, then backfill existing rows
  `seq = row_number ordered by created ASC` (oldest = lowest seq, so the backlog drains oldest-first).

### `run_auto_linker`
Replace `SELECT ... FROM nodes` (all rows) with:

```sql
SELECT id, content, title, type, space, seq FROM nodes
WHERE seq > :watermark ORDER BY seq ASC LIMIT :max_nodes_per_run
```

The inner match logic (encode → `vec_store.search` → cross-space penalty → threshold →
`auto_link_checked`/existing-edge skip → `_llm_classify_link` → `_apply_edge`) is unchanged.
The watermark advances to the `seq` of the last **fully-resolved** node (see Correctness).
Per-run cost: **O(batch · n)** instead of O(n²).

### `_find_link_candidates`
Shares the same incremental select (`seq > watermark ORDER BY seq ASC`), **reads** the
watermark but never advances it (it is a side-effect-free preview). Replaces `ORDER BY RANDOM()`.

## Correctness

- **New & changed nodes both re-enter Δ (crit. #2 fixed):** every content write bumps `seq`
  past the watermark via the single builder site, so reindex/import/restore are covered by
  construction — independent of markdown timestamps. Node update also deletes the node's
  `auto_link_checked` rows (`memory_engine.py` :806/:850/:1201/:1206), so its pairs are
  re-evaluated.
- **Transient LLM failure retries; poison node never deadlocks (crit. #1 v1 + crit. #2 v2):**
  `_llm_classify_link` distinguishes two cases. **LLM unavailable** (`raw is None`) → returns
  `None` → the node is left unresolved and the watermark does not pass it (next run retries). This
  is the transient case (Ollama down) and does not deadlock: when the LLM is down, *every* node
  is unresolved, so the whole run simply waits — no single node blocks others. **Invalid/unexpected
  LLM output** (malformed JSON or missing `relationship`) → returns `{"relationship": "error"}` →
  `_apply_edge` records `result='error'` in `auto_link_checked` (no edge) and the node is treated
  as resolved, so the watermark advances. A content-specific "poison" pair therefore never blocks
  later nodes. (An `error` pair is re-tried only if the node's content changes — which bumps `seq`
  and clears its `auto_link_checked` rows.)
- **Bounded recall, stated honestly (importante #3):** `vec_store.search(limit=6)` returns only
  the top-6 neighbours, and top-k membership is **not** symmetric — a pair where neither node
  is in the other's top-6 is never evaluated. This is a pre-existing limit of the full scan
  too; the cursor additionally removes the re-scan "second chance". Accepted as bounded recall
  (not lossless); revisiting strategies are out of scope for this perf fix.
- **Partial run / crash / `max_edges` hit mid-node:** the watermark only advances to the last
  fully-resolved node; reprocessing is idempotent (checked pairs skipped; no duplicate edges).
- **No ties:** `seq` is unique and monotonic, so the cursor is a single integer (no composite
  comparison needed).

## Testing (TDD — tests first)

1. `seq` is bumped on content write (insert and re-write via the builder) and is monotonic.
2. Metadata-only updates (e.g. access_count) do **not** bump `seq`.
3. Run processes only `seq > watermark`, never the full table; respects the batch limit.
4. Watermark advances to the last fully-resolved node's `seq`; never past unprocessed nodes.
5. **LLM `None` (crit. #1):** run 1 returns `None` for the pair → watermark does not pass the
   node → run 2 re-evaluates it.
6. **`max_edges` mid-node (importante #4):** ≥3 linkable nodes, `max_nodes`/`max_edges=1`, LLM
   always creates an edge → after run 1 the watermark sits on the 1st node; run 2 processes the 2nd.
7. **Reindex/import (crit. #2):** re-indexing an old markdown file bumps its `seq` so it
   re-enters Δ even though its `updated` timestamp is behind the watermark.
8. Backlog drains across multiple runs; once Δ is empty a run is cheap.
9. Absent watermark ⇒ 0; empty store ⇒ no-op.
10. `_find_link_candidates` returns only the seq-window, without advancing the watermark.
11. Existing `auto_link_checked` tests stay green (no re-check, invalidation on update).

## Out of scope

- LLM edge-quality (local gemma3:4b classifies ~90% of pairs as `supports`, ~16.7k edges).
- ANN for the vector search itself (upstream #25) — benefit compounds once added.
- Revisiting the bounded-recall limit (symmetric candidate generation) — separate concern.
- Reusing stored embeddings instead of re-encoding each Δ node — possible later optimization.
