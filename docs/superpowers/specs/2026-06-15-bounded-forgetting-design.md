# Bounded Forgetting (#28) — Design

**Issue:** #28 — `feat(background): bounded forgetting — delete dead-weight archival nodes`
**Branch:** off `perf/auto-linker-incremental` (#26 not yet merged to main)
**Date:** 2026-06-15

## Problem

Decay is a one-way ratchet. `decay_manager` demotes stale `working` nodes to `archival`
(`src/ormah/background/decay_manager.py`) but nothing ever removes them. `archival` only
grows. On André's real store (8.355 nodes, 80% archival) that graveyard:

- counts toward total node count (graph UI struggles),
- is scanned by every background job (auto-linker is O(n²) over *all* nodes),
- still surfaces in recall as noise (only down-ranked via `tier_boost_archival = -0.1`).

Decay reduces *priority*, not *cost* or *noise*. **Deletion is the only lever that shrinks
the store.** Shrinking `n` is also a free multiplier for #26 (auto-linker) and #25 (vector
search).

## Design constraint

Deletion is irreversible and memory lives on trust. A false negative — deleting something
that later mattered — is unrecoverable. Therefore:

> Deletion never depends on a single signal. It acts only on the `archival` tail, requires
> the conjunction of several independent signals, and keeps a reversibility window before
> anything leaves disk. Every decision is explainable per node.

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Full issue: §1 gates + §2 soft→hard-purge + §3 cap backstop | André opted for the complete lifecycle in one delivery. |
| Default state | Opt-in, **OFF** (`deletion_enabled = False`) | Irreversible action in a trust-critical system → fail-closed default; user arms it explicitly. |
| Enablement surface | **Env / `.env` only, no UI toggle** | The web UI is too slow to open at André's node count. All config is pydantic-settings (`ORMAH_` prefix), read from `~/.config/ormah/.env` and `./.env`. |
| Purge timing | `deleted_at` stamped in the moved file's **frontmatter** | Self-contained: survives backup/restore and mtime/TCC-copy resets on this machine. |
| "archived ≥ N days" signal | Dedicated **`nodes.archived_at`** column | Only robust way to express graveyard age without coupling to unrelated `update_node` writes. |

## Architecture: one new background job

New module `src/ormah/background/forgetting_manager.py` → `run_forgetting(engine)`,
registered in `src/ormah/background/scheduler.py` with its own interval. Each run has two
phases, both guarded by the master switch `deletion_enabled` (when `False`, the job returns
immediately — a no-op):

### Phase A — gates → soft-delete (+ cap backstop)

1. Select `archival` candidates (SQL prefilter on cheap columns).
2. Apply the conjunction gates (§1). For each eligible node, call `engine.delete_node(id)`,
   which already does: index removal, `audit_log` (`operation='delete'`), `auto_link_checked`
   cleanup, and `file_store.soft_delete` (move to `deleted/`).
3. If `archival_soft_cap > 0` and the surviving archival count still exceeds it, evict
   worst-first by forget-score (§3) down to the cap, **respecting every protection in §1**.

### Phase B — hard-purge

1. `file_store.list_deleted()` → iterate files in `deleted/`.
2. Parse `deleted_at` from each file's frontmatter.
3. For files older than `deletion_retention_days`, `file_store.purge(node_id)` (remove from
   disk) and write `audit_log` (`operation='purge'`).

Reusing `engine.delete_node` for soft-delete keeps a single deletion path (index + audit +
auto-link cleanup all handled), so the new job stays small.

## Schema change (one migration)

Add `nodes.archived_at TEXT` (nullable):

- Set when a node is demoted to `archival` — in the `decay_manager` demotion path (and any
  other path that writes `tier = archival`).
- Migration in `Database._migrate` adds the column and backfills existing `archival` rows
  with their current `updated` value (best available proxy for legacy data). The backfill
  writes the **source markdown files**, not only the index: the index is derived, so an
  index-only backfill is wiped by the next `full_rebuild`/restore (which re-parses
  `archived_at=None`), permanently excluding legacy archival nodes from forgetting. A test
  asserts the value survives a full rebuild.

## file_store changes

- `soft_delete(node_id)` — before/while moving to `deleted/`, stamp `deleted_at: <iso>` into
  the file's frontmatter. Centralized here, so manual `delete_node` also gets a timed
  reversibility window.
- `list_deleted()` — new: enumerate files currently in `deleted/`.
- `purge(node_id)` — new: hard-remove a file from `deleted/`.

## §1 Eligibility gates (delete only when ALL hold)

Over `tier == archival` candidates; cheap predicates in SQL, FSRS `R` computed in Python
(same formula as `decay_manager`: `R = exp(-days_since_anchor / stability)`).

1. `tier == archival` — never `working`/`core`.
2. `archived_at <= now − deletion_min_archival_days` **AND**
   `last_accessed <= now − deletion_min_archival_days` — sustained staleness, not a point
   reading.
3. `R < deletion_retrievability_floor` — retrievability below a hard floor (deeper than the
   decay demotion threshold).
4. `importance < decay_importance_threshold` — high importance never deletes.
5. `NOT EXISTS (SELECT 1 FROM affinity WHERE node_id = ? AND signal > 0)` — never positively
   useful (any `submit_feedback(+1)` or positive affinity ⇒ protected forever). Ties into #21.
6. `degree <= deletion_max_degree` **AND** no edge with `weight >= deletion_strong_edge_weight`
   — leaves are safe; never delete a bridge/hub.
7. Not the user/self node. (No "pin" concept exists in the schema today; protection is the
   self node. If a pin feature is added later, it must be respected here.)

Each gate alone produces false positives; the conjunction only catches genuine dead weight:
archival, unimportant, unused, never useful, weakly connected, old.

The gates split into two kinds, sharing one `_is_protected(node)` predicate reused by Phase A
and the §3 cap so they can never disagree:

- **Protections (hard "never delete"):** self node, `archived_at IS NULL`, high importance
  (#4), positive feedback (#5), hub/strong-edge or degree > max (#6). `archived_at IS NULL`
  covers nodes created directly as `archival` via `remember(tier=archival)` — un-aged, so
  protected.
- **Staleness signals (sustained dead weight):** archived long enough (#2), not re-accessed
  (#2), `R` below the floor (#3). Phase A requires *not protected AND stale*.

**`archived_at` lifecycle:** stamped on **every** transition into archival and **cleared** when
the node leaves archival — so an `archival → working → archival` round-trip resets the clock
instead of reusing a stale months-old timestamp.

**Atomic delete-if-eligible:** background jobs run concurrently, so eligibility is rechecked
**inside** the `BEGIN IMMEDIATE` deletion transaction via `engine.delete_node_guarded(id, guard)`.
The guard is **hybrid**: it reads the volatile protective fields (`tier`, `last_accessed`,
`archived_at`) from the **source markdown file**, because mutators (`update_node`,
`_touch_access`) write the file *before* the index — a guard reading only the index could act on
stale state and delete a node mid-promotion. `importance`/affinity/edges stay index-authoritative
and are serialized by the transaction. The deletion **moves the file first, then removes the
index row** inside the same transaction, so a crash leaves at worst a dangling index row (healed
on rebuild), never a resurrected node. No global lock around `recall` — that would reintroduce
the #18/#19 contention PR#19 fixed. The soft-delete is reversible for `deletion_retention_days`,
so even a missed race is recoverable, not irreversible.

## §3 Cap backstop (forget-score)

When `archival_soft_cap > 0` and exceeded, evict worst-first by a composite forget-score:

```
score = (1 − R) · (1 − importance) · age_days · 1/(1 + degree) · no_positive_feedback
```

where `age_days = now − archived_at`. The cap evaluates candidates through the **same
`_is_protected` predicate** as Phase A, so protected nodes (self, null-`archived_at`, high
importance, positive feedback, hub) are never evicted — the forget-score only orders the
*unprotected* remainder. The cap deliberately ignores the staleness signals (it may evict a
node younger than the graveyard window — that is the backstop's purpose). Sort descending,
evict down to the cap. If only protected nodes remain, the store stays above the cap — better
to exceed the cap than delete a valuable memory.

## Config (new fields in `src/ormah/config.py`, `ORMAH_` env prefix)

| Field | Default | Env var |
|---|---|---|
| `deletion_enabled` | `False` | `ORMAH_DELETION_ENABLED` |
| `forgetting_interval_hours` | `24` | `ORMAH_FORGETTING_INTERVAL_HOURS` |
| `deletion_min_archival_days` | `90` | `ORMAH_DELETION_MIN_ARCHIVAL_DAYS` |
| `deletion_retrievability_floor` | `0.05` | `ORMAH_DELETION_RETRIEVABILITY_FLOOR` |
| `deletion_max_degree` | `2` | `ORMAH_DELETION_MAX_DEGREE` |
| `deletion_strong_edge_weight` | `0.7` | `ORMAH_DELETION_STRONG_EDGE_WEIGHT` |
| `deletion_retention_days` | `30` | `ORMAH_DELETION_RETENTION_DAYS` |
| `archival_soft_cap` | `0` (off) | `ORMAH_ARCHIVAL_SOFT_CAP` |

Reuses existing `decay_importance_threshold` (0.5) for gate #4. Validators mirror the
existing decay/threshold validators (positive intervals, `0..1` floors).

## Operation

Enablement and tuning are env-driven, no UI involvement. To activate on André's machine:
add `ORMAH_DELETION_ENABLED=true` (and any tuning overrides) to `~/.config/ormah/.env`, then
restart the server. The slow web UI is never on the critical path.

## Testing (TDD)

Unit tests (mock external boundaries, real SQLite index):

- **Per gate, positive + negative:** stale-enough vs too-recent (`archived_at`,
  `last_accessed`), `R` below vs above floor, importance below vs above threshold, positive
  affinity present (protected) vs absent, degree/strong-edge below vs above limits, self node
  skipped.
- **Conjunction:** a node failing exactly one gate is not deleted; a node passing all is
  soft-deleted.
- **Master switch:** `deletion_enabled = False` ⇒ zero deletions (no-op), even with eligible
  nodes.
- **`soft_delete` stamps `deleted_at`** in frontmatter; file lands in `deleted/`.
- **Phase B purge:** files past `deletion_retention_days` are removed; files inside the window
  survive; `audit_log` records `operation='purge'`.
- **Cap backstop:** with `archival_soft_cap` set, worst-first eviction down to the cap;
  protected/feedback-positive/hub nodes never evicted even if it leaves the count above cap.
- **`archived_at`:** set on demotion in `decay_manager`; migration backfills existing
  archival rows from `updated`.
- **Idempotence:** a second run with no newly-eligible nodes deletes nothing.

## Known limitation: deletion/mutation race (accepted, tracked separately)

Four council rounds converged on this: the engine's mutators (`update_node`, `_touch_access`)
write the markdown **file before** acquiring the index transaction, and file operations are not
covered by `Database.transaction`'s lock. So a vanishingly small window remains: between the
guard's source-file read and `soft_delete`'s move, a concurrent recall/promotion could re-save
the file, and the node would be soft-deleted anyway. The hybrid guard + move-first ordering
narrow this to microseconds but cannot close it at the forgetting layer.

**Why it is acceptable to ship:** the feature is opt-in and OFF by default; deletion is **soft**
and reversible for `deletion_retention_days` (30); and the precondition (a node untouched for
90+ days being recalled in the exact microsecond of a daily job) makes the race astronomically
improbable and fully recoverable.

**Root fix (separate issue — #29):** reorder the elegibility-affecting mutators to acquire
`db.transaction()` **before** `file_store.save()`, making the index/lock authoritative so the
guard's `BEGIN IMMEDIATE` serializes everything. This touches the recall hot path, so it is its
own change with explicit sign-off — not gated inside #28 (a global recall lock is rejected: it
reintroduces the #18/#19 contention PR#19 fixed).

## Out of scope

- UI changes (the issue's #22 active-graph-first is separate).
- Restore-from-`deleted/` UX (the reversibility window exists; a restore command is not part
  of this slice).
- #25 ANN / vector-search work (deferred per the perf roadmap).
- The deletion/mutation race root fix (see Known limitation above) — separate engine issue.
