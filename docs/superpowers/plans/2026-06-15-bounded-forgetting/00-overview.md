# Bounded Forgetting (#28) Implementation Plan — Overview

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Each subagent gets THIS overview + its own task file (`NN-*.md`). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an opt-in background job that deletes dead-weight `archival` nodes through a conjunction of safety gates, with a reversible soft-delete → hard-purge lifecycle and a cap backstop.

**Architecture:** One new job `forgetting_manager.run_forgetting(engine)`, two phases (gates→soft-delete + cap, then hard-purge), both behind a master switch `deletion_enabled` (default OFF). `archived_at` becomes a first-class node timestamp (frontmatter → index, since the SQLite index is *derived* and rebuilt from files). `deleted_at` is a tombstone stamped into the moved file's frontmatter, used to time the purge.

**Tech Stack:** Python 3.11, SQLite (`sqlite3` via `Database`), pydantic-settings, python-frontmatter, APScheduler, pytest (`asyncio_mode=auto`, default run excludes `integration`).

**Spec:** `docs/superpowers/specs/2026-06-15-bounded-forgetting-design.md`

---

## Conventions for every task

- Run tests with the working-tree interpreter: `.venv/bin/python -m pytest <path> -v`
  (per `[[ormah-dev-run-setup]]` — the global `uv`-tool ormah is a different version).
- Lint touched files: `.venv/bin/ruff check src/ tests/` (line-length 100, py311).
- The `engine` pytest fixture (`tests/conftest.py`) gives a live `MemoryEngine` with
  `engine.remember(CreateNodeRequest(...)) -> (node_id, msg)`, `engine.db.conn`,
  `engine.file_store`, `engine.builder`, `engine.update_node`, `engine.connect`,
  `engine.user_node_id`.
- Commit after each task with the message shown in its final step.

## File map

| File | Responsibility | Task |
|------|----------------|------|
| `src/ormah/config.py` | 8 new deletion settings + validators | 01 |
| `src/ormah/models/node.py`, `store/markdown.py`, `index/schema.sql`, `index/db.py`, `index/builder.py` | `archived_at` plumbing (model→frontmatter→schema→migration→index) | 02 |
| `src/ormah/engine/memory_engine.py` | stamp `archived_at` on demotion to archival (`update_node` chokepoint) | 03 |
| `src/ormah/store/file_store.py` | `soft_delete` stamps `deleted_at`; new `list_deleted()`, `purge()` | 04 |
| `src/ormah/background/forgetting_manager.py` | Phase A: gates → soft-delete | 05 |
| `src/ormah/background/forgetting_manager.py` | §3 cap backstop (forget-score eviction) | 06 |
| `src/ormah/engine/memory_engine.py` | `delete_node_guarded` — atomic delete-if-eligible (closes TOCTOU) | 05 |
| `src/ormah/background/forgetting_manager.py` | Phase B: hard-purge expired + audit | 07 |
| `src/ormah/background/scheduler.py` | register `forgetting_manager` job | 08 |
| `src/ormah/background/forgetting_manager.py` | lazy atomic legacy `archived_at` file backfill | 09 |

## Task order & dependencies

1. **01 Config** — no deps.
2. **02 archived_at plumbing** — no deps (schema/model; index-only migration backfill).
3. **03 Stamp on demotion** — needs 02 (stamp on entry, clear on exit).
4. **04 file_store tombstone** — no deps (can parallel 01–03).
5. **05 Forgetting gates (Phase A) + `delete_node_guarded`** — needs 01, 02, 03, 04.
6. **06 Cap backstop** — needs 05.
7. **07 Phase B purge** — needs 04 (and 05's module).
8. **08 Scheduler** — needs 05 (run_forgetting exists).
9. **09 Legacy file backfill** — needs 02, 03, 05 (durable file stamp for legacy nodes).

## Council revisions baked in (R1 + R2)

- **Cap reuses the protection set** (`_evaluate_protection`) — never deletes a protected node;
  accepts overflow instead. Staleness is Phase-A-only.
- **TOCTOU closed atomically** — `delete_node_guarded` re-checks inside the `BEGIN IMMEDIATE`
  deletion transaction (no global recall lock → no #18/#19 regression).
- **`archived_at` durable** — stamped on every archival entry, cleared on exit; legacy files
  backfilled atomically once (Task 09), proven across `full_rebuild`.

**Accepted known limitation (R4):** a microscopic deletion/mutation race remains because the
engine writes files before the index lock — see the spec's "Known limitation" section. Accepted
for ship (opt-in, OFF default, soft-delete reversible 30d); the root fix (mutators lock-before-
save) is a separate engine issue, not gated inside #28.

## Definition of done

- All new tests pass under `.venv/bin/python -m pytest tests/ -v`.
- `ruff check` clean on touched files.
- With `deletion_enabled=False` (default), the job is a verified no-op.
- Activation is env-only: `ORMAH_DELETION_ENABLED=true` in `~/.config/ormah/.env`.
