# Storage Layer

Verified against the current repository state on 2026-04-07.

Ormah uses a dual-storage model:

- markdown node files are the durable source of truth
- SQLite is a derived index for search, graph traversal, metadata, and operational logs

## Source of Truth vs Derived Index

The intended contract is:

1. write durable memory state to markdown
2. derive SQLite state from those markdown files
3. rebuild the SQLite side whenever necessary

That means markdown persistence is not just an export format. It is the canonical persisted node representation.

## Markdown Node Files

**Code**: `src/ormah/store/file_store.py`, `src/ormah/store/markdown.py`

Each memory node is stored as a markdown file under the configured memory directory.

Typical persisted fields include:

- id
- title
- content
- type
- tier
- space
- created / updated timestamps
- tags
- connections
- validity / decay-related metadata

`FileStore` handles:

- loading nodes from disk
- saving nodes atomically
- soft deletion
- path resolution

## SQLite Index

**Code**: `src/ormah/index/schema.sql`, `src/ormah/index/db.py`, `src/ormah/index/builder.py`

SQLite stores the derived operational view of memory.

Important table families:

- `nodes`
- `edges`
- `node_tags`
- FTS tables for full-text search. FTS tables are SQLite's full-text search indexes. Ormah uses them to do fast, ranked keyword retrieval over memory titles, content, and tags. Unlike normal tables, which store rows for exact lookup and filtering, FTS tables are optimized for searching natural-language text.
- vector-search storage used by `sqlite-vec`
- maintenance/audit tables such as `proposals`, `merge_history`, `audit_log`
- whisper / feedback tables such as `whisper_log`, `review_log`, `affinity`

## Index Builder

`IndexBuilder` is responsible for translating markdown node files into SQLite rows and search indexes.

Key responsibilities:

- full rebuild from markdown files
- indexing a single saved node
- incremental update of changed content
- removing deleted nodes from derived state

For normal Ormah-managed writes, the engine saves markdown and then updates the derived index immediately.

## Graph Access

**Code**: `src/ormah/index/graph.py`

`GraphIndex` reads from SQLite and provides:

- FTS search helpers
- graph neighbor lookup
- edge lookup
- whole-graph fetches for UI

## What Happens On `remember()`

```mermaid
sequenceDiagram
    participant Agent as client
    participant Engine as MemoryEngine
    participant Store as FileStore
    participant Builder as IndexBuilder
    participant DB as SQLite

    Agent->>Engine: remember(...)
    Engine->>Store: save markdown node
    Store-->>Engine: node path
    Engine->>Builder: index_single(path)
    Builder->>DB: update nodes / tags / edges / FTS / vec state
    Engine->>Engine: optional inline auto-linking
```

## Rebuild Behavior

Because SQLite is derived, the index can be rebuilt from the markdown layer.

Operationally this happens through:

- `engine.rebuild_index()`
- `POST /admin/rebuild`
- startup / maintenance paths that invoke index-building logic

## Manual Edits and Rebuild Behavior

There is a watcher implementation in:

- `src/ormah/store/watcher.py`

It can watch the node directory for `.md` changes and invoke a callback.

However, in the current app runtime this watcher is **not wired into startup**. `main.py` starts:

- APScheduler
- hippocampus observers
- session watcher observers

It does **not** start `store.watcher.start_watcher()`.

- Ormah-managed writes update both markdown and the derived index
- manual node-file edits can be picked up by rebuild or incremental index operations
- there is currently no app-started live node-file watcher for the node store itself

## One Server Per Memory Directory

Only one Ormah process may own a memory directory at a time. The server keeps an
in-memory id-to-path cache and an in-process lock that no other process can see, so a
second server on the same directory (for example a dev server started with a different
`ORMAH_PORT`) could overwrite or lose writes.

On startup, before the engine is created, the server takes an exclusive, non-blocking OS
lock on `<memory_dir>/.ormah/server.lock` and holds it until it shuts down. The kernel drops
the lock if the process dies, so a crash never leaves a stale lock. A second server fails
immediately with a message like:

```text
another Ormah server (pid 4242, port 8787, started 2026-11-13T09:00:00+00:00) is already
using /Users/you/.local/share/ormah/memory. Only one Ormah server can use a memory
directory at a time. Stop it, or give this one its own directory with ORMAH_MEMORY_DIR.
```

`.ormah/server.json` records the holder's pid, port and start time so the message can name
it. It is informational only; the lock is the truth. To run a dev server, give it its own
directory: `ORMAH_MEMORY_DIR=/tmp/ormah-dev ORMAH_PORT=8790 ormah server start`.

`ormah backup restore` takes the same lock for the duration of the restore, so it refuses
while any server (on any port) uses the directory, and no server can start mid-restore.
The lock is separate from `.ormah/store.lock`, which cloud protection takes briefly while
the server runs. It does not protect against non-Ormah writers such as folder sync or hand
edits.

## Walkthrough Example

If you manually edit a node markdown file on disk:

1. the markdown file changes immediately
2. SQLite does **not** automatically update through a live node-store watcher
3. the change becomes visible to derived search state only after an index rebuild / update path processes it

This is different from hippocampus, which is a separate watcher for external content ingestion.

## Code Anchors

- `src/ormah/store/file_store.py`
- `src/ormah/store/markdown.py`
- `src/ormah/store/watcher.py`
- `src/ormah/server_lock.py`
- `src/ormah/index/schema.sql`
- `src/ormah/index/builder.py`
- `src/ormah/index/graph.py`
