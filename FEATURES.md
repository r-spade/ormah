# Ormah — Features Reference

A complete catalogue of everything ormah does. Use this to update the website, write copy, or plan what to highlight.

---

## Core Concept

Ormah is a **local-first persistent memory system for AI agents**. It gives Claude (and other agents) a long-term memory that persists across conversations, learns what matters to you, and surfaces the right context at the right time — automatically.

---

## Memory Model

### Nodes

Every memory is a **node** with:

| Field | Description |
|-------|-------------|
| **Content** | The memory text |
| **Title** | Short searchable label |
| **Type** | `fact`, `decision`, `preference`, `event`, `person`, `project`, `concept`, `procedure`, `goal`, `observation` |
| **Tier** | `core` (always loaded), `working` (searchable), `archival` (deep storage) |
| **Space** | Project namespace — auto-detected from the current git repo |
| **Confidence** | 0–1 belief strength (uncertain memories are deprioritised) |
| **Importance** | 0–1 dynamic score computed from access frequency, centrality, and recency |
| **Valid Until** | Optional expiry date — expired memories are demoted in search |
| **Tags** | Categorical labels |
| **Source** | Where the memory came from (`agent`, `cli`, `consolidator`, etc.) |

### Edges (Relationships)

Memories are connected via typed **edges**:

| Type | Meaning |
|------|---------|
| `supports` | Provides evidence for |
| `part_of` | Is a sub-component of |
| `depends_on` | Requires to function |
| `defines` | Defines or describes |
| `derived_from` | Synthesised from (used after consolidation) |
| `evolved_from` | Newer version of (temporal succession) |
| `related_to` | Semantically related |
| `instance_of` | Is an example of a category |
| `caused_by` | Resulted from |
| `preceded_by` | Happened after (temporal ordering) |
| `contradicts` | Logically incompatible with |

Edges have a **weight** (0–1) and optional **reason** (LLM-generated explanation).

### Tiers

- **Core** — Always included in context. Capped at 50 nodes. Best for identity, preferences, and key architectural decisions.
- **Working** — Actively searchable. Decays to archival after ~14 days of inactivity (FSRS spaced repetition).
- **Archival** — Deep storage. Still searchable but deprioritised. Can be promoted back by access.

---

## How Memories Are Retrieved

### Hybrid Search (FTS + Vector)

Every recall query runs two searches in parallel and fuses the results:

1. **Full-text search (FTS5)** — BM25-style keyword matching on title, content, and tags
2. **Vector similarity** — Cosine similarity via `sqlite-vec` (768-dim embeddings, `BAAI/bge-base-en-v1.5`)
3. **Reciprocal Rank Fusion (RRF)** — Merges the ranked lists using `score = Σ weight / (k + rank)`, with configurable FTS/vector weights

**Post-fusion scoring boosts:**
- Title keyword match → 2× multiplicative boost
- Core tier → +0.1
- Archival tier → −0.1
- Recency (exponential decay with 7-day half-life) → up to +0.05
- Access frequency (log-normalised) → up to +0.05
- Question queries (where/what/who/why/how) → auto-scale toward vector search

### Spreading Activation

After finding the top-5 seed results, ormah traverses the knowledge graph:
- Follows edges up to 2 hops deep
- Weights decay per hop (default 0.5×)
- Brings in related memories not directly matched by the query

### Whisper Reranking

For context injection specifically, a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) re-scores candidates by true semantic relevance:

- Formula: `0.4 × sigmoid(cross_encoder_score) + 0.6 × embedding_score`
- Prevents over-filtering semantically valid matches that lack keyword overlap
- Uses ONNX runtime (no PyTorch/CUDA required)

---

## Automatic Context Injection (Whisper)

The flagship feature. Before every Claude prompt, ormah injects the most relevant memories from your knowledge graph — silently, as a whisper.

**How it works:**
1. `UserPromptSubmit` hook fires before Claude sees the prompt
2. Ormah searches for memories relevant to your current prompt + recent conversation context
3. Topic-shift detection skips injection if the current topic is the same as the last (avoids redundancy)
4. Budget-aware: up to 1500 chars total, 100–600 per node
5. Injection gate: only injects if blended relevance score ≥ 0.55
6. Formatted as a markdown context block prepended to the prompt

**Session awareness:**
- Maintains a rolling buffer of your recent 5 prompts per session
- Recognises session boundaries (10-minute gap = new session)

---

## Background Intelligence

8 background jobs run continuously while the server is alive:

| Job | Frequency | What it does |
|-----|-----------|-------------|
| **Auto-linker** | Daily | Discovers relationships between memories using embeddings + LLM; creates typed edges automatically |
| **Conflict detector** | Daily | Finds contradictions between memories; classifies as evolution (newer supersedes older) or tension (simultaneous conflict) |
| **Duplicate merger** | Daily | Detects near-duplicate memories using embedding + title + token overlap; LLM validates and merges; fully reversible |
| **Consolidator** | Daily | Groups similar working-tier memories; LLM synthesises them into a single richer memory; originals archived |
| **Importance scorer** | Every 2h | Recomputes importance for every node based on access frequency, graph centrality, and recency |
| **Decay manager** | Daily | Demotes working-tier memories to archival when retrievability drops below 0.3 (FSRS algorithm) |
| **Auto-cluster** | Every 1h | Assigns orphaned memories to the right project space via majority vote of neighbours |
| **Index updater** | Every 1m | Incremental FTS + vector index updates for newly created or modified nodes |

All jobs are **pause/resume/manually-triggerable** from the UI admin panel.

---

## Spaced Repetition (FSRS)

Working-tier memories follow the **FSRS spaced repetition model**:

- **Stability (S)** — Days until ~37% retrievability. Grows by 1.5× each time a memory is accessed; capped at 365 days.
- **Retrievability (R)** — `R = exp(−days_since_review / stability)`
- When `R < 0.3`, the memory is demoted to archival (unless importance ≥ 0.5)
- Initial stability: 1.0 day (seeded from access history on first run)

This means frequently-used memories stay active; neglected ones gracefully fade.

---

## Ingestion

### Conversation Ingestion

Feed any conversation text to ormah and let the LLM extract memorable information:

- Via REST: `POST /ingest/conversation`
- Via CLI: `ormah ingest <file>`
- Dry-run mode: preview extraction without storing
- Deduplication: skips near-duplicate nodes (embedding + token overlap check)
- Auto-tags and auto-types each extracted memory

### Claude Code Session Watcher

Optional: ormah watches `~/.claude/projects/` for Claude Code JSONL transcripts:

- Automatically ingests sessions after they end (debounced 60s)
- Min-turn filter (default 5 turns) to skip trivial sessions
- Space auto-detected from the project directory name
- State-tracked so no session is double-ingested

### Hippocampus (File Watcher)

Watch any local directories for markdown files:

- Detects creates and modifications (debounced 2s)
- Hash-based change detection (no re-processing on timestamp-only updates)
- Useful for syncing Obsidian vaults, notes, or any markdown-based second brain

---

## Insights

### Belief Evolutions

When a memory's view has changed over time, ormah detects the succession and creates `evolved_from` edges. The Insights view shows:

- Old view → New view, with the reason the change was detected

### Tensions

When two simultaneously-held memories contradict each other, ormah flags them as a tension and creates `contradicts` edges. You can resolve them through the review queue.

---

## Review Queue

Ormah never modifies memories autonomously without your awareness:

- **Merge proposals** — Shows the two nodes and the proposed merged result; approve or reject
- **Conflict proposals** — Shows the tension; approve (creates `contradicts` edge) or reject
- **Merge undo** — All merges are fully reversible from the merge history

---

## Storage

### SQLite Database (`memory/index.db`)

All nodes, edges, vectors, proposals, merge history, and audit logs live in a single SQLite file. Tables:

- `nodes` — Core memory records
- `edges` — Typed relationships
- `node_tags` — Tag assignments
- `nodes_fts` — FTS5 virtual table (full-text search)
- `node_vectors` — 768-dim embeddings (sqlite-vec)
- `proposals` — Pending merge/conflict actions
- `merge_history` — Undo-able merge records
- `auto_link_checked` — Pair-checked cache (prevents re-analysis)
- `audit_log` — All deletions, updates, mark-outdated operations
- `meta` — Key-value metadata store

### Markdown File Backup (`memory/nodes/*.md`)

Every node also exists as a human-readable `.md` file with frontmatter. Easy to inspect, version-control, or audit without touching the database.

---

## Embeddings

| Provider | Details |
|----------|---------|
| **Local** (default) | FastEmbed + ONNX runtime; `BAAI/bge-base-en-v1.5` (768-dim); **no PyTorch, no CUDA, ~16MB runtime** |
| **Ollama** | Local Ollama server with any supported model |
| **LiteLLM** | Any of 100+ cloud embedding APIs |

Default model download: ~420MB (one-time, cached).

---

## LLM Integration

Used for: conversation ingestion, auto-linking, conflict detection, duplicate validation, consolidation.

| Provider | Details |
|----------|---------|
| **LiteLLM** (default) | Gateway to 100+ APIs; default model: `claude-haiku-4-5-20251001` |
| **Ollama** | Fully local; default model: `llama3.2` |
| **None** | Disable all LLM features — ormah still works for search + injection, just no extraction or auto-linking |

LLM is **optional** — ormah is fully functional for storage, search, and whisper without any LLM configured.

---

## MCP Tools (Agent-Facing API)

7 tools exposed over the Model Context Protocol to Claude and other MCP clients:

| Tool | What it does |
|------|-------------|
| `remember` | Store a new memory (content, type, tier, space, tags, confidence, about_self) |
| `recall` | Hybrid search with natural language |
| `get_context` | Load core memories for system prompt; accepts `task_hint` to filter to most relevant |
| `get_self` | Get the user's identity profile (all personal/preference memories) |
| `mark_outdated` | Mark a memory as stale with an optional reason |

---

## REST API

Full programmatic access over HTTP:

- **`/agent/*`** — Memory operations (remember, recall, context, whisper, proposals, audit, merge undo)
- **`/admin/*`** — Server health, stats, job management, manual job triggers, index rebuild
- **`/ingest/*`** — Conversation and file ingestion
- **`/ui/*`** — Frontend data (graph, search, insights, node detail)

---

## CLI

`ormah <command>` — 20+ commands:

```
# Server
ormah server start [-d]       # foreground or daemon
ormah server stop
ormah server status

# Setup
ormah setup                   # one-shot interactive setup
ormah mcp                     # run MCP stdio server

# Memory
ormah remember <content>      # --type --tier --tags --about-self --space
ormah recall <query>          # --limit --types --json --space
ormah context                 # --task --space
ormah node <id>               # --json

# Ingestion
ormah ingest <file>           # --space
ormah ingest-session <path>   # --dry-run --space --min-turns

# Hooks (called by Claude Code)
ormah whisper inject          # inject context before prompt
ormah whisper store           # extract memories at session end
```

---

## Claude Code Integration

### Automatic Hooks

After `ormah setup`, three hooks fire automatically in Claude Code:

| Hook | Event | Action |
|------|-------|--------|
| `whisper inject` | Before every prompt | Injects relevant memories into the prompt context |
| `whisper store` | At session compaction | Extracts and stores new memories from the conversation |
| `whisper store` | At session end | Same — ensures nothing is lost |

### CLAUDE.md Block

`ormah setup` writes a block into `~/.claude/CLAUDE.md` that teaches Claude about the memory tools, guidelines for when to save, and how to handle project vs global memories.

### MCP Server

Registered in `~/.claude/settings.json` so all 7 tools are available to Claude Code via MCP.

---

## Web UI

React + TypeScript (Vite), served by the ormah server and bundled inside the Python package — no separate install.

| View | What you see |
|------|-------------|
| **Graph** | Interactive Cytoscape.js graph of all nodes + edges. Nodes coloured by tier. Filterable by tier, type, space, edge type. Cluster by project. |
| **Search** | Hybrid search with score, type, and tier badges. Click → detail panel + graph focus. |
| **Node Detail** | Full metadata: content, edges, tags, confidence, importance, timestamps. |
| **Review Queue** | Pending merge and conflict proposals. Approve / reject. |
| **Insights Panel** | Belief evolutions and simultaneous tensions. |
| **Admin Panel** | Background job status, next-run times, pause/resume, manual trigger, health snapshot. |

---

## Setup & Installation

### One-liner

```bash
curl -fsSL https://ormah.me/install.sh | sh
```

### What happens:

1. Detects platform (macOS / Linux)
2. Bootstraps `uv` package manager
3. Installs ormah from PyPI
4. Interactive setup:
   - Your name
   - LLM provider + API key (optional — skip for offline use)
5. Starts server
6. Registers MCP in Claude Code
7. Installs whisper hooks
8. Writes `CLAUDE.md` block
9. Configures auto-start service (launchd on macOS, systemd on Linux)

### Service Management

- **macOS:** `~/Library/LaunchAgents/com.ormah.server.plist` — `KeepAlive=true`, starts on login
- **Linux:** `~/.config/systemd/user/ormah.service` — `Restart=on-failure`, user-level service

---

## Configuration

All via `.env` at `~/.config/ormah/.env` (or environment variables):

| Category | Key vars |
|----------|---------|
| **Server** | `ORMAH_PORT` (8787), `ORMAH_HOST`, `ORMAH_LOG_FORMAT` |
| **Paths** | `ORMAH_MEMORY_DIR` |
| **Embeddings** | `ORMAH_EMBEDDING_PROVIDER`, `ORMAH_EMBEDDING_MODEL`, `ORMAH_EMBEDDING_DIM` |
| **LLM** | `ORMAH_LLM_PROVIDER`, `ORMAH_LLM_MODEL`, `ORMAH_LLM_BASE_URL` |
| **Search** | `ORMAH_FTS_WEIGHT`, `ORMAH_VECTOR_WEIGHT`, `ORMAH_SIMILARITY_THRESHOLD`, `ORMAH_RRF_K` |
| **Whisper** | `ORMAH_WHISPER_MAX_NODES`, `ORMAH_WHISPER_MIN_RELEVANCE_SCORE`, `ORMAH_WHISPER_RERANKER_ENABLED` |
| **FSRS** | `ORMAH_FSRS_INITIAL_STABILITY`, `ORMAH_FSRS_DECAY_THRESHOLD`, `ORMAH_FSRS_MAX_STABILITY` |
| **Jobs** | Per-job interval vars for all 8 background jobs |
| **Hippocampus** | `ORMAH_HIPPOCAMPUS_WATCH_DIRS`, `ORMAH_HIPPOCAMPUS_ENABLED` |
| **Session watcher** | `ORMAH_SESSION_WATCHER_ENABLED`, `ORMAH_SESSION_WATCHER_DIR` |
| **Limits** | `ORMAH_CORE_MEMORY_CAP` (50), `ORMAH_CONTEXT_MAX_NODES` (20) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Database | SQLite + FTS5 + sqlite-vec |
| Embeddings | FastEmbed (ONNX runtime) — no PyTorch required |
| Scheduling | APScheduler |
| File watching | Watchdog |
| LLM gateway | LiteLLM |
| MCP | `mcp` Python SDK |
| Frontend | React + TypeScript + Vite + Cytoscape.js |
| Package manager | uv |
| Distribution | PyPI |

---

## Design Principles

- **Local-first** — All data stays on your machine. No cloud, no telemetry.
- **No PyTorch** — Embeddings and reranking use ONNX runtime (FastEmbed). Installs in seconds, not gigabytes.
- **LLM-optional** — Core memory storage, search, and whisper work without any LLM configured.
- **Single binary** — `uv tool install ormah` and you're done.
- **Non-intrusive** — Whisper injects context silently; you never have to think about it.
- **Transparent** — Every autonomous action (merge, link, conflict flag) is reviewable and reversible.
- **Project-aware** — Memories are automatically scoped to the current git repo. Cross-project recall still works.
