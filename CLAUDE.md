# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
make install     # Install Python package (dev mode) + UI deps
make dev         # Start backend + UI together
make server      # Start backend only (auto-reloads)
make ui-dev      # Start Vite dev server (hot-reload)
make restart     # Rebuild UI and restart backend
make test        # Run pytest suite
make lint        # Run ruff linter (src/, tests/)
```

Run a single test: `uv run pytest tests/path/to/test.py::test_name -v`

The CLI is also the primary interface:
```bash
ormah server start [-d]    # Start server (foreground or daemon)
ormah setup                # One-shot setup (hooks, MCP, server)
ormah mcp                  # Run MCP stdio server
```

## Architecture

Ormah is a local-first persistent memory system for AI agents. It exposes memory operations via MCP (for Claude), REST API, and CLI.

### Core Concepts

**Nodes** are memories with: content, type (fact/decision/preference/event/etc.), tier (core/working/archival), space (project namespace), confidence (0–1), importance (0–1), and FSRS stability for spaced repetition decay.

**Edges** are typed relationships between nodes (supports, contradicts, part_of, evolved_from, etc.) with a weight and optional reason.

**Space** scopes memories to a project — auto-detected from the git repo name.

### Layer Stack

```
Adapters (MCP, CLI, OpenAI)
    ↓
API (FastAPI routes: /agent, /admin, /ui, /ingest)
    ↓
MemoryEngine  ←  Background Jobs (auto-linker, conflict detector, decay, etc.)
    ↓
Embeddings layer (hybrid FTS5 + vector via sqlite-vec, fused with RRF)
    ↓
Index layer (SQLite: nodes, edges, proposals, audit_log)
    ↓
Store layer (markdown files in memory/nodes/*.md as backup)
```

### Key Implementation Details

- **Hybrid search**: FTS5 full-text + vector similarity fused via Reciprocal Rank Fusion (RRF). Weights configurable via `ORMAH_FTS_WEIGHT` / `ORMAH_VECTOR_WEIGHT`.
- **Embeddings**: Default provider is `local` with `BAAI/bge-base-en-v1.5` (768-dim, no task prefixes needed). Also supports ollama and litellm.
- **Background jobs** (APScheduler): auto-linker, conflict detector, duplicate merger, importance scorer, decay manager, consolidator, hippocampus (file watcher), session watcher.
- **Tier decay**: working-tier memories decay after ~14 days using FSRS spaced repetition; `core` is capped at 50.
- **MCP tools exposed**: `remember`, `recall`, `get_context`, `get_self`, `mark_outdated`, `ingest_conversation`, `get_insights`.

### Frontend

React + TypeScript (Vite). Key views: search results, node detail, graph visualization (Cytoscape.js with fcose layout), review queue, insights panel, admin panel. API client in `ui/src/api.ts`.

### Configuration

Via `.env` (see `.env.example`). Key vars:
- `ORMAH_PORT` (default 8787), `ORMAH_MEMORY_DIR`
- `ORMAH_EMBEDDING_MODEL`, `ORMAH_EMBEDDING_PROVIDER`
- `ORMAH_LLM_PROVIDER`, `ORMAH_LLM_MODEL` (for background jobs)

---

## Ormah Memory System

You have access to a persistent memory system (via MCP tools) that maintains knowledge across conversations.

### Tools

- **remember**: Store new facts, decisions, preferences, events, or any knowledge worth retaining. Set `confidence` (0.0–1.0) to indicate belief strength when storing uncertain information. Set `about_self` to true for identity/preference memories.
- **recall**: Search for relevant memories using natural language. Results are ranked by importance and confidence; expired memories are demoted.
- **get_context**: Load core memories (call at conversation start). Pass `task_hint` to filter context to only the most relevant memories for the current task instead of loading everything.
- **get_self**: Get the user's identity profile — name, preferences, and personal facts. Returns all identity-linked memories.
- **mark_outdated**: Mark a memory as no longer valid. Optionally provide a `reason`. Outdated memories are heavily demoted in search results.
- **ingest_conversation**: Bulk-import memories from raw conversation text. The server extracts memorable information via its LLM. Use this for importing logs or transcripts. For individual memories you've already identified, use `remember` directly.
- **get_insights**: Show belief evolutions and conflicting ideas detected by the system. Evolutions are cases where the user's view changed over time; conflicting ideas are genuinely incompatible beliefs held simultaneously.

### Project Awareness

Memories are automatically scoped to the current project directory. The MCP server detects the project from the git repo name (or directory name as fallback) and uses it as the default `space` for all operations:

- **`remember`**: Memories automatically get `space` set to the current project. Explicitly set `space` to `null` for personal/global memories (identity, preferences, cross-project facts).
- **`recall`**: Results are prioritized — current project first, then global (`space=null`), then other projects.
- **`get_context`**: Returns core memories (all projects) + working-tier memories for the current project. With `task_hint`, returns only the top-N most relevant memories.
- **Cross-project recall**: Memories from other projects are included in `recall` results (with lower priority), enabling cross-project knowledge.

### Guidelines

1. **Proactively remember**: When the user shares important information (preferences, decisions, facts about themselves or their projects), store it *without being asked*. Names, preferences, project context, technical decisions — all worth remembering.
2. **Remember at natural save points**: Call `remember` immediately after these events — don't wait for the conversation to end:
   - **After committing code**: Architectural decisions, design patterns chosen, and the reasoning behind them.
   - **After choosing between alternatives**: Why option A was picked over B (e.g., "chose bge-base-en-v1.5 over nomic-embed because it needs no task prefixes").
   - **After completing a feature or fix**: What was built, how it works, and key implementation details a future session would need.
   - **After the user states a preference or corrects you**: Their preferred approach, style, or constraints.
   Each memory should be self-contained — someone reading it later should understand it without the original conversation.
3. **Notice what stands out**: Humans form strong memories around novelty, mistakes, and emotion. Use the same instincts:
   - **Something unexpected happened** (a bug had a surprising cause, a library behaved differently than docs say) → remember the lesson.
   - **The user corrected you or said "no"** → remember what they wanted instead and why.
   - **You tried something and it failed** → remember what didn't work so you don't repeat it.
   - **The user repeated themselves or said "I already told you"** → something was missed. Store it at `core` tier so it's never missed again.
   - **A pattern is emerging** (user keeps preferring X over Y, the codebase follows a convention) → remember the pattern.
4. **Check before assuming**: Use `recall` to search for relevant context before making assumptions about past conversations. For personal info (name, location, preferences), prefer `get_self` — it returns all identity-linked memories directly.
5. **Memory supports the flow, not the other way around**: Don't let recalled memories override or derail the current working context. If you're mid-task and `recall` returns something from a different context, let it go — stay in the flow. Use `recall` when you're genuinely unsure or the user asks about something from a prior session. Memory should feel like a natural extension of your knowledge, not an interruption. A whisper, not a shout.
6. **Keep memories atomic**: One concept per memory. The system automatically links related memories.
7. **Use appropriate tiers**: `core` for always-relevant info (user identity, preferences, key architectural decisions), `working` for active project details, `archival` for historical/reference data.
8. **Tag and categorize**: Use tags and spaces to organize memories for efficient retrieval.
9. **Start with context**: Call `get_context` at the beginning of conversations to load core memories. Use `task_hint` when the task is known to get focused context.
10. **Global vs project memories**: Use `space=null` explicitly for memories that apply everywhere (user identity, general preferences). Let project-specific memories use the auto-detected space.
11. **Mark outdated info**: When a memory is wrong or outdated, call `mark_outdated` with a reason so it gets demoted in future searches.
12. **Set confidence**: When storing information you're not fully certain about, set `confidence` below 1.0. This affects how prominently the memory appears in search results.


Strict Rule: When fixing issues always make sure you are fixing the root cause and not patching or papering over issues.
