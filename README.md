# Ormah

Ormah is the collective, self-maintaining memory layer all your agents can tap into.

The core idea is simple: memory should be involuntary. Your agents should not have to remember to remember. Ormah works in the background, learning your preferences, decisions, patterns, mistakes, and ongoing work, then whispering the right memory at the right time.

Your memory has always been yours. Ormah helps keep it that way.

Local. Private. Portable.
Yours to keep. Yours to move.

<p align="center">
  <img src="docs/graph.png" alt="Ormah knowledge graph" width="100%">
</p>

The name comes from the Malayalam word ഓർമ (`ormah`), meaning "memory" or "remember."

## Memory Should Whisper

In real life, memory does not work like search. When something in front of you connects to something you already know, the memory surfaces on its own. You do not stop and decide to remember.

Ormah is built around that idea. Instead of waiting for an agent to ask for context, Ormah looks at what is happening and whispers the right memory before the agent processes the next prompt, so it starts with the context, preferences, constraints, and hints that matter.

That is what makes Ormah feel like memory instead of search. Search waits to be asked. Memory shows up when it matters.

Silence is better than noise. Ormah should whisper, not shout.

## Install

### Terminal

```bash
bash <(curl -fsSL https://ormah.me/install.sh)
```

One command gets you a working local Ormah runtime with setup for supported clients.

### Claude Code Plugin

1. Add the marketplace and install the plugin:
   ```
   /plugin marketplace add r-spade/ormah
   /plugin install ormah@ormah
   ```
2. Reload: `/reload-plugins`
3. Run `/ormah:setup`
4. Check that the Ormah MCP server is enabled via `/mcp` — if not, enable it there

Ormah is agent-agnostic by design. It can be wired into any agent that exposes the right hook or prompt-injection path, and it also exposes CLI, MCP, and HTTP surfaces.

`ormah setup` will:

1. Start the Ormah server and install auto-start
2. Preload the local models used for search and whisper retrieval
3. Detect supported clients and wire them up automatically
4. Offer agent-backed maintenance when Claude Code or Codex are available
5. Offer transcript backfill to help bootstrap memory from earlier sessions

Today, setup can wire up:

- Claude Code
- Codex
- Claude Desktop (MCP)

Local search, embeddings, storage, the graph UI, and whisper retrieval do not require an API key. If you want Ormah's LLM-backed features to run independently of your agent, setup can opt into a provider explicitly. Ormah stores only provider policy in `~/.config/ormah/.env`; it does not copy API key values into its config. Local Ollama generation uses a configurable `ORMAH_LLM_NUM_PREDICT` token budget.

## Features

### Recall and Whisper

Ormah supports both deliberate recall and involuntary recall.

When an agent knows it needs something, it can explicitly search memory. But memory should not always wait to be asked. Ormah is built to whisper the right memory at the right time, before the next prompt, so the agent starts with context instead of having to go looking for it.

Read more: [Whisper - Involuntary Recall](<docs/04 - Whisper - Involuntary Recall.md>), [Search and Ranking](<docs/03 - Search and Ranking.md>), [Affinity and Feedback](<docs/09 - Affinity and Feedback.md>)

### Memory Capture

Memory is only useful if it keeps growing with you.

Ormah can capture memory from ongoing sessions, stored transcripts, and external markdown sources. `whisper store` turns conversations into durable memory, the session watcher ingests completed sessions automatically, and Hippocampus watches note directories so project docs, journals, and markdown knowledge can flow into the graph over time.

Read more: [Hippocampus and Session Watcher](<docs/10 - Hippocampus and Session Watcher.md>), [Storage Layer](<docs/02 - Storage Layer.md>)

### Self-Maintaining Memory

Memory should not become a junk drawer.

Ormah continuously maintains the graph in the background: linking related memories, detecting conflicts, tracking belief evolution, merging duplicates, consolidating overlap, scoring importance, and decaying stale context. Some of that work is automatic, and some of it can be delegated to an agent when judgment is required.

Read more: [Background Jobs](<docs/05 - Background Jobs.md>)

### Local Backups

Memories should be hard to lose.

Ormah can create timestamped local backups of the source-of-truth memory files and keeps the latest 10 by default. Backups include active and deleted memory nodes, but exclude derived indexes, logs, config, and API keys. When memory nodes exist, the server checks for a due backup in the background. Manual workflows are available through `ormah backup create`, `ormah backup list`, `ormah backup status`, and `ormah backup restore`.

Read more: [Configuration Reference](<docs/12 - Configuration Reference.md>)

### Agent-Agnostic Surfaces

Ormah is not tied to a single agent.

It can integrate wherever there is a usable hook or interface. Ormah exposes multiple surfaces for that: hooks for whisper, MCP for tool-calling agents, a CLI for direct workflows, and an HTTP API for custom integrations. The memory layer stays the same even when the agent changes.

Read more: [MCP and Adapters](<docs/07 - MCP and Adapters.md>), [API Surface](<docs/08 - API Surface.md>), [Setup and Installation](<docs/11 - Setup and Installation.md>)

### Agent-Assisted or Independent

Ormah can use the intelligence of the agents you already work with, like Codex or Claude Code, for judgment-heavy tasks such as maintenance. But it does not have to depend on them. If you want Ormah to run those features independently, you can configure your own provider and API key.

Read more: [Configuration Reference](<docs/12 - Configuration Reference.md>), [Setup and Installation](<docs/11 - Setup and Installation.md>)

### Graph UI

Memory should be inspectable.

Ormah includes a graph UI so you can see what it knows, how memories connect, what is becoming central, and where conflicts or belief changes are forming. That makes the system easier to trust, debug, and improve.

The graph UI includes dark/light themes plus browser-persisted controls for tier colors.

Read more: [Web UI](<docs/14 - Web UI.md>)

## Integrations

Ormah is agent-agnostic, but it already has first-class integrations for:

- Claude Code — whisper hooks, MCP, transcript backfill, maintenance agent
- Codex — whisper hooks, MCP, maintenance agent
- Claude Desktop (macOS) — MCP

It can also be used through:

- MCP for compatible clients
- the CLI for terminal workflows
- the HTTP API for local apps and custom integrations
- OpenAI-compatible tool schemas for custom tool-calling stacks

Main integration surfaces:

- Hooks — whisper before the next prompt
- MCP — `remember`, `recall`, `recall_node`, `mark_outdated`, `submit_feedback`, `run_maintenance`
- CLI — setup, server management, memory ops, ingestion, whisper hooks, evals
- HTTP API — `/agent/*`, `/admin/*`, `/ingest/*`, `/ui/*`

Read more: [Setup and Installation](<docs/11 - Setup and Installation.md>), [MCP and Adapters](<docs/07 - MCP and Adapters.md>), [API Surface](<docs/08 - API Surface.md>), [Configuration Reference](<docs/12 - Configuration Reference.md>)

## Development

```bash
git clone https://github.com/r-spade/ormah.git
cd ormah
make install
uv run pytest
```

## License

MIT
