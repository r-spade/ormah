/**
 * Ormah-Pi — local persistent memory for the Pi coding agent.
 *
 * Pi port of r-spade/ormah's Claude Code integration (integrations/claude-plugin/).
 * Wires Pi into the local Ormah server:
 *   - whisper:   involuntary recall injected before each prompt (before_agent_start)
 *   - tools:     remember / recall / recall_node / mark_outdated / submit_feedback /
 *                run_maintenance, proxied to the Ormah HTTP API
 *   - store:     transcript → memory extraction on compact / session end
 *   - commands:  /ormah:setup /ormah:status /ormah:maintenance /ormah:upgrade /ormah:reload
 *
 * Transport is HTTP fetch to the Ormah server (default http://127.0.0.1:8787),
 * the same surface Ormah's own MCP adapter uses — no `ormah mcp` subprocess needed.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { loadConfig } from "./src/config.js";
import { OrmahClient } from "./src/client.js";
import { beginSession } from "./src/session-id.js";
import { WhisperManager } from "./src/whisper.js";
import { StoreManager } from "./src/store.js";
import { registerTools } from "./src/tools.js";
import { registerCommands } from "./src/commands.js";

export default function ormahPi(pi: ExtensionAPI) {
	const cfg = loadConfig();
	const client = new OrmahClient(cfg);

	const whisper = new WhisperManager(client, cfg.whisperNudgeInterval);
	const store = new StoreManager(client, cfg.whisperOutMinTurns);

	// ── session lifecycle ─────────────────────────────────────────────────────
	pi.on("session_start", async (_event, ctx) => {
		beginSession(ctx);
		try {
			const stats = await client.health();
			ctx.ui.setStatus("ormah", `connected · ${stats.total_nodes} mem`);
		} catch {
			ctx.ui.setStatus("ormah", "down");
		}
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		try {
			await store.store(ctx, "shutdown");
		} catch {
			/* fail-silent */
		}
		ctx.ui.setStatus("ormah", undefined);
	});

	// ── whisper: involuntary recall before each prompt ─────────────────────────
	pi.on("before_agent_start", async (event, ctx) => {
		return whisper.onBeforeAgentStart(
			event as { prompt: string; images?: unknown[]; systemPrompt: string },
			ctx,
		);
	});

	// ── store: memory extraction on compaction (PreCompact equivalent) ─────────
	pi.on("session_before_compact", async (_event, ctx) => {
		try {
			await store.store(ctx, "compact");
		} catch {
			/* fail-silent */
		}
	});

	// ── tools + commands ──────────────────────────────────────────────────────
	registerTools(pi, { client });
	registerCommands(pi, client);
}
