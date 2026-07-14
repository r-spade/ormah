/**
 * Slash commands — the Pi analogue of integrations/claude-plugin/commands/.
 *
 *   /ormah:setup       Install/repair the Ormah runtime + wire the Pi extension
 *   /ormah:status      Server health + memory stats
 *   /ormah:maintenance Run the maintenance agent prompt in-session
 *   /ormah:upgrade     Upgrade the Ormah runtime
 *   /ormah:reload      Reload Pi extensions/skills/prompts/themes
 */

import type {
	ExtensionAPI,
	ExtensionCommandContext,
} from "@earendil-works/pi-coding-agent";
import type { OrmahClient } from "./client.js";
import { loadMaintenancePrompt } from "./maintenance.js";

export function registerCommands(pi: ExtensionAPI, client: OrmahClient): void {
	pi.registerCommand("ormah:setup", {
		description: "Install/repair the Ormah runtime and wire this Pi extension",
		handler: async (_args, ctx: ExtensionCommandContext) => {
			if (!ctx.hasUI) return;
			const ok = await ctx.ui.confirm(
				"Ormah setup",
				"Run `ormah setup --skip-client-setup` (starts server + preloads models + autostart) and install the Pi guidance block into ~/.pi/agent/AGENTS.md?",
			);
			if (!ok) return;
			ctx.ui.setStatus("ormah", "running ormah setup…");
			const result = await pi.exec("ormah", ["setup", "--skip-client-setup"], {
				timeout: 300_000,
			});
			if (result.code !== 0) {
				ctx.ui.setStatus("ormah", undefined);
				const detail = (result.stderr || result.stdout || "unknown error")
					.trim()
					.slice(-240);
				ctx.ui.notify(`Ormah setup failed: ${detail}`, "warning");
				return;
			}
			const guidance = await pi.exec(
				"ormah",
				["pi-md", "install", "--scope", "user"],
				{ timeout: 30_000 },
			);
			ctx.ui.setStatus("ormah", undefined);
			const tail = (result.stdout || result.stderr || "")
				.split("\n")
				.filter(Boolean)
				.slice(-3)
				.join(" | ");
			ctx.ui.notify(
				`Ormah setup done. ${tail}`,
				"info",
			);
			if (guidance.code === 0) {
				ctx.ui.notify(
					"Guidance installed in ~/.pi/agent/AGENTS.md. Run /reload to activate.",
					"info",
				);
			} else {
				ctx.ui.notify(
					`Ormah setup succeeded, but guidance installation failed: ${(guidance.stderr || guidance.stdout || "unknown error").trim().slice(-160)}`,
					"warning",
				);
			}
		},
	});

	pi.registerCommand("ormah:status", {
		description: "Show Ormah server health and memory stats",
		handler: async (_args, ctx: ExtensionCommandContext) => {
			try {
				const stats = await client.health();
				const byTier = Object.entries(stats.by_tier)
					.map(([k, v]) => `${k}:${v}`)
					.join(" ");
				ctx.ui.notify(
					`Ormah: ${stats.total_nodes} memories, ${stats.total_edges} edges (${byTier})`,
					"info",
				);
				ctx.ui.setStatus("ormah", `connected · ${stats.total_nodes} mem`);
			} catch (e) {
				ctx.ui.notify(
					`Ormah: server not reachable — ${(e as Error).message.slice(0, 80)}`,
					"warning",
				);
				ctx.ui.setStatus("ormah", "down");
			}
		},
	});

	pi.registerCommand("ormah:maintenance", {
		description: "Run the Ormah memory maintenance flow in this session",
		handler: async (_args, _ctx: ExtensionCommandContext) => {
			const prompt = await loadMaintenancePrompt();
			// Runs maintenance in-session; an isolated background subagent is the upgrade path.
			pi.sendUserMessage(prompt);
		},
	});

	pi.registerCommand("ormah:upgrade", {
		description:
			"Upgrade the Ormah runtime (uv tool upgrade) and restart the server",
		handler: async (_args, ctx: ExtensionCommandContext) => {
			if (!ctx.hasUI) return;
			const ok = await ctx.ui.confirm(
				"Ormah upgrade",
				"Run `uv tool upgrade ormah` then restart the server?",
			);
			if (!ok) return;
			ctx.ui.setStatus("ormah", "upgrading…");
			const up = await pi.exec("uv", ["tool", "upgrade", "ormah"], {
				timeout: 300_000,
			});
			if (up.code !== 0) {
				ctx.ui.setStatus("ormah", undefined);
				ctx.ui.notify(
					`Ormah upgrade failed: ${(up.stderr || up.stdout || "unknown error").trim().slice(-200)}`,
					"warning",
				);
				return;
			}
			await pi.exec("ormah", ["server", "stop"], { timeout: 15_000 });
			await pi.exec("ormah", ["server", "start", "-d"], { timeout: 30_000 });
			ctx.ui.setStatus("ormah", undefined);
			ctx.ui.notify(
				"Ormah upgrade done.",
				"info",
			);
		},
	});

	pi.registerCommand("ormah:reload", {
		description: "Reload Pi extensions, skills, prompts, and themes",
		handler: async (_args, ctx: ExtensionCommandContext) => {
			await ctx.reload();
		},
	});
}
