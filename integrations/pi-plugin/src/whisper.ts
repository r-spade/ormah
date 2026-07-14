/**
 * Whisper — involuntary pre-prompt recall injection.
 *
 * Pi equivalent of Claude Code's `UserPromptSubmit` hook → `ormah whisper inject`.
 * Fires on `before_agent_start`, asks the Ormah server for relevant context,
 * and returns a display:none custom message so the LLM starts the turn with
 * memory without polluting the user's system prompt.
 *
 * Also ports the per-session prompt counter from cmd_whisper_inject to nudge
 * the agent to use `remember` every N prompts. Maintenance signals are returned
 * by the Ormah server as part of the whisper response.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { OrmahClient } from "./client.js";
import { getSessionId } from "./session-id.js";
import { resolveSpace } from "./space.js";

const NUDGE =
	"\n\n---\n" +
	"Remember to use ormah's remember tool to store decisions, " +
	"preferences or noteworthy facts from this conversation.";

interface BeforeAgentStartEvent {
	prompt: string;
	images?: unknown[];
	systemPrompt: string;
}

export class WhisperManager {
	/** Per-session prompt counter (in-memory; reset per extension instance). */
	private readonly promptCount = new Map<string, number>();

	constructor(
		private readonly client: OrmahClient,
		private readonly nudgeInterval: number,
	) {}

	async onBeforeAgentStart(
		event: BeforeAgentStartEvent,
		ctx: ExtensionContext,
	): Promise<
		| { message: { customType: string; content: string; display: boolean } }
		| undefined
	> {
		const prompt = event.prompt ?? "";
		if (!prompt.trim()) return undefined;

		const sessionId = getSessionId(ctx);
		const space = resolveSpace(undefined, ctx.cwd);

		let text: string;
		try {
			const resp = await this.client.whisper(
				{ prompt, space, session_id: sessionId },
				ctx.signal,
			);
			text = resp.text ?? "";
		} catch {
			// Server down / timeout / any error — silence beats noise.
			return undefined;
		}

		const count = (this.promptCount.get(sessionId) ?? 0) + 1;
		this.promptCount.set(sessionId, count);

		// Periodic nudge to use the remember tool.
		if (this.nudgeInterval > 0 && count % this.nudgeInterval === 0) {
			text += NUDGE;
		}

		if (!text.trim()) return undefined;

		return {
			message: {
				customType: "ormah-whisper",
				content: text,
				display: false,
			},
		};
	}
}
