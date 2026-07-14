/**
 * Whisper store — involuntary memory extraction at session end / compaction.
 *
 * Mirrors the Claude `PreCompact` + `SessionEnd` hooks (both call
 * `ormah whisper store`). Pi sessions don't live where Ormah's transcript
 * resolver looks, so we normalize entries ourselves and POST /ingest/conversation.
 *
 * Fail-silent: store must never break shutdown or compaction.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { OrmahClient } from "./client.js";
import { getSessionId } from "./session-id.js";
import { resolveSpace } from "./space.js";
import { normalizeSession } from "./session.js";

export class StoreManager {
	/** Last successfully stored transcript state per session. */
	private readonly storedTurns = new Map<string, string[]>();

	constructor(
		private readonly client: OrmahClient,
		private readonly minTurns: number,
	) {}

	async store(
		ctx: ExtensionContext,
		reason: "compact" | "shutdown",
	): Promise<void> {
		const sessionFile = getSessionId(ctx);

		const { turns, userTurnCount } = normalizeSession(ctx);
		const previouslyStored = this.storedTurns.get(sessionFile);
		if (!previouslyStored && userTurnCount < this.minTurns) return;

		const extendsStoredTranscript =
			previouslyStored !== undefined &&
			previouslyStored.every((turn, index) => turns[index] === turn);
		const pendingTurns = extendsStoredTranscript
			? turns.slice(previouslyStored.length)
			: turns;
		const conversation = pendingTurns.join("\n\n");
		if (!conversation.trim()) return;

		const space = resolveSpace(undefined, ctx.cwd);
		try {
			const result = await this.client.ingestConversation({
				content: conversation,
				agent_id: "pi",
				space,
				extra_tags: ["pi-session"],
			});
			if (result.status !== "error") {
				this.storedTurns.set(sessionFile, [...turns]);
			}
			ctx.ui.notify?.(
				`Ormah: stored ${result.extracted ?? 0} memor${(result.extracted ?? 0) === 1 ? "y" : "ies"} from session (${reason})`,
				"info",
			);
		} catch (err) {
			// Silent — store is best-effort. Surface only in status, never block.
			ctx.ui.setStatus?.(
				"ormah-store",
				`store failed: ${(err as Error).message.slice(0, 60)}`,
			);
		}
	}
}
