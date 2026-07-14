/**
 * Normalize Pi session entries into the plain "User: …\n\nAssistant: …" form
 * Ormah's /ingest/conversation expects.
 *
 * Ormah's transcript parser (src/ormah/transcript/parser.py) only handles
 * Claude Code / Codex JSONL; Pi sessions use a different AgentMessage shape and
 * live under ~/.pi/agent/sessions. Rather than teach Ormah a third format, we
 * normalize here and feed the text straight to /ingest/conversation.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

interface TextBlock {
	type: "text" | "output_text";
	text: string;
}

function isTextBlock(b: unknown): b is TextBlock {
	return (
		typeof b === "object" &&
		b !== null &&
		((b as TextBlock).type === "text" ||
			(b as TextBlock).type === "output_text") &&
		typeof (b as TextBlock).text === "string"
	);
}

export function extractText(content: unknown): string | null {
	if (typeof content === "string") return content.trim() || null;
	if (!Array.isArray(content)) return null;
	const texts: string[] = [];
	for (const block of content) {
		if (isTextBlock(block) && block.text?.trim()) texts.push(block.text.trim());
	}
	return texts.length ? texts.join("\n") : null;
}

export interface NormalizedTranscript {
	conversation: string;
	userTurnCount: number;
	turns: string[];
}

/**
 * Walk the current session branch and produce cleaned conversation text.
 * Skips toolResult, system, tool_use, thinking — only user + assistant text,
 * matching Ormah's parser semantics.
 */
export function normalizeSession(ctx: ExtensionContext): NormalizedTranscript {
	const entries = ctx.sessionManager.getBranch();
	const turns: string[] = [];
	let userTurnCount = 0;

	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const msg = (entry as { message?: { role?: string; content?: unknown } })
			.message;
		if (!msg) continue;
		const role = msg.role;
		if (role !== "user" && role !== "assistant") continue;
		const text = extractText(msg.content);
		if (!text) continue;
		if (role === "user") {
			turns.push(`User: ${text}`);
			userTurnCount++;
		} else {
			turns.push(`Assistant: ${text}`);
		}
	}

	return { conversation: turns.join("\n\n"), userTurnCount, turns };
}
