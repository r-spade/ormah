import assert from "node:assert/strict";
import test from "node:test";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { extractText, normalizeSession } from "../src/session.js";

test("extractText ignores null and non-text blocks without throwing", () => {
	assert.equal(
		extractText([
			null,
			undefined,
			{ type: "thinking", text: "hidden" },
			{ type: "text", text: " visible " },
			{ type: "output_text", text: " output " },
		]),
		"visible\noutput",
	);
});

test("normalizeSession returns plain turns and user count", () => {
	const ctx = {
		sessionManager: {
			getBranch: () => [
				{
					type: "message",
					message: { role: "user", content: "First question" },
				},
				{
					type: "message",
					message: {
						role: "assistant",
						content: [{ type: "text", text: "First answer" }],
					},
				},
				{
					type: "message",
					message: { role: "toolResult", content: "ignored" },
				},
				{
					type: "message",
					message: { role: "user", content: "Second question" },
				},
			],
		},
	} as unknown as ExtensionContext;

	const transcript = normalizeSession(ctx);

	assert.deepEqual(transcript.turns, [
		"User: First question",
		"Assistant: First answer",
		"User: Second question",
	]);
	assert.equal(transcript.userTurnCount, 2);
	assert.equal(transcript.conversation, transcript.turns.join("\n\n"));
});
