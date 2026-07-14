import assert from "node:assert/strict";
import test from "node:test";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { beginSession, getSessionId } from "../src/session-id.js";

function context(sessionFile?: string): ExtensionContext {
	return {
		sessionManager: { getSessionFile: () => sessionFile },
	} as unknown as ExtensionContext;
}

test("file-backed sessions use the session file as their stable identity", () => {
	const ctx = context("/tmp/session.jsonl");

	assert.equal(beginSession(ctx), "/tmp/session.jsonl");
	assert.equal(getSessionId(ctx), "/tmp/session.jsonl");
});

test("in-memory sessions receive stable, resettable, process-unique identities", () => {
	const first = context();
	const second = context();

	const firstId = beginSession(first);
	assert.equal(getSessionId(first), firstId);
	assert.notEqual(getSessionId(second), firstId);

	const nextFirstId = beginSession(first);
	assert.notEqual(nextFirstId, firstId);
	assert.equal(getSessionId(first), nextFirstId);
});
