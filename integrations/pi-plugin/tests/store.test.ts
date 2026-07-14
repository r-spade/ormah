import assert from "node:assert/strict";
import test from "node:test";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { IngestBody, OrmahClient } from "../src/client.js";
import { StoreManager } from "../src/store.js";

function message(role: "user" | "assistant", content: string) {
	return { type: "message", message: { role, content } };
}

test("store ingests only new turns after a compaction store", async () => {
	const calls: IngestBody[] = [];
	const client = {
		ingestConversation: async (body: IngestBody) => {
			calls.push(body);
			return { status: "ok", extracted: 1, memories: [] };
		},
	} as unknown as OrmahClient;
	const branch = [
		message("user", "Question one"),
		message("assistant", "Answer one"),
		message("user", "Question two"),
		message("assistant", "Answer two"),
	];
	const ctx = {
		cwd: "/tmp/project",
		sessionManager: {
			getSessionFile: () => "/tmp/session.jsonl",
			getBranch: () => branch,
		},
		ui: { notify: () => undefined, setStatus: () => undefined },
	} as unknown as ExtensionContext;
	const store = new StoreManager(client, 2);

	await store.store(ctx, "compact");
	branch.push(
		message("user", "Question after compact"),
		message("assistant", "Answer after compact"),
	);
	await store.store(ctx, "shutdown");
	await store.store(ctx, "shutdown");

	assert.equal(calls.length, 2);
	assert.match(calls[0].content, /Question one/);
	assert.equal(
		calls[1].content,
		"User: Question after compact\n\nAssistant: Answer after compact",
	);
});

test("store accepts a post-compaction branch with fewer than the initial minimum", async () => {
	const calls: IngestBody[] = [];
	const client = {
		ingestConversation: async (body: IngestBody) => {
			calls.push(body);
			return { status: "ok", extracted: 1, memories: [] };
		},
	} as unknown as OrmahClient;
	let branch = [
		message("user", "Question one"),
		message("assistant", "Answer one"),
		message("user", "Question two"),
		message("assistant", "Answer two"),
	];
	const ctx = {
		cwd: "/tmp/project",
		sessionManager: {
			getSessionFile: () => "/tmp/session.jsonl",
			getBranch: () => branch,
		},
		ui: { notify: () => undefined, setStatus: () => undefined },
	} as unknown as ExtensionContext;
	const store = new StoreManager(client, 2);

	await store.store(ctx, "compact");
	branch = [
		message("user", "Only new question"),
		message("assistant", "Only new answer"),
	];
	await store.store(ctx, "shutdown");

	assert.equal(calls.length, 2);
	assert.equal(
		calls[1].content,
		"User: Only new question\n\nAssistant: Only new answer",
	);
});
