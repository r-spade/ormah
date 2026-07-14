import assert from "node:assert/strict";
import test from "node:test";
import { OrmahClient } from "../src/client.js";
import type { OrmahConfig } from "../src/config.js";

const config: OrmahConfig = {
	baseUrl: "http://127.0.0.1:8787",
	whisperTimeoutMs: 10,
	toolTimeoutMs: 30_000,
	maintenanceTimeoutMs: 300_000,
	storeTimeoutMs: 120_000,
	whisperNudgeInterval: 10,
	whisperOutMinTurns: 3,
};

test("remember preserves explicit global scope instead of adding a default space", async () => {
	const originalFetch = globalThis.fetch;
	let requestUrl = "";
	let requestBody = "";
	globalThis.fetch = (async (input, init) => {
		requestUrl = String(input);
		requestBody = String(init?.body);
		return new Response(JSON.stringify({ text: "stored" }), { status: 200 });
	}) as typeof fetch;

	try {
		const client = new OrmahClient(config);
		await client.remember({ content: "global", space: null }, "project-space");
	} finally {
		globalThis.fetch = originalFetch;
	}

	assert.equal(requestUrl, "http://127.0.0.1:8787/agent/remember");
	assert.equal(JSON.parse(requestBody).space, null);
});

test("request timeout still applies when Pi supplies a cancellation signal", async () => {
	const originalFetch = globalThis.fetch;
	globalThis.fetch = ((_input, init) => {
		return new Promise((_resolve, reject) => {
			init?.signal?.addEventListener(
				"abort",
				() => reject(new Error("request aborted")),
				{ once: true },
			);
		});
	}) as typeof fetch;

	try {
		const client = new OrmahClient(config);
		const caller = new AbortController();
		await assert.rejects(
			client.whisper({ prompt: "test" }, caller.signal),
			/request aborted/,
		);
	} finally {
		globalThis.fetch = originalFetch;
	}
});
