import assert from "node:assert/strict";
import test from "node:test";
import { OrmahClient } from "../src/client.js";
import type { OrmahConfig } from "../src/config.js";

const config: OrmahConfig = {
	baseUrl: "http://isolated-test",
	whisperTimeoutMs: 10,
	toolTimeoutMs: 30_000,
	maintenanceTimeoutMs: 300_000,
	storeTimeoutMs: 120_000,
	whisperNudgeInterval: 10,
	whisperOutMinTurns: 3,
};

test("busy returns immediately without polling or receipt", async (t) => {
	let requests = 0;
	t.mock.method(globalThis, "fetch", async () => {
		requests++;
		return Response.json({ status: "busy", message: "Another run is underway. Stop." });
	});
	const result = await new OrmahClient(config).runMaintenance({});
	assert.equal(result.status, "busy");
	assert.equal(result.job_id, undefined);
	assert.equal(requests, 1);
});

test("results require an explicit receipt, including empty results", async (t) => {
	const fetch = t.mock.method(globalThis, "fetch", async () => {
		throw new Error("must not send a request");
	});
	await assert.rejects(new OrmahClient(config).runMaintenance({ results: {} }), /job_id is required/);
	assert.equal(fetch.mock.callCount(), 0);
});

test("empty results submit with the exact receipt and return its summary", async (t) => {
	t.mock.method(globalThis, "fetch", async (_url: Parameters<typeof fetch>[0], init?: RequestInit) => {
		assert.deepEqual(JSON.parse(String(init?.body)), { job_id: "old-receipt", results: {} });
		return Response.json({ status: "completed", job_id: "old-receipt", apply_summary: { edges: 0 } });
	});
	assert.deepEqual(await new OrmahClient(config).runMaintenance({ jobId: "old-receipt", results: {} }), {
		status: "completed", job_id: "old-receipt", apply_summary: { edges: 0 },
	});
});

for (const terminal of ["expired", "failed", "replaced", "idle"]) {
	for (const submitting of [false, true]) {
		test(`polling stops on ${terminal} during ${submitting ? "application" : "preparation"}`, async (t) => {
			// Advance only the polling delay; HTTP timeout timers retain normal behavior.
			const realTimeout = globalThis.setTimeout;
			t.mock.method(globalThis, "setTimeout", (callback: (...args: unknown[]) => void, ms?: number, ...args: unknown[]) => {
					if (ms === 1000) {
						queueMicrotask(() => callback(...args));
						return undefined;
					}
					return realTimeout(callback, ms, ...args);
				});
			let requests = 0;
			t.mock.method(globalThis, "fetch", async (url: Parameters<typeof fetch>[0]) => {
				requests++;
				if (requests === 1) return Response.json({
					status: submitting ? "running_phase2" : "running_phase1", job_id: "receipt",
				});
				assert.equal(String(url), "http://isolated-test/agent/maintenance?job_id=receipt");
				return Response.json({ status: terminal, job_id: "receipt" });
			});
			await assert.rejects(new OrmahClient(config).runMaintenance(
				submitting ? { jobId: "receipt", results: {} } : {},
			), new RegExp(terminal));
			assert.equal(requests, 2);
		});
	}
}

for (const response of [
	{ status: "completed", job_id: "newer", apply_summary: { edges: 99 } },
	{ status: "completed", job_id: "receipt", apply_summary: null },
	{ status: "completed", job_id: "receipt" },
	{ status: "awaiting_results", job_id: "receipt", batches: {} },
]) {
	test(`does not manufacture success from ${JSON.stringify(response)}`, async (t) => {
		t.mock.method(globalThis, "fetch", async () => Response.json(response));
		await assert.rejects(new OrmahClient(config).runMaintenance({ jobId: "receipt", results: {} }));
	});
}

test("duplicate submission rejection never starts polling", async (t) => {
	const fetch = t.mock.method(globalThis, "fetch", async () =>
		Response.json({ detail: "Maintenance results are already applying. Poll this job_id." }, { status: 409 }));
	await assert.rejects(new OrmahClient(config).runMaintenance({ jobId: "receipt", results: {} }), /already applying/);
	assert.equal(fetch.mock.callCount(), 1);
});

for (const status of ["running_phase2", "completed", "expired", "replaced", "failed"]) {
	test(`receipt-only calls observe ${status} without claiming to submit decisions`, async (t) => {
		const fetch = t.mock.method(globalThis, "fetch", async (_url: Parameters<typeof globalThis.fetch>[0], init?: RequestInit) => {
			assert.deepEqual(JSON.parse(String(init?.body)), { job_id: "receipt" });
			return Response.json({ status, job_id: "receipt" });
		});
		assert.equal((await new OrmahClient(config).runMaintenance({ jobId: "receipt" })).status, status);
		assert.equal(fetch.mock.callCount(), 1);
	});
}
