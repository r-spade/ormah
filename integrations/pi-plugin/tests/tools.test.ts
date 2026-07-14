import assert from "node:assert/strict";
import test from "node:test";
import type {
	ExtensionAPI,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import type { MaintenanceResults, OrmahClient } from "../src/client.js";
import { registerTools } from "../src/tools.js";

interface MaintenanceCall {
	jobId?: string;
	results?: MaintenanceResults;
}

interface RegisteredTool {
	execute: (
		id: string,
		params: { results?: MaintenanceResults },
		signal: AbortSignal | undefined,
		onUpdate: undefined,
		ctx: ExtensionContext,
	) => Promise<unknown>;
}

function context(sessionFile: string): ExtensionContext {
	return {
		cwd: "/tmp/ormah",
		sessionManager: { getSessionFile: () => sessionFile },
	} as unknown as ExtensionContext;
}

test("maintenance phase two reuses the phase one job id per session", async () => {
	const tools = new Map<string, RegisteredTool>();
	const pi = {
		registerTool: (tool: RegisteredTool & { name: string }) => {
			tools.set(tool.name, tool);
		},
	} as unknown as ExtensionAPI;
	const calls: MaintenanceCall[] = [];
	let nextJob = 1;
	const client = {
		runMaintenance: async (options: MaintenanceCall) => {
			calls.push(options);
			if (options.results) {
				return { status: "completed", job_id: options.jobId };
			}
			return {
				status: "awaiting_results",
				job_id: `maintenance-job-${nextJob++}`,
				batches: {},
			};
		},
	} as unknown as OrmahClient;
	registerTools(pi, { client });
	const maintenance = tools.get("ormah_run_maintenance");
	assert.ok(maintenance);
	const sessionA = context("/tmp/session-a.jsonl");
	const sessionB = context("/tmp/session-b.jsonl");

	await maintenance.execute(
		"phase-1-a",
		{},
		new AbortController().signal,
		undefined,
		sessionA,
	);
	await maintenance.execute(
		"phase-1-b",
		{},
		new AbortController().signal,
		undefined,
		sessionB,
	);
	await maintenance.execute(
		"phase-2-a",
		{ results: { edges: [] } },
		new AbortController().signal,
		undefined,
		sessionA,
	);

	assert.deepEqual(calls, [
		{ jobId: undefined, results: undefined },
		{ jobId: undefined, results: undefined },
		{
			jobId: "maintenance-job-1",
			results: { edges: [] },
		},
	]);
});
