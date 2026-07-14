import assert from "node:assert/strict";
import test from "node:test";
import type {
	ExtensionAPI,
	ExtensionCommandContext,
} from "@earendil-works/pi-coding-agent";
import type { OrmahClient } from "../src/client.js";
import { registerCommands } from "../src/commands.js";

interface RegisteredCommand {
	handler: (args: string, ctx: ExtensionCommandContext) => Promise<void>;
}

function commandHarness(
	results: Array<{ code: number; stdout: string; stderr: string }>,
) {
	const commands = new Map<string, RegisteredCommand>();
	const execCalls: Array<{ command: string; args: string[] }> = [];
	const notifications: Array<{ message: string; level: string }> = [];
	const pi = {
		registerCommand: (name: string, command: RegisteredCommand) => {
			commands.set(name, command);
		},
		exec: async (command: string, args: string[]) => {
			execCalls.push({ command, args });
			const result = results.shift();
			if (!result) throw new Error("unexpected exec");
			return result;
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		ui: {
			confirm: async () => true,
			setStatus: () => undefined,
			notify: (message: string, level: string) => {
				notifications.push({ message, level });
			},
		},
	} as unknown as ExtensionCommandContext;
	registerCommands(pi, {} as OrmahClient);
	return { commands, ctx, execCalls, notifications };
}

test("setup installs canonical Pi guidance after Ormah setup succeeds", async () => {
	const harness = commandHarness([
		{ code: 0, stdout: "setup complete", stderr: "" },
		{ code: 0, stdout: "guidance installed", stderr: "" },
	]);
	const setup = harness.commands.get("ormah:setup");
	assert.ok(setup);

	await setup.handler("", harness.ctx);

	assert.deepEqual(harness.execCalls, [
		{
			command: "ormah",
			args: ["setup", "--skip-client-setup"],
		},
		{
			command: "ormah",
			args: ["pi-md", "install", "--scope", "user"],
		},
	]);
});

test("setup failure does not overwrite guidance", async () => {
	const harness = commandHarness([
		{ code: 1, stdout: "", stderr: "server failed" },
	]);
	const setup = harness.commands.get("ormah:setup");
	assert.ok(setup);

	await setup.handler("", harness.ctx);

	assert.equal(harness.execCalls.length, 1);
	assert.match(harness.notifications.at(-1)?.message ?? "", /setup failed/);
});

test("upgrade failure does not stop the current server", async () => {
	const harness = commandHarness([
		{ code: 1, stdout: "", stderr: "registry unavailable" },
	]);
	const upgrade = harness.commands.get("ormah:upgrade");
	assert.ok(upgrade);

	await upgrade.handler("", harness.ctx);

	assert.deepEqual(harness.execCalls, [
		{
			command: "uv",
			args: ["tool", "upgrade", "ormah"],
		},
	]);
});
