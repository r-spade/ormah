import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { loadConfig } from "../src/config.js";

test("loadConfig reads Ormah's env file", () => {
	const dir = mkdtempSync(join(tmpdir(), "ormah-pi-config-"));
	const envPath = join(dir, ".env");
	writeFileSync(
		envPath,
		[
			"ORMAH_HOST=0.0.0.0",
			"ORMAH_PORT=9123",
			"ORMAH_PI_TOOL_TIMEOUT_MS=45000",
		].join("\n"),
	);

	const config = loadConfig({}, envPath);

	assert.equal(config.baseUrl, "http://0.0.0.0:9123");
	assert.equal(config.toolTimeoutMs, 45_000);
});

test("process environment overrides values from the env file", () => {
	const dir = mkdtempSync(join(tmpdir(), "ormah-pi-config-"));
	const envPath = join(dir, ".env");
	writeFileSync(envPath, "ORMAH_PORT=9123\nORMAH_BASE_URL=http://file:9123\n");

	const config = loadConfig(
		{
			ORMAH_PORT: "9444",
			ORMAH_BASE_URL: "http://process:9444",
		},
		envPath,
	);

	assert.equal(config.baseUrl, "http://process:9444");
});

test("invalid numeric values use defaults", () => {
	const config = loadConfig(
		{ ORMAH_PI_WHISPER_TIMEOUT_MS: "not-a-number" },
		"/missing/ormah-pi.env",
	);

	assert.equal(config.whisperTimeoutMs, 12_000);
});

test("unquoted inline comments are ignored without stripping URL fragments", () => {
	const dir = mkdtempSync(join(tmpdir(), "ormah-pi-config-"));
	const envPath = join(dir, ".env");
	writeFileSync(
		envPath,
		[
			"ORMAH_PORT=9123 # local test server",
			"ORMAH_BASE_URL=http://host/path#fragment",
		].join("\n"),
	);

	const config = loadConfig({}, envPath);

	assert.equal(config.baseUrl, "http://host/path#fragment");
	assert.equal(config.toolTimeoutMs, 30_000);
});
