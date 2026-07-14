/**
 * Ormah-Pi runtime configuration.
 *
 * Reads ORMAH_* env vars (and ~/.config/ormah/.env by convention) the same way
 * the Ormah server does, so the extension and server stay in sync without a
 * second source of truth. Only the handful of settings the extension actually
 * consumes are modeled here — everything else lives server-side.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

type Environment = Record<string, string | undefined>;

function readEnvFile(path: string): Environment {
	let content: string;
	try {
		content = readFileSync(path, "utf8");
	} catch {
		return {};
	}

	const values: Environment = {};
	for (const line of content.split(/\r?\n/)) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const assignment = trimmed.startsWith("export ")
			? trimmed.slice("export ".length)
			: trimmed;
		const separator = assignment.indexOf("=");
		if (separator <= 0) continue;
		const key = assignment.slice(0, separator).trim();
		let value = assignment.slice(separator + 1).trim();
		if (
			value.length >= 2 &&
			((value.startsWith('"') && value.endsWith('"')) ||
				(value.startsWith("'") && value.endsWith("'")))
		) {
			value = value.slice(1, -1);
		} else {
			value = value.replace(/\s+#.*$/, "").trimEnd();
		}
		values[key] = value;
	}
	return values;
}

function envInt(env: Environment, name: string, fallback: number): number {
	const raw = env[name];
	if (!raw) return fallback;
	const n = Number(raw);
	return Number.isFinite(n) ? n : fallback;
}

function envStr(env: Environment, name: string, fallback: string): string {
	const raw = env[name];
	return raw && raw.length ? raw : fallback;
}

export interface OrmahConfig {
	/** Base URL of the local Ormah HTTP server. */
	baseUrl: string;
	/** Whisper inject HTTP timeout (ms). */
	whisperTimeoutMs: number;
	/** Per-tool HTTP timeout (ms). run_maintenance uses maintenanceTimeoutMs. */
	toolTimeoutMs: number;
	/** run_maintenance HTTP timeout (ms) — extraction/application can be slow. */
	maintenanceTimeoutMs: number;
	/** Whisper store (/ingest) timeout (ms) — LLM extraction can take 30s+. */
	storeTimeoutMs: number;
	/** Nudge the agent to use `remember` every N user prompts (0 = off). Mirrors ORMAH_WHISPER_NUDGE_INTERVAL. */
	whisperNudgeInterval: number;
	/** Minimum user turns before a session is worth storing. Mirrors ORMAH_WHISPER_OUT_MIN_TURNS. */
	whisperOutMinTurns: number;
}

export function loadConfig(
	processEnv: Environment = process.env,
	envPath = join(homedir(), ".config", "ormah", ".env"),
): OrmahConfig {
	const env = readEnvFile(envPath);
	for (const [key, value] of Object.entries(processEnv)) {
		if (value !== undefined) env[key] = value;
	}
	const host = envStr(env, "ORMAH_HOST", "127.0.0.1");
	const port = envInt(env, "ORMAH_PORT", 8787);
	// ORMAH_BASE_URL wins if explicitly set.
	const baseUrl = envStr(env, "ORMAH_BASE_URL", `http://${host}:${port}`);
	return {
		baseUrl,
		whisperTimeoutMs: envInt(env, "ORMAH_PI_WHISPER_TIMEOUT_MS", 12_000),
		toolTimeoutMs: envInt(env, "ORMAH_PI_TOOL_TIMEOUT_MS", 30_000),
		maintenanceTimeoutMs: envInt(
			env,
			"ORMAH_PI_MAINTENANCE_TIMEOUT_MS",
			300_000,
		),
		storeTimeoutMs: envInt(env, "ORMAH_PI_STORE_TIMEOUT_MS", 120_000),
		whisperNudgeInterval: envInt(env, "ORMAH_WHISPER_NUDGE_INTERVAL", 10),
		whisperOutMinTurns: envInt(env, "ORMAH_WHISPER_OUT_MIN_TURNS", 3),
	};
}
