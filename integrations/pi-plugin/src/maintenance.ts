/**
 * Maintenance trigger.
 *
 * Mirrors upstream Claude's `/ormah-maintenance` slash command: loads the
 * maintenance agent prompt (agents/ormah-maintenance.md) so the current Pi
 * session performs the two-step ormah_run_maintenance flow. The whisper
 * `maintenance_due` signal asks the agent to do the same thing unprompted.
 */

import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

export async function loadMaintenancePrompt(): Promise<string> {
	const path = join(HERE, "..", "agents", "ormah-maintenance.md");
	const raw = await readFile(path, "utf8");
	// Strip YAML frontmatter — the body is the agent prompt.
	const fmEnd = raw.indexOf("\n---\n", raw.startsWith("---\n") ? 4 : 0);
	return fmEnd >= 0 ? raw.slice(fmEnd + 5).trim() : raw;
}
