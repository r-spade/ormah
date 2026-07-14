/**
 * Space detection — mirrors src/ormah/adapters/space_detect.py so the Pi
 * extension and the Ormah server agree on which "space" a memory belongs to.
 *
 * Order: explicit override > ORMAH_SPACE env > git repo basename (from cwd) >
 * directory basename > null (home/root).
 */

import { execSync } from "node:child_process";
import { homedir } from "node:os";
import { realpathSync } from "node:fs";
import { basename } from "node:path";

function gitRepoName(dir: string): string | null {
	try {
		const top = execSync("git rev-parse --show-toplevel", {
			cwd: dir,
			timeout: 5000,
			stdio: ["ignore", "pipe", "ignore"],
		})
			.toString()
			.trim();
		return top ? basename(top) : null;
	} catch {
		return null;
	}
}

export function detectSpaceFromDir(dir: string): string | null {
	let real: string;
	try {
		real = realpathSync(dir);
	} catch {
		return null;
	}
	const home = homedir();
	if (real === home || real === "/") return null;
	return gitRepoName(real) ?? basename(real);
}

export function resolveSpace(
	explicit?: string | null,
	cwd?: string,
): string | null {
	if (explicit !== undefined) return explicit;
	const envSpace = process.env.ORMAH_SPACE;
	if (envSpace) return envSpace;
	return cwd ? detectSpaceFromDir(cwd) : null;
}
