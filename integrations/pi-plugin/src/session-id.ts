import { randomUUID } from "node:crypto";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

const inMemorySessionIds = new WeakMap<object, string>();

export function beginSession(ctx: ExtensionContext): string {
	const sessionFile = ctx.sessionManager.getSessionFile();
	if (sessionFile) return sessionFile;

	const sessionId = `pi-${randomUUID()}`;
	inMemorySessionIds.set(ctx.sessionManager, sessionId);
	return sessionId;
}

export function getSessionId(ctx: ExtensionContext): string {
	const sessionFile = ctx.sessionManager.getSessionFile();
	if (sessionFile) return sessionFile;

	const existing = inMemorySessionIds.get(ctx.sessionManager);
	return existing ?? beginSession(ctx);
}
