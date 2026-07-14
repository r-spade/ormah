/**
 * Thin HTTP client for the local Ormah server.
 *
 * Mirrors the request/response shapes used by Ormah's own MCP adapter
 * (src/ormah/adapters/mcp_adapter.py _dispatch) and the /agent/* routes in
 * src/ormah/api/routes_agent.py. An HTTP-only client reproduces every MCP tool
 * natively, so the Pi extension does not need to spawn `ormah mcp`.
 */

import type { OrmahConfig } from "./config.js";

export interface WhisperResponse {
	text: string;
	node_id?: string | null;
}

export interface RememberBody {
	content: string;
	type?: string;
	tier?: string;
	title?: string;
	space?: string | null;
	tags?: string[];
	about_self?: boolean;
	confidence?: number;
	/** node IDs to link at creation time (maps to connections[].target). */
	links?: string[];
}

export interface RecallBody {
	query: string;
	limit?: number;
	types?: string[];
	tiers?: string[];
	spaces?: string[];
	tags?: string[];
	created_after?: string;
	created_before?: string;
	session_id?: string;
}

export interface IngestBody {
	content: string;
	agent_id?: string;
	space?: string | null;
	extra_tags?: string[];
	dry_run?: boolean;
}

export interface MaintenanceResults {
	edges?: Array<{
		node_a_id: string;
		node_b_id: string;
		edge_type: string;
		reason?: string;
	}>;
	merges?: Array<{
		keep_id: string;
		discard_id: string;
		merged_content?: string;
		merged_title?: string;
	}>;
	consolidations?: Array<{
		node_ids: string[];
		title?: string;
		content?: string;
		type?: string;
	}>;
}

export class OrmahClient {
	constructor(private readonly cfg: OrmahConfig) {}

	private async req<T>(
		path: string,
		init: RequestInit & { timeoutMs?: number } = {},
	): Promise<T> {
		const { timeoutMs, ...rest } = init;
		const controller = new AbortController();
		const timer = setTimeout(
			() => controller.abort(),
			timeoutMs ?? this.cfg.toolTimeoutMs,
		);
		try {
			const signal = rest.signal
				? AbortSignal.any([controller.signal, rest.signal])
				: controller.signal;
			const resp = await fetch(`${this.cfg.baseUrl}${path}`, {
				...rest,
				signal,
				headers: {
					"Content-Type": "application/json",
					...(rest.headers ?? {}),
				},
			});
			if (!resp.ok) {
				const detail = await resp.text().catch(() => "");
				throw new OrmahHttpError(resp.status, detail || resp.statusText);
			}
			return (await resp.json()) as T;
		} finally {
			clearTimeout(timer);
		}
	}

	/** GET /admin/stats — lightweight liveness + store size. */
	async health(
		signal?: AbortSignal,
	): Promise<{
		total_nodes: number;
		by_tier: Record<string, number>;
		total_edges: number;
	}> {
		return this.req("/admin/stats", { method: "GET", signal, timeoutMs: 4000 });
	}

	/** POST /agent/whisper — involuntary recall context for the next prompt. */
	async whisper(
		body: { prompt: string; space?: string | null; session_id?: string },
		signal?: AbortSignal,
	): Promise<WhisperResponse> {
		return this.req("/agent/whisper", {
			method: "POST",
			body: JSON.stringify(body),
			signal,
			timeoutMs: this.cfg.whisperTimeoutMs,
		});
	}

	/** POST /agent/remember — store a memory. */
	async remember(
		body: RememberBody,
		defaultSpace?: string | null,
	): Promise<WhisperResponse> {
		const payload: Record<string, unknown> = {
			content: body.content,
			type: body.type ?? "fact",
			tier: body.tier ?? "working",
		};
		if (body.title) payload.title = body.title;
		if (body.space !== undefined) payload.space = body.space;
		if (body.tags) payload.tags = body.tags;
		if (body.about_self) payload.about_self = true;
		if (body.confidence !== undefined) payload.confidence = body.confidence;
		if (body.links?.length)
			payload.connections = body.links.map((id) => ({ target: id }));
		const params = body.space === undefined && defaultSpace
			? `?default_space=${encodeURIComponent(defaultSpace)}`
			: "";
		return this.req(`/agent/remember${params}`, {
			method: "POST",
			body: JSON.stringify(payload),
			timeoutMs: this.cfg.toolTimeoutMs,
		});
	}

	/** POST /agent/recall — hybrid search. */
	async recall(
		body: RecallBody,
		defaultSpace?: string | null,
	): Promise<WhisperResponse> {
		const params = defaultSpace
			? `?default_space=${encodeURIComponent(defaultSpace)}`
			: "";
		return this.req(`/agent/recall${params}`, {
			method: "POST",
			body: JSON.stringify(body),
		});
	}

	/** GET /agent/recall/{node_id} — one memory + connections. */
	async recallNode(
		nodeId: string,
		sessionId?: string,
	): Promise<WhisperResponse> {
		const params = sessionId
			? `?session_id=${encodeURIComponent(sessionId)}`
			: "";
		return this.req(`/agent/recall/${encodeURIComponent(nodeId)}${params}`, {
			method: "GET",
		});
	}

	/** POST /agent/outdated/{node_id} — demote a memory. */
	async markOutdated(
		nodeId: string,
		reason?: string,
	): Promise<WhisperResponse> {
		const body = reason ? JSON.stringify({ reason }) : undefined;
		return this.req(`/agent/outdated/${encodeURIComponent(nodeId)}`, {
			method: "POST",
			body,
		});
	}

	/** POST /agent/feedback — record a relevance signal. */
	async submitFeedback(
		nodeId: string,
		signal: 1 | -1,
		source: "explicit" | "implicit" = "explicit",
	): Promise<WhisperResponse> {
		return this.req("/agent/feedback", {
			method: "POST",
			body: JSON.stringify({ node_id: nodeId, signal, source }),
		});
	}

	/**
	 * POST /agent/maintenance — two-step graph maintenance.
	 * No results -> phase 1 (returns batches + job_id, status "awaiting_results").
	 * With results -> phase 2 (applies decisions, returns apply_summary).
	 */
	async runMaintenance(
		opts: { jobId?: string; results?: MaintenanceResults },
		signal?: AbortSignal,
	): Promise<Record<string, unknown>> {
		const body: Record<string, unknown> = {};
		if (opts.jobId) body.job_id = opts.jobId;
		if (opts.results) body.results = opts.results;
		const initial = await this.req<Record<string, unknown>>("/agent/maintenance", {
			method: "POST",
			body: JSON.stringify(body),
			signal,
			timeoutMs: this.cfg.maintenanceTimeoutMs,
		});
		return this._pollMaintenance(initial, !!opts.results, signal);
	}

	/** Poll /agent/maintenance until the requested phase is ready. Mirrors
	ormah mcp_adapter._poll_maintenance_until_ready: phase 1 (no results) waits for
	status "awaiting_results" + batches; phase 2 (with results) waits for "completed" + apply_summary. */
	private async _pollMaintenance(
		initial: Record<string, unknown>,
		expectApplySummary: boolean,
		signal?: AbortSignal,
	): Promise<Record<string, unknown>> {
		const deadline = Date.now() + this.cfg.maintenanceTimeoutMs;
		let data = initial;
		let jobId = (data.job_id as string | undefined) ?? undefined;
		while (true) {
			const status = data.status as string | undefined;
			if (expectApplySummary) {
				if (status === "completed" && typeof data.apply_summary === "object") return data;
			} else {
				if (status === "awaiting_results" && typeof data.batches === "object") return data;
			}
			if (status === "failed") {
				throw new Error(`maintenance job failed: ${(data.last_error as string) || "unknown"}`);
			}
			if (Date.now() >= deadline) {
				throw new OrmahHttpError(0, `run_maintenance timed out after ${this.cfg.maintenanceTimeoutMs}ms`);
			}
			await new Promise((r) => setTimeout(r, 1000));
			if (signal?.aborted) throw new Error("aborted");
			data = await this.getMaintenanceStatus(jobId);
		}
	}

	/** GET /agent/maintenance — poll a running maintenance job. */
	async getMaintenanceStatus(jobId?: string): Promise<Record<string, unknown>> {
		const params = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
		return this.req(`/agent/maintenance${params}`, {
			method: "GET",
			timeoutMs: this.cfg.toolTimeoutMs,
		});
	}

	/** POST /ingest/conversation — server-side memory extraction from text. */
	async ingestConversation(
		body: IngestBody,
	): Promise<{ status: string; extracted: number; memories: unknown[] }> {
		const params = new URLSearchParams();
		if (body.space) params.set("default_space", body.space);
		if (body.dry_run) params.set("dry_run", "true");
		if (body.extra_tags?.length)
			params.set("extra_tags", body.extra_tags.join(","));
		const qs = params.toString() ? `?${params.toString()}` : "";
		return this.req(`/ingest/conversation${qs}`, {
			method: "POST",
			body: JSON.stringify({ agent_id: body.agent_id, content: body.content }),
			timeoutMs: this.cfg.storeTimeoutMs,
		});
	}
}

export class OrmahHttpError extends Error {
	constructor(
		readonly status: number,
		detail: string,
	) {
		super(`Ormah HTTP ${status}: ${detail}`);
		this.name = "OrmahHttpError";
	}
}
