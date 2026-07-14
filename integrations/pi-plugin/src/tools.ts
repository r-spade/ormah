/**
 * Ormah memory tools exposed to the Pi LLM.
 *
 * One-to-one with the 6 MCP tools in src/ormah/adapters/tool_schemas.py,
 * proxied to the Ormah HTTP API (same routes the MCP adapter uses). Registering
 * them as native Pi tools (rather than relying on an external MCP server) keeps
 * configuration in one place and lets whisper + tools share one client.
 */

import type {
	ExtensionAPI,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";
import type { OrmahClient, MaintenanceResults } from "./client.js";
import { truncateHead, formatSize } from "@earendil-works/pi-coding-agent";
import { getSessionId } from "./session-id.js";
import { resolveSpace } from "./space.js";

const NODE_TYPES = [
	"fact",
	"decision",
	"preference",
	"event",
	"person",
	"project",
	"concept",
	"procedure",
	"goal",
	"observation",
] as const;
const TIERS = ["core", "working", "archival"] as const;
const EDGE_TYPES = [
	"supports",
	"contradicts",
	"evolved_from",
	"part_of",
	"depends_on",
	"related_to",
	"none",
] as const;

export interface ToolCtx {
	client: OrmahClient;
}

function textResult(text: string) {
	return { content: [{ type: "text" as const, text }], details: {} };
}

/** Wrap tool output with the same truncation policy built-in tools use. */
function safeText(raw: string): string {
	const t = truncateHead(raw, {});
	let out = t.content;
	if (t.truncated) {
		out += `\n\n[Output truncated: ${t.outputLines} of ${t.totalLines} lines (${formatSize(t.outputBytes)} of ${formatSize(t.totalBytes)}).]`;
	}
	return out;
}

export function registerTools(pi: ExtensionAPI, tctx: ToolCtx): void {
	const { client } = tctx;
	const maintenanceJobs = new Map<string, string>();

	// ── remember ──────────────────────────────────────────────────────────────
	pi.registerTool({
		name: "ormah_remember",
		label: "Ormah Remember",
		description:
			"Store a new memory in Ormah. Use to save facts, decisions, preferences, observations, or anything worth remembering across sessions. Memories are indexed and embedded automatically.",
		promptSnippet:
			"Store a durable memory in Ormah (facts, decisions, preferences, observations)",
		promptGuidelines: [
			"Use ormah_remember when the user states a preference, makes a decision, or shares a fact worth keeping across sessions. Write self-contained content and a one-line title.",
		],
		parameters: Type.Object({
			content: Type.String({
				description:
					"The memory content to store. Be specific and self-contained.",
			}),
			type: Type.Optional(
				StringEnum(NODE_TYPES, {
					description: "The type of memory.",
					default: "fact",
				}),
			),
			tier: Type.Optional(
				StringEnum(TIERS, {
					description:
						"core = always loaded, working = searchable, archival = deep storage.",
					default: "working",
				}),
			),
			title: Type.Optional(
				Type.String({
					description:
						"Short self-contained one-line summary; whisper shows only the title for non-top results.",
				}),
			),
			space: Type.Optional(
				Type.Union([Type.String(), Type.Null()], {
					description:
						"Project/space this memory belongs to. Omit to auto-detect; set to null for personal/global memories.",
				}),
			),
			tags: Type.Optional(
				Type.Array(Type.String(), { description: "Tags for categorization." }),
			),
			about_self: Type.Optional(
				Type.Boolean({
					description:
						"True if this memory is about the user's identity or preferences.",
					default: false,
				}),
			),
			confidence: Type.Optional(
				Type.Number({
					description: "Belief strength 0.0-1.0. Lower = less certain.",
					default: 1.0,
				}),
			),
			links: Type.Optional(
				Type.Array(Type.String(), {
					description: "Node IDs of related memories to link at creation time.",
				}),
			),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			try {
				const resp = await client.remember(
					params,
					resolveSpace(undefined, ctx.cwd),
				);
				return textResult(resp.text);
			} catch (e) {
				throw new Error(`ormah_remember failed: ${(e as Error).message}`);
			}
		},
	});

	// ── recall ────────────────────────────────────────────────────────────────
	pi.registerTool({
		name: "ormah_recall",
		label: "Ormah Recall",
		description:
			"Search Ormah memories by natural language query. Hybrid full-text + semantic search, prioritized for the current project then global then other projects. Use to find info from past sessions or stored knowledge.",
		promptSnippet: "Search Ormah memories by natural language query",
		promptGuidelines: [
			"Use ormah_recall when the user asks about past work, prior decisions, or context that may already be stored in memory.",
		],
		parameters: Type.Object({
			query: Type.String({ description: "Natural language search query." }),
			limit: Type.Optional(
				Type.Integer({ description: "Max results.", default: 10 }),
			),
			types: Type.Optional(
				Type.Array(Type.String(), { description: "Filter by memory types." }),
			),
			spaces: Type.Optional(
				Type.Array(Type.String(), { description: "Filter by spaces." }),
			),
			created_after: Type.Optional(
				Type.String({
					description: "ISO datetime — only memories created after.",
				}),
			),
			created_before: Type.Optional(
				Type.String({
					description: "ISO datetime — only memories created before.",
				}),
			),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			try {
				const resp = await client.recall(
					{ ...params, session_id: getSessionId(ctx) },
					resolveSpace(undefined, ctx.cwd),
				);
				return textResult(safeText(resp.text));
			} catch (e) {
				throw new Error(`ormah_recall failed: ${(e as Error).message}`);
			}
		},
	});

	// ── recall_node ───────────────────────────────────────────────────────────
	pi.registerTool({
		name: "ormah_recall_node",
		label: "Ormah Recall Node",
		description:
			"Get a specific Ormah memory by ID, including its connections. Use to dive deeper into a memory found via search.",
		promptSnippet: "Fetch one Ormah memory by ID with its connections",
		parameters: Type.Object({
			node_id: Type.String({
				description: "The UUID of the memory to retrieve.",
			}),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			try {
				const resp = await client.recallNode(
					params.node_id,
					getSessionId(ctx),
				);
				return textResult(safeText(resp.text));
			} catch (e) {
				throw new Error(`ormah_recall_node failed: ${(e as Error).message}`);
			}
		},
	});

	// ── mark_outdated ─────────────────────────────────────────────────────────
	pi.registerTool({
		name: "ormah_mark_outdated",
		label: "Ormah Mark Outdated",
		description:
			"Mark an Ormah memory as outdated. Sets expiry to now, demoting it in search results. Optionally append a reason.",
		promptSnippet: "Mark an Ormah memory outdated (demote in search)",
		parameters: Type.Object({
			node_id: Type.String({
				description: "The UUID of the memory to mark outdated.",
			}),
			reason: Type.Optional(
				Type.String({
					description: "Optional explanation of why it is outdated.",
				}),
			),
		}),
		async execute(_id, params) {
			try {
				const resp = await client.markOutdated(params.node_id, params.reason);
				return textResult(resp.text);
			} catch (e) {
				throw new Error(`ormah_mark_outdated failed: ${(e as Error).message}`);
			}
		},
	});

	// ── submit_feedback ───────────────────────────────────────────────────────
	pi.registerTool({
		name: "ormah_submit_feedback",
		label: "Ormah Feedback",
		description:
			"Record relevance feedback on an Ormah memory. Use source='implicit' (signal=1 useful / -1 not relevant) when you can judge from context; only use source='explicit' when you genuinely cannot tell. Improves future whisper ranking.",
		promptSnippet: "Record relevance feedback on an Ormah memory",
		parameters: Type.Object({
			node_id: Type.String({
				description: "The memory node ID the feedback is about.",
			}),
			signal: Type.Union([Type.Literal(1), Type.Literal(-1)], {
				description: "1 = useful, -1 = not relevant.",
			}),
			source: StringEnum(["explicit", "implicit"], {
				description:
					"'explicit' = user answered; 'implicit' = inferred from context.",
				default: "explicit",
			}),
		}),
		async execute(_id, params) {
			try {
				const resp = await client.submitFeedback(
					params.node_id,
					params.signal as 1 | -1,
					params.source as "explicit" | "implicit",
				);
				return textResult(resp.text);
			} catch (e) {
				throw new Error(
					`ormah_submit_feedback failed: ${(e as Error).message}`,
				);
			}
		},
	});

	// ── run_maintenance ───────────────────────────────────────────────────────
	pi.registerTool({
		name: "ormah_run_maintenance",
		label: "Ormah Maintenance",
		description:
			"Maintain the Ormah memory graph: link, conflict-check, dedupe, consolidate. Two-call protocol — Phase 1: call with no results to get pending batches (link_candidates, conflict_candidates, merge_candidates, consolidation_clusters). Phase 2: call with results (edges, merges, consolidations) to apply decisions. Use when whisper signals maintenance_due.",
		promptSnippet:
			"Run Ormah graph maintenance (two-call link/dedup/consolidate flow)",
		promptGuidelines: [
			"Use ormah_run_maintenance when whisper includes a maintenance_due signal. Phase 1 with no results returns batches; Phase 2 with results applies your decisions.",
		],
		parameters: Type.Object({
			results: Type.Optional(
				Type.Object({
					edges: Type.Optional(
						Type.Array(
							Type.Object({
								node_a_id: Type.String(),
								node_b_id: Type.String(),
								edge_type: StringEnum(EDGE_TYPES),
								reason: Type.Optional(Type.String()),
							}),
						),
					),
					merges: Type.Optional(
						Type.Array(
							Type.Object({
								keep_id: Type.String(),
								discard_id: Type.String(),
								merged_content: Type.Optional(Type.String()),
								merged_title: Type.Optional(Type.String()),
							}),
						),
					),
					consolidations: Type.Optional(
						Type.Array(
							Type.Object({
								node_ids: Type.Array(Type.String()),
								title: Type.String(),
								content: Type.String(),
								type: Type.Optional(Type.String()),
							}),
						),
					),
				}),
			),
		}),
		async execute(_id, params, signal, _onUpdate, ctx: ExtensionContext) {
			try {
				const sessionId = getSessionId(ctx);
				const resp = await client.runMaintenance(
					{
						jobId: params.results
							? maintenanceJobs.get(sessionId)
							: undefined,
						results: params.results as MaintenanceResults | undefined,
					},
					signal,
				);
				if (!params.results && typeof resp.job_id === "string") {
					maintenanceJobs.set(sessionId, resp.job_id);
				} else if (params.results && resp.status === "completed") {
					maintenanceJobs.delete(sessionId);
				}
				return textResult(safeText(JSON.stringify(resp, null, 2)));
			} catch (e) {
				throw new Error(
					`ormah_run_maintenance failed: ${(e as Error).message}`,
				);
			}
		},
	});
}
