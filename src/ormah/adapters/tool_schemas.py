"""Canonical tool definitions shared across MCP and OpenAI adapters.

TOOLS: The core set of tools exposed via MCP to AI agents (10 tools).
ADMIN_TOOLS: Tools for human administration via CLI/API only (4 tools).
ALL_TOOLS: Combined list for adapters that want the full set.
"""

from __future__ import annotations

_RECALL_NODE_TOOL = {
    "name": "recall_node",
    "description": (
        "Get a specific memory by its ID, including its connections to other memories. "
        "Use this to dive deeper into a memory found via search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "The UUID of the memory to retrieve.",
            },
        },
        "required": ["node_id"],
    },
}

TOOLS = [
    {
        "name": "remember",
        "description": (
            "Store a new memory. Use this to save facts, decisions, "
            "preferences, observations, or any information worth remembering across sessions. "
            "Memories are automatically indexed and embedded. If you have node IDs from a recent "
            "recall, pass them via `links` to connect related memories at creation time — "
            "background jobs will also discover relationships automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Be specific and self-contained.",
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "fact", "decision", "preference", "event", "person",
                        "project", "concept", "procedure", "goal", "observation",
                    ],
                    "description": "The type of memory.",
                    "default": "fact",
                },
                "tier": {
                    "type": "string",
                    "enum": ["core", "working", "archival"],
                    "description": "Memory importance tier. 'core' = always loaded, 'working' = searchable, 'archival' = deep storage.",
                    "default": "working",
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Short descriptive title for the memory. "
                        "Write it as a self-contained one-line summary — "
                        "whisper shows only the title when this memory is not in the top 2 results, "
                        "so it must convey the key fact on its own."
                    ),
                },
                "space": {
                    "type": "string",
                    "description": (
                        "Organizational space/project this memory belongs to. "
                        "Auto-detected from the current project directory if not set. "
                        "Explicitly set to null for personal/global memories (identity, preferences, cross-project facts)."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization.",
                },
                "about_self": {
                    "type": "boolean",
                    "description": (
                        "Set to true if this memory is about the user's identity, "
                        "personal information, or preferences. This marks the memory "
                        "as user-related for recall and whisper."
                    ),
                    "default": False,
                },
                "confidence": {
                    "type": "number",
                    "description": "Belief strength 0.0-1.0. Lower values mean less certain.",
                    "default": 1.0,
                },
                "links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Node IDs of related memories to link at creation time. "
                        "Use when you have IDs from a recent recall and know these memories belong together. "
                        "Creates 'related_to' edges. Background jobs will classify the relationship type further."
                    ),
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Search memories by natural language query. Returns the most relevant memories "
            "using hybrid full-text + semantic search. Results are automatically prioritized "
            "for the current project, then global memories, then other projects. "
            "Use this when you need to find information from past conversations or stored knowledge, "
            "including personal facts, preferences, and prior project context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 10,
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by memory types.",
                },
                "spaces": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by spaces.",
                },
                "created_after": {
                    "type": "string",
                    "description": "ISO datetime — only return memories created after this time.",
                },
                "created_before": {
                    "type": "string",
                    "description": "ISO datetime — only return memories created before this time.",
                },
            },
            "required": ["query"],
        },
    },
    _RECALL_NODE_TOOL,
    {
        "name": "mark_outdated",
        "description": (
            "Mark a memory as outdated. Sets the expiry date to now, which "
            "demotes it in search results. Optionally append a reason explaining "
            "why the memory is no longer valid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The UUID of the memory to mark as outdated.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional explanation of why the memory is outdated.",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "submit_feedback",
        "description": (
            "Record relevance feedback on a memory. "
            "Call with source='implicit' when you can judge from context whether a whispered or recalled "
            "memory was useful — signal=1 if useful, signal=-1 if not relevant. "
            "Only ask the user (source='explicit') when you genuinely cannot determine relevance yourself. "
            "The system uses this feedback to improve which memories surface in future sessions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the memory node the feedback is about.",
                },
                "signal": {
                    "type": "integer",
                    "enum": [1, -1],
                    "description": "1 if the memory would have been useful, -1 if it was not relevant.",
                },
                "source": {
                    "type": "string",
                    "enum": ["explicit", "implicit"],
                    "description": (
                        "'explicit' when the user answered a review question (default). "
                        "'implicit' when you infer usefulness from the conversation without asking."
                    ),
                    "default": "explicit",
                },
            },
            "required": ["node_id", "signal"],
        },
    },
    {
        "name": "run_maintenance",
        "description": (
            "Maintain the memory graph by linking, conflict-checking, deduplicating, and "
            "consolidating memories. Uses a two-call protocol:\n\n"
            "**Phase 1** — call with no arguments to get pending work. Returns four batches:\n"
            "  - link_candidates: pairs of memories to classify (supports/part_of/etc./none)\n"
            "  - conflict_candidates: belief pairs to check for contradictions or evolutions\n"
            "  - merge_candidates: near-duplicate pairs to merge\n"
            "  - consolidation_clusters: groups of similar memories to synthesize into one\n\n"
            "**Phase 2** — analyze all four batches in-context, then call again with 'results':\n"
            "  - edges: list of {node_a_id, node_b_id, edge_type, reason} — use 'none' to skip\n"
            "  - merges: list of {keep_id, discard_id, merged_content, merged_title}\n"
            "  - consolidations: list of {node_ids, title, content, type}\n\n"
            "Use when whisper or another Ormah signal indicates maintenance is due. "
            "No separate API key needed — the calling LLM performs the analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "object",
                    "description": (
                        "Phase 2 only: your analysis of the batches returned in Phase 1. "
                        "Omit this parameter in Phase 1 to get the pending work."
                    ),
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_a_id": {"type": "string"},
                                    "node_b_id": {"type": "string"},
                                    "edge_type": {
                                        "type": "string",
                                        "enum": [
                                            "supports", "contradicts", "evolved_from",
                                            "part_of", "depends_on", "related_to", "none",
                                        ],
                                    },
                                    "reason": {"type": "string"},
                                },
                                "required": ["node_a_id", "node_b_id", "edge_type"],
                            },
                        },
                        "merges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "keep_id": {"type": "string"},
                                    "discard_id": {"type": "string"},
                                    "merged_content": {"type": "string"},
                                    "merged_title": {"type": "string"},
                                },
                                "required": ["keep_id", "discard_id"],
                            },
                        },
                        "consolidations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["node_ids", "title", "content"],
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "name": "update_memory",
        "description": (
            "Correct or enhance an existing memory. Use this when you find that a stored memory "
            "has wrong content, an outdated type/tier, or needs better tags. "
            "Only pass the fields you want to change — omitted fields are left unchanged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The UUID of the memory to update.",
                },
                "content": {
                    "type": "string",
                    "description": "Corrected or updated content.",
                },
                "title": {
                    "type": "string",
                    "description": "New short title.",
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "fact", "decision", "preference", "event", "person",
                        "project", "concept", "procedure", "goal", "observation",
                    ],
                    "description": "New memory type.",
                },
                "tier": {
                    "type": "string",
                    "enum": ["core", "working", "archival"],
                    "description": "New importance tier. Use 'core' to pin permanently, 'archival' to deprioritize.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Updated confidence 0.0–1.0.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement tag list (replaces existing tags entirely).",
                },
                "space": {
                    "type": "string",
                    "description": "New space/project scope. Use null to make the memory global.",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "connect_memories",
        "description": (
            "Create a typed edge between two memories. Use this when you recognise a meaningful "
            "relationship between nodes — e.g. one decision supports another, a fact is part of a "
            "concept, or two observations contradict each other. "
            "Background auto-linker runs periodically, but call this for immediate graph updates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source memory UUID."},
                "target_id": {"type": "string", "description": "Target memory UUID."},
                "edge": {
                    "type": "string",
                    "enum": [
                        "related_to", "supports", "contradicts", "part_of",
                        "derived_from", "depends_on", "defines", "evolved_from",
                    ],
                    "description": "The relationship type.",
                    "default": "related_to",
                },
                "weight": {
                    "type": "number",
                    "description": "Connection strength 0.0–1.0.",
                    "default": 0.5,
                },
            },
            "required": ["source_id", "target_id"],
        },
    },
    {
        "name": "promote_memory",
        "description": (
            "Promote a memory to core tier, marking it as permanently important. "
            "Use this when you discover a memory that should always be accessible — "
            "key decisions, identity facts, architectural constants. "
            "Core memories are never decayed and always surface in whisper context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "UUID of the memory to promote.",
                },
                "tier": {
                    "type": "string",
                    "enum": ["core", "working"],
                    "description": "Target tier (default: core). Use 'working' to un-archive a demoted memory.",
                    "default": "core",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "recall_history",
        "description": (
            "Show the full edit history for a memory. Returns a chronological changelog "
            "of all updates with field-by-field old→new diffs. "
            "Use this to understand how a memory has changed over time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The UUID of the memory to show history for.",
                },
            },
            "required": ["node_id"],
        },
    },
]

# Admin tools — available via CLI and HTTP API but not exposed to AI agents via MCP.
# These are for human review and administration of the memory system.
ADMIN_TOOLS = [
    {
        "name": "list_proposals",
        "description": (
            "Show pending merge/conflict proposals with human-readable reasons. "
            "Use this to review what the system has detected as potential duplicates or conflicts."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "resolve_proposal",
        "description": (
            "Approve or reject a pending proposal. When a merge proposal is approved, "
            "the merge is executed automatically — the duplicate node is removed and its "
            "edges and tags are transferred to the kept node."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "proposal_id": {
                    "type": "string",
                    "description": "The UUID of the proposal to resolve.",
                },
                "action": {
                    "type": "string",
                    "enum": ["approved", "rejected"],
                    "description": "Whether to approve or reject the proposal.",
                },
            },
            "required": ["proposal_id", "action"],
        },
    },
    {
        "name": "list_merges",
        "description": (
            "Show recent merge history with kept/removed node info and undo status. "
            "Use this to review what merges have been performed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of merges to return.",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "undo_merge",
        "description": (
            "Rollback a merge by its ID (supports prefix match). "
            "Restores the removed node with its original edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "merge_id": {
                    "type": "string",
                    "description": "The UUID (or prefix) of the merge to undo.",
                },
            },
            "required": ["merge_id"],
        },
    },
    {
        "name": "list_audit_log",
        "description": (
            "Show recent audit log entries for memory operations (deletes, updates, mark_outdated). "
            "Each entry includes the full node snapshot before the operation. "
            "Use this to review what changes have been made to memories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return.",
                    "default": 20,
                },
                "node_id": {
                    "type": "string",
                    "description": "Filter entries for a specific node ID.",
                },
                "operation": {
                    "type": "string",
                    "enum": ["delete", "update", "mark_outdated"],
                    "description": "Filter by operation type.",
                },
            },
        },
    },
]

ALL_TOOLS = TOOLS + ADMIN_TOOLS
