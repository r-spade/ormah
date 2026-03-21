"""Format graph data as human/agent-readable text."""

from __future__ import annotations

from typing import Any


def format_node(node: dict[str, Any], edges: list[dict[str, Any]] | None = None) -> str:
    """Format a single node as readable text."""
    lines = []
    title = node.get("title") or _excerpt(node.get("content", ""))
    lines.append(f"## [{node['type']}] {title}")
    lines.append(f"ID: {node['id']}")
    lines.append(f"Tier: {node['tier']} | Space: {node.get('space') or 'unassigned'}")

    content = node.get("content", "").strip()
    if content:
        lines.append("")
        lines.append(content)

    if edges:
        lines.append("")
        lines.append("### Connections")
        for e in edges:
            other_id = e["target_id"] if e["source_id"] == node["id"] else e["source_id"]
            direction = "→" if e["source_id"] == node["id"] else "←"
            lines.append(f"  {direction} {e['edge_type']} ({e['weight']:.1f}) {other_id}")

    return "\n".join(lines)


def format_node_with_neighbors(
    node: dict[str, Any],
    edges: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
) -> str:
    """Format a node with its neighborhood for agent consumption."""
    lines = [format_node(node, edges)]

    if neighbors:
        lines.append("")
        lines.append("---")
        lines.append("### Nearby Memories")
        for n in neighbors[:10]:
            title = n.get("title") or _excerpt(n.get("content", ""))
            lines.append(f"- [{n['type']}] {title} (id: {n['id'][:8]}...)")

    return "\n".join(lines)


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results for agent consumption.

    Results are sorted by score. Each entry is labelled by its source:
    - ``hybrid`` / ``fts`` — direct search hit
    - ``activated`` — reached via graph edge from a direct hit
    - ``conflict`` — reached via a ``contradicts`` edge (flagged for the agent)
    """
    if not results:
        return "No memories found matching your query."

    direct = [r for r in results if r.get("source") not in ("activated", "conflict")]
    activated = [r for r in results if r.get("source") == "activated"]
    conflicts = [r for r in results if r.get("source") == "conflict"]

    parts: list[str] = []
    if activated or conflicts:
        extra = len(activated) + len(conflicts)
        parts.append(f"Found {len(direct)} memories (+{extra} related):\n")
    else:
        parts.append(f"Found {len(direct)} memories:\n")

    # Direct results
    for i, r in enumerate(direct, 1):
        node = r["node"] if "node" in r else r
        title = node.get("title") or _excerpt(node.get("content", ""))
        score = r.get("score", 0)
        parts.append(f"{i}. [{node['type']}] {title}")
        created = (node.get("created") or "")[:19]  # YYYY-MM-DDTHH:MM:SS
        parts.append(f"   ID: {node['id']} | Tier: {node['tier']} | Score: {score:.3f} | Created: {created}")
        content = node.get("content", "").strip()
        if content:
            parts.append(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
        parts.append("")

    # Activated (non-conflict) results
    if activated:
        parts.append("--- Related (via graph) ---\n")
        for i, r in enumerate(activated, 1):
            node = r["node"] if "node" in r else r
            title = node.get("title") or _excerpt(node.get("content", ""))
            score = r.get("score", 0)
            edge_type = r.get("activation_edge", "related_to")
            seed_id = r.get("activated_by", "")
            parts.append(f"{i}. [{node['type']}] {title}")
            parts.append(f"   ID: {node['id']} | Score: {score:.3f} | via {edge_type} from {seed_id[:8]}...")
            content = node.get("content", "").strip()
            if content:
                parts.append(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
            parts.append("")

    # Conflicting context
    if conflicts:
        parts.append("--- Conflicting context ---\n")
        for i, r in enumerate(conflicts, 1):
            node = r["node"] if "node" in r else r
            title = node.get("title") or _excerpt(node.get("content", ""))
            score = r.get("score", 0)
            seed_id = r.get("activated_by", "")
            parts.append(f"{i}. [{node['type']}] {title}")
            parts.append(f"   ID: {node['id']} | Score: {score:.3f} | contradicts {seed_id[:8]}...")
            content = node.get("content", "").strip()
            if content:
                parts.append(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
            parts.append("")

    return "\n".join(parts)


def format_identity_section(
    identity_nodes: list[dict[str, Any]],
    max_content_len: int = 300,
    header_prefix: str = "#",
    include_ids: bool = False,
) -> str:
    """Format identity-linked nodes as an 'About the User' section."""
    if not identity_nodes:
        return "No user identity information stored yet."

    lines = [f"{header_prefix} About the User\n"]
    for node in identity_nodes:
        title = node.get("title") or _excerpt(node.get("content", ""))
        id_suffix = f" (id: {node['id'][:8]})" if include_ids and node.get("id") else ""
        lines.append(f"- **[{node['type']}]** {title}{id_suffix}")
        content = node.get("content", "").strip()
        if content and content != title:
            lines.append(f"  {content[:max_content_len]}")
        lines.append("")

    return "\n".join(lines)


def format_context(
    core_nodes: list[dict[str, Any]],
    max_content_len: int = 300,
    header_prefix: str = "#",
    include_ids: bool = False,
) -> str:
    """Format core memories for system prompt injection."""
    if not core_nodes:
        return ""

    lines = [f"{header_prefix} Core Memories\n"]
    for node in core_nodes:
        title = node.get("title") or _excerpt(node.get("content", ""))
        id_suffix = f" (id: {node['id'][:8]})" if include_ids and node.get("id") else ""
        lines.append(f"- **[{node['type']}]** {title}{id_suffix}")
        content = node.get("content", "").strip()
        if content:
            lines.append(f"  {content[:max_content_len]}")
        lines.append("")

    return "\n".join(lines)


def format_context_with_project(
    core_nodes: list[dict[str, Any]],
    project_nodes: list[dict[str, Any]],
    space: str,
    max_content_len: int = 300,
    header_prefix: str = "#",
    include_ids: bool = False,
) -> str:
    """Format core memories + project working memories for system prompt injection."""
    if not core_nodes and not project_nodes:
        return ""

    lines = []

    # Core section (always present)
    if core_nodes:
        lines.append(f"{header_prefix} Core Memories\n")
        for node in core_nodes:
            title = node.get("title") or _excerpt(node.get("content", ""))
            id_suffix = f" (id: {node['id'][:8]})" if include_ids and node.get("id") else ""
            lines.append(f"- **[{node['type']}]** {title}{id_suffix}")
            content = node.get("content", "").strip()
            if content:
                lines.append(f"  {content[:max_content_len]}")
            lines.append("")

    # Project section
    if project_nodes:
        lines.append(f"{header_prefix} Project: {space}\n")
        for node in project_nodes:
            title = node.get("title") or _excerpt(node.get("content", ""))
            id_suffix = f" (id: {node['id'][:8]})" if include_ids and node.get("id") else ""
            lines.append(f"- **[{node['type']}]** {title}{id_suffix}")
            content = node.get("content", "").strip()
            if content:
                lines.append(f"  {content[:max_content_len]}")
            lines.append("")
    elif core_nodes:
        lines.append(f"{header_prefix} Project: {space}\n")
        lines.append("No project-specific memories yet.\n")

    return "\n".join(lines)


def _excerpt(text: str, max_len: int = 60) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."
