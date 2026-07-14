"""Format graph data as human/agent-readable text."""

from __future__ import annotations

from typing import Any


def _feedback_id_suffix(node: dict[str, Any], *, short: bool = False) -> str:
    node_id = node.get("id", "")
    if short:
        label = f"id: {node_id[:8]}..." if node_id else "id: unknown"
    else:
        label = f"ID: {node_id}"
    whisper_log_id = node.get("_whisper_log_id")
    if whisper_log_id is not None:
        label += f" | whisper_log_id: {whisper_log_id}"
    return label


def format_node(node: dict[str, Any], edges: list[dict[str, Any]] | None = None) -> str:
    """Format a single node as readable text."""
    lines = []
    title = node.get("title") or _excerpt(node.get("content", ""))
    lines.append(f"## [{node['type']}] {title}")
    lines.append(_feedback_id_suffix(node))
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
            lines.append(f"- [{n['type']}] {title} ({_feedback_id_suffix(n, short=True)})")

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
        if r.get("_whisper_log_id") is not None:
            node = {**node, "_whisper_log_id": r["_whisper_log_id"]}
        created = (node.get("created") or "")[:19]  # YYYY-MM-DDTHH:MM:SS
        parts.append(
            f"   {_feedback_id_suffix(node)} | Tier: {node['tier']} | "
            f"Score: {score:.3f} | Created: {created}"
        )
        content = node.get("content", "").strip()
        if content:
            parts.append(f"   {content}")
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
            if r.get("_whisper_log_id") is not None:
                node = {**node, "_whisper_log_id": r["_whisper_log_id"]}
            parts.append(
                f"   {_feedback_id_suffix(node)} | Score: {score:.3f} | "
                f"via {edge_type} from {seed_id[:8]}..."
            )
            content = node.get("content", "").strip()
            if content:
                parts.append(f"   {content}")
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
            if r.get("_whisper_log_id") is not None:
                node = {**node, "_whisper_log_id": r["_whisper_log_id"]}
            parts.append(
                f"   {_feedback_id_suffix(node)} | Score: {score:.3f} | "
                f"contradicts {seed_id[:8]}..."
            )
            content = node.get("content", "").strip()
            if content:
                parts.append(f"   {content}")
            parts.append("")

    return "\n".join(parts)


def _excerpt(text: str, max_len: int = 60) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."
