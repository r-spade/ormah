"""Character budget for the Phase 1 result of ``run_maintenance``.

Claude Code saves an MCP text result above 50,000 characters to a file, and the
maintenance agent has no tool to read that file. The engine therefore cuts the
four batches to one character budget before it issues them, so the batches the
server issues are the batches the agent sees.

Claude Code measures the result with JS ``.length``, so every size here is in
UTF-16 code units: a character outside the BMP (an emoji) counts 2. The field
caps are in the same unit, which keeps ``max_item_cost`` an upper bound.

The cost of an item is an upper bound of its formatted size in
``mcp_adapter._format_maintenance_batches``: it counts every normalized field in
full (the formatter cuts content shorter) plus a fixed overhead for the labels.
"""

from __future__ import annotations

from ormah.models.node import NodeType

TITLE_MAX_CHARS = 200
SPACE_MAX_CHARS = 64
CONTENT_MAX_CHARS = 400

PAIR_BATCHES = ("link_candidates", "conflict_candidates", "merge_candidates")
CLUSTER_BATCH = "consolidation_clusters"

# Formatter labels, measured on the formatter with 3-digit counters. An empty
# space prints as "global", so the pair overhead carries 2 x 6 extra characters.
_SECTION_OVERHEAD = {
    "link_candidates": 128,
    "conflict_candidates": 128,
    "merge_candidates": 184,
    "consolidation_clusters": 88,
}
_PAIR_OVERHEAD = 80
_CLUSTER_OVERHEAD = 32
_CLUSTER_NODE_OVERHEAD = 24

# Node ids are uuid4 strings; the type is a NodeType value.
_ID_MAX_CHARS = 36
_TYPE_MAX_CHARS = max(len(t.value) for t in NodeType)
_NODE_MAX_CHARS = (
    _ID_MAX_CHARS + _TYPE_MAX_CHARS + TITLE_MAX_CHARS + SPACE_MAX_CHARS + CONTENT_MAX_CHARS
)


def utf16_len(text: str) -> int:
    """Length of *text* as JS ``.length`` counts it."""
    return len(text) + sum(ord(ch) > 0xFFFF for ch in text)


def cut_utf16(text: str, limit: int) -> str:
    """The longest prefix of *text* whose ``utf16_len`` is at most *limit*."""
    units = 0
    for i, ch in enumerate(text):
        units += 2 if ord(ch) > 0xFFFF else 1
        if units > limit:
            return text[:i]
    return text


def _node_cost(node: dict) -> int:
    return sum(utf16_len(node[k]) for k in ("id", "title", "type", "space", "content"))


def _item_cost(batch: str, item) -> int:
    if batch == CLUSTER_BATCH:
        return _CLUSTER_OVERHEAD + sum(_CLUSTER_NODE_OVERHEAD + _node_cost(n) for n in item)
    return _PAIR_OVERHEAD + _node_cost(item["node_a"]) + _node_cost(item["node_b"])


def max_item_cost(cluster_max_nodes: int) -> int:
    """Largest cost one item can have at the field caps, its section headers included.

    A pair can occur in all three pair batches and is emitted as one group.
    """
    pair_group = sum(
        _SECTION_OVERHEAD[name] + _PAIR_OVERHEAD + 2 * _NODE_MAX_CHARS for name in PAIR_BATCHES
    )
    cluster = (
        _SECTION_OVERHEAD[CLUSTER_BATCH]
        + _CLUSTER_OVERHEAD
        + cluster_max_nodes * (_CLUSTER_NODE_OVERHEAD + _NODE_MAX_CHARS)
    )
    return max(pair_group, cluster)


def _pair_key(pair: dict) -> tuple[str, str]:
    a, b = sorted((pair["node_a"]["id"], pair["node_b"]["id"]))
    return a, b


def fit_to_budget(batches: dict[str, list], budget: int) -> dict[str, list]:
    """Return the batches cut to *budget* characters of estimated formatted size.

    Items are taken in turns (link, conflict, merge, cluster). When the next item
    of a batch does not fit, that batch leaves the turn and the others continue.
    All occurrences of one unordered pair across the pair batches are emitted
    together or cut together, because one verdict marks the pair checked for all
    three finders. Order inside each batch is kept.
    """
    order = (*PAIR_BATCHES, CLUSTER_BATCH)
    groups: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for name in PAIR_BATCHES:
        for i, pair in enumerate(batches[name]):
            groups.setdefault(_pair_key(pair), []).append((name, i))

    emitted: dict[str, set[int]] = {name: set() for name in order}
    cursor = dict.fromkeys(order, 0)
    active = [name for name in order if batches[name]]
    remaining = budget
    while active:
        for name in list(active):
            items = batches[name]
            while cursor[name] < len(items) and cursor[name] in emitted[name]:
                cursor[name] += 1  # already emitted with its pair group
            if cursor[name] >= len(items):
                active.remove(name)
                continue
            i = cursor[name]
            occurrences = [(name, i)] if name == CLUSTER_BATCH else groups[_pair_key(items[i])]
            cost = sum(_item_cost(b, batches[b][j]) for b, j in occurrences)
            cost += sum(_SECTION_OVERHEAD[b] for b in {b for b, _ in occurrences} if not emitted[b])
            if cost > remaining:
                active.remove(name)
                continue
            remaining -= cost
            for b, j in occurrences:
                emitted[b].add(j)
            cursor[name] += 1

    return {
        name: [item for i, item in enumerate(batches[name]) if i in emitted[name]]
        for name in order
    }
