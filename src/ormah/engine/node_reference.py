"""Canonical resolution of a caller's node reference to one Full id (#280).

A node's identity is its Full id, the 36-character UUID. Its Short id, the first 8
characters, is what older Ormah output showed agents, so agents and old transcripts
still hand it back. It is not unique: two memories can share it, and the chance that
some pair in a store does grows with the square of the store's size. A Short id is
therefore only an input alias, and every entry point resolves it here, once, before
any side effect. Everything written afterwards names the Full id.

Resolution has three outcomes, and callers must keep them apart:

- resolved: exactly one memory has this reference; `full_id` is its Full id.
- absent: no memory has it. Safe to report as "not found".
- unresolved: Ormah cannot tell which memory it names — an ambiguous Short id, a
  candidate file that will not parse, or a reference too short to be an alias.
  This is not absence. A caller must not act on it, and must not pick a candidate.

Accepted references, and why:

- A Full id, or any id the index holds verbatim (exact match always wins).
- A prefix of at least 8 characters. The index must find at most one node under it.
  For exactly the 8-character Short id the file store must agree as well, because a
  file that is saved but not yet indexed can share it; `FileStore.resolve` reads
  only that width, since the Short id is what filenames carry.
- Nothing shorter. `GraphIndex.resolve_node` would answer any unique prefix, but a
  4-character prefix that happens to be unique today is a guess, the file store
  cannot vouch for it, and Ormah has never shown an agent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ormah.store.file_store import UnresolvedNodeReference

if TYPE_CHECKING:
    from ormah.index.graph import GraphIndex
    from ormah.store.file_store import FileStore

SHORT_ID_LENGTH = 8
FULL_ID_LENGTH = 36

# SQLite LIKE metacharacters. No node id contains them, and the index answers a prefix
# with LIKE, so letting one through would turn a reference into a pattern.
_LIKE_METACHARACTERS = frozenset("%_\\")


@dataclass(frozen=True)
class NodeResolution:
    """What a caller's reference names. See the module docstring for the three outcomes."""

    reference: str
    full_id: str | None = None
    problem: str | None = None

    @property
    def absent(self) -> bool:
        """True only for a confirmed absence, never for "could not tell"."""
        return self.full_id is None and self.problem is None


def resolve_node_reference(
    graph: GraphIndex, file_store: FileStore, reference: str
) -> NodeResolution:
    """Resolve *reference* to the one Full id it names, or say why it does not.

    Read-only. Takes no lock beyond the ones the index and the store take for
    themselves, so it is safe both inside a serialized engine operation and in front of
    one. An OSError from the store propagates: not being able to read is not absence.
    """
    ref = reference.strip() if isinstance(reference, str) else ""
    if not ref:
        return NodeResolution(reference, problem="The node reference is empty.")

    exact = graph.conn.execute("SELECT id FROM nodes WHERE id = ?", (ref,)).fetchone()
    if exact is not None:
        return NodeResolution(reference, full_id=exact["id"])

    if len(ref) >= FULL_ID_LENGTH:
        # A Full id the index does not hold. Only the store can still know it: a file
        # saved but not yet indexed.
        try:
            stored = file_store.resolve(ref)
        except UnresolvedNodeReference as exc:
            return NodeResolution(reference, problem=str(exc))
        return NodeResolution(reference, full_id=stored.id if stored else None)

    if len(ref) < SHORT_ID_LENGTH:
        return NodeResolution(
            reference,
            problem=(
                f"Node reference {ref} is too short to name one memory: "
                "use its Full id."
            ),
        )
    if any(ch in _LIKE_METACHARACTERS for ch in ref):
        return NodeResolution(reference)

    candidates: set[str] = set()
    try:
        indexed = graph.resolve_node(ref)
    except UnresolvedNodeReference as exc:
        return NodeResolution(reference, problem=str(exc))
    if indexed is not None:
        candidates.add(indexed["id"])
    if len(ref) == SHORT_ID_LENGTH:
        try:
            stored = file_store.resolve(ref)
        except UnresolvedNodeReference as exc:
            return NodeResolution(reference, problem=str(exc))
        if stored is not None:
            candidates.add(stored.id)

    if len(candidates) > 1:
        return NodeResolution(
            reference,
            problem=(
                f"Node reference {ref} is ambiguous: it matches "
                f"{', '.join(sorted(candidates))}. Retry with one of these Full ids."
            ),
        )
    if not candidates:
        return NodeResolution(reference)
    return NodeResolution(reference, full_id=candidates.pop())
