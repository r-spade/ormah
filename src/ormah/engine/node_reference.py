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

- A Full id, or any id the index holds verbatim (exact match always wins). Case is
  ignored: ids are stored lower-case, and the index's prefix match ignores case
  anyway, so an upper-case Full id must not be the one form that fails.
- A prefix of at least 8 characters. The index and the file store must together
  find exactly one node under it, at every width: a file that is saved but not yet
  indexed can share any prefix, not only the 8-character Short id.
- Nothing that is not an id. A reference carrying SQL LIKE or glob syntax is
  reported absent without being looked up.
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

# SQLite LIKE and glob syntax. No node id contains these, the index answers a prefix
# with LIKE and the store with a glob, so letting one through would turn a reference
# into a pattern.
PATTERN_CHARACTERS = frozenset("%_\\*?[]")


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

    for exact_form in dict.fromkeys((ref, ref.lower())):
        exact = graph.conn.execute(
            "SELECT id FROM nodes WHERE id = ?", (exact_form,)
        ).fetchone()
        if exact is not None:
            return NodeResolution(reference, full_id=exact["id"])
    ref = ref.lower()

    if any(ch in PATTERN_CHARACTERS for ch in ref):
        return NodeResolution(reference)

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

    candidates: set[str] = set()
    try:
        indexed = graph.resolve_node(ref)
        if indexed is not None:
            candidates.add(indexed["id"])
        # The store is asked at every width, not only for the bare Short id: a file
        # saved but not yet indexed can share a longer prefix just as well.
        candidates.update(file_store.ids_with_prefix(ref))
    except UnresolvedNodeReference as exc:
        return NodeResolution(reference, problem=str(exc))

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
