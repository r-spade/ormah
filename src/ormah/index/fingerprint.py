"""Fingerprint of the node fields that can change a linking decision (#126)."""

from __future__ import annotations

import hashlib


def content_fingerprint(
    title: str | None, content: str, node_type: str, space: str | None
) -> str:
    """sha256 over the fields a linking decision actually depends on.

    `title`/`content` feed the embedding (`embedding_text`) and the LLM judge prompt;
    `type` is shown to the judge; `space` is shown to the judge AND drives
    `cross_space_penalty` during candidate selection. Anything else — connections, tags,
    tier, importance, access_count — cannot change what the linker would decide, so it must
    not requeue the node (#126).

    The separator is a NUL byte: it cannot occur in any of these fields, so
    ("ab", "c") and ("a", "bc") cannot collide.
    """
    parts = [title or "", content, node_type, space or ""]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
