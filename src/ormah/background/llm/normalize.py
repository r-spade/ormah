"""Normalize LLM responses to canonical edge/conflict types."""

from __future__ import annotations

_CONFLICT_TYPE_ALIASES: dict[str, str] = {
    "evolution": "evolution",
    "tension": "tension",
    "none": "none",
    # Common LLM misnames
    "disagreement": "tension",
    "contradiction": "tension",
    "conflict": "tension",
    "change": "evolution",
    "update": "evolution",
    "revised": "evolution",
    "refinement": "evolution",
}

_LINK_TYPE_ALIASES: dict[str, str] = {
    "supports": "supports",
    "contradicts": "contradicts",
    "related_to": "related_to",
    "none": "none",
    # Common LLM misnames
    "support": "supports",
    "contradict": "contradicts",
    "contradiction": "contradicts",
    "related": "related_to",
    "relevant": "related_to",
    "part_of": "part_of",
    "derived_from": "derived_from",
    "depends_on": "depends_on",
    "dependency": "depends_on",
}


def normalize_conflict_type(raw: str) -> str:
    """Map a raw LLM conflict type to a canonical value.

    Unknown values default to ``"tension"``.
    """
    return _CONFLICT_TYPE_ALIASES.get(raw.lower().strip(), "tension")


def normalize_link_type(raw: str) -> str:
    """Map a raw LLM link type to a canonical value.

    Unknown values default to ``"related_to"``.
    """
    return _LINK_TYPE_ALIASES.get(raw.lower().strip(), "related_to")
