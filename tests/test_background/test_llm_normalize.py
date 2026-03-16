"""Tests for LLM response normalization functions."""

from __future__ import annotations

from ormah.background.llm.normalize import normalize_conflict_type, normalize_link_type


# --- normalize_conflict_type ---

def test_canonical_conflict_types_pass_through():
    assert normalize_conflict_type("evolution") == "evolution"
    assert normalize_conflict_type("tension") == "tension"
    assert normalize_conflict_type("none") == "none"


def test_conflict_aliases_map_correctly():
    assert normalize_conflict_type("disagreement") == "tension"
    assert normalize_conflict_type("contradiction") == "tension"
    assert normalize_conflict_type("conflict") == "tension"
    assert normalize_conflict_type("change") == "evolution"
    assert normalize_conflict_type("update") == "evolution"
    assert normalize_conflict_type("revised") == "evolution"
    assert normalize_conflict_type("refinement") == "evolution"


def test_unknown_conflict_type_defaults_to_tension():
    assert normalize_conflict_type("banana") == "tension"
    assert normalize_conflict_type("something_else") == "tension"


def test_conflict_type_case_insensitive():
    assert normalize_conflict_type("EVOLUTION") == "evolution"
    assert normalize_conflict_type("Disagreement") == "tension"
    assert normalize_conflict_type("  tension  ") == "tension"


# --- normalize_link_type ---

def test_canonical_link_types_pass_through():
    assert normalize_link_type("supports") == "supports"
    assert normalize_link_type("contradicts") == "contradicts"
    assert normalize_link_type("related_to") == "related_to"
    assert normalize_link_type("none") == "none"


def test_link_aliases_map_correctly():
    assert normalize_link_type("support") == "supports"
    assert normalize_link_type("contradict") == "contradicts"
    assert normalize_link_type("contradiction") == "contradicts"
    assert normalize_link_type("related") == "related_to"
    assert normalize_link_type("relevant") == "related_to"


def test_unknown_link_type_defaults_to_related_to():
    assert normalize_link_type("banana") == "related_to"
    assert normalize_link_type("xyz") == "related_to"


def test_link_type_case_insensitive():
    assert normalize_link_type("SUPPORTS") == "supports"
    assert normalize_link_type("Contradicts") == "contradicts"
    assert normalize_link_type("  related_to  ") == "related_to"
