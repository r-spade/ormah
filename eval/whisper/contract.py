"""Versioned measurement contract for proactive whisper evaluation.

This module deliberately describes labels and measurements, not retrieval policy.
Keeping that boundary explicit lets experiments change ranking without changing the
yardstick at the same time.
"""
from __future__ import annotations

SCHEMA_VERSION = "2.0"

TARGET_DECISIONS = frozenset({"inject", "abstain", "ask_qualify"})
CORPUS_PARTITIONS = frozenset({"public_regression", "adversarial", "replay", "holdout"})
ADJUDICATION_STATUSES = frozenset({"draft", "reviewed", "adjudicated"})
BINDING_ADJUDICATION_STATUSES = frozenset({"reviewed", "adjudicated"})


def infer_target_decision(expected: dict) -> str:
    """Return the v2 decision label, with a legacy-corpus compatibility path."""
    target = expected.get("target_decision")
    if target:
        return target
    if expected.get("must_be_silent", expected.get("should_suppress", False)):
        return "abstain"
    return "inject"


def corpus_schema_mode(cases: list[dict]) -> str:
    """Identify whether a loaded corpus uses legacy compatibility or v2 labels."""
    versions = {case.get("schema_version") for case in cases}
    if versions == {SCHEMA_VERSION}:
        return SCHEMA_VERSION
    if versions <= {None}:
        return "legacy_compat"
    return "mixed"
