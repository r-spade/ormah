"""Load and validate whisper eval corpus files (JSONL format)."""
from __future__ import annotations

import json
from pathlib import Path

from eval.whisper.contract import (
    ADJUDICATION_STATUSES,
    BINDING_ADJUDICATION_STATUSES,
    CORPUS_PARTITIONS,
    SCHEMA_VERSION,
    TARGET_DECISIONS,
)

# Kept for report ordering and examples. The eval harness should not be
# restricted to these categories.
VALID_CATEGORIES = frozenset({
    "preference", "factual", "decision", "technical",
    "identity", "temporal", "noise", "continuation",
})


class CorpusError(Exception):
    """Raised on corpus file or validation errors."""


def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus file. Skips blank lines. Validates each case."""
    if not path.exists():
        raise CorpusError(f"Corpus file not found: {path}")
    cases = []
    seen_case_ids: set[str] = set()
    schema_versions: set[str | None] = set()
    dataset_identities: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise CorpusError(f"Line {line_number} must contain a case object")
        try:
            validate_case(obj)
        except CorpusError as exc:
            raise CorpusError(f"Line {line_number}: {exc}") from exc
        case_id = obj.get("id")
        if case_id in seen_case_ids:
            raise CorpusError(f"Line {line_number}: duplicate case id '{case_id}'")
        seen_case_ids.add(case_id)
        schema_versions.add(obj.get("schema_version"))
        if obj.get("schema_version") == SCHEMA_VERSION:
            corpus = obj["corpus"]
            dataset_identities.add((corpus["dataset_version"], corpus["partition"]))
        cases.append(obj)
    if len(schema_versions) > 1:
        raise CorpusError("A corpus file cannot mix legacy and schema v2 cases")
    if len(dataset_identities) > 1:
        raise CorpusError(
            "All schema v2 cases in a file must use the same dataset_version and partition"
        )
    return cases


def filter_binding_cases(cases: list[dict], *, include_provisional: bool) -> tuple[list[dict], int]:
    """Return cases/prompts whose labels may bind, plus the skipped prompt count."""
    filtered: list[dict] = []
    skipped = 0
    for case in cases:
        if case.get("provisional") and not include_provisional:
            skipped += len(case.get("prompts", []))
            continue
        if case.get("schema_version") != SCHEMA_VERSION or include_provisional:
            filtered.append(case)
            continue
        prompts = []
        for prompt in case.get("prompts", []):
            status = prompt["expected"]["adjudication"]["status"]
            if status in BINDING_ADJUDICATION_STATUSES:
                prompts.append(prompt)
            else:
                skipped += 1
        if prompts:
            filtered.append({**case, "prompts": prompts})
    return filtered, skipped


def validate_case(case: dict) -> None:
    """Validate a single corpus case. Raises CorpusError on structural issues."""
    case_id = case.get("id", "<unknown>")
    seen_ids: set[str] = set()

    schema_version = case.get("schema_version")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        raise CorpusError(
            f"Case '{case_id}' uses unsupported schema_version '{schema_version}'; "
            f"expected '{SCHEMA_VERSION}'"
        )
    if schema_version == SCHEMA_VERSION:
        if not isinstance(case.get("id"), str) or not case["id"].strip():
            raise CorpusError("Schema v2 requires a non-empty case id")
        if not isinstance(case.get("memories"), list) or not case["memories"]:
            raise CorpusError(f"Case '{case_id}' schema v2 requires non-empty memories")
        if not isinstance(case.get("prompts"), list) or not case["prompts"]:
            raise CorpusError(f"Case '{case_id}' schema v2 requires non-empty prompts")
        _validate_v2_case_metadata(case, case_id)

    for i, mem in enumerate(case.get("memories", [])):
        if not isinstance(mem, dict):
            raise CorpusError(f"Case '{case_id}' memory[{i}] must be an object")
        node_id = mem.get("node_id")
        if not node_id:
            raise CorpusError(f"Case '{case_id}' memory[{i}] missing 'node_id' field")
        if schema_version == SCHEMA_VERSION and (
            not isinstance(node_id, str) or not node_id.strip()
        ):
            raise CorpusError(
                f"Case '{case_id}' schema v2 memory[{i}] node_id must be a non-empty string"
            )
        if node_id in seen_ids:
            raise CorpusError(f"Case '{case_id}' has duplicate node_id: '{node_id}'")
        seen_ids.add(node_id)

    # Validate optional in-case connections (enables spread activation / identity graph evals).
    from ormah.models.node import EdgeType
    for i, mem in enumerate(case.get("memories", [])):
        for j, conn in enumerate(mem.get("connections", []) or []):
            if not isinstance(conn, dict):
                raise CorpusError(f"Case '{case_id}' memory[{i}] connections[{j}] must be an object")
            target = conn.get("target")
            if not target:
                raise CorpusError(f"Case '{case_id}' memory[{i}] connections[{j}] missing 'target'")
            if target not in seen_ids:
                raise CorpusError(
                    f"Case '{case_id}' memory[{i}] connections[{j}] references unknown node_id '{target}'"
                )
            edge = conn.get("edge", "related_to")
            try:
                EdgeType(edge)
            except Exception:
                raise CorpusError(
                    f"Case '{case_id}' memory[{i}] connections[{j}] has invalid edge '{edge}'. "
                    f"Valid: {[e.value for e in EdgeType]}"
                )

    seen_turn_ids: set[str] = set()
    for i, prompt in enumerate(case.get("prompts", [])):
        if not isinstance(prompt, dict):
            raise CorpusError(f"Case '{case_id}' prompt[{i}] must be an object")
        category = prompt.get("category")
        if category is not None:
            if not isinstance(category, str) or not category.strip():
                raise CorpusError(f"Case '{case_id}' prompt[{i}] category must be a non-empty string")
        expected = prompt.get("expected", {})
        if schema_version == SCHEMA_VERSION:
            _validate_v2_prompt(prompt, expected, case_id, i, seen_ids, seen_turn_ids)
        # Backwards compatible fields: should_inject / should_not_inject / should_suppress
        # Newer fields: must_include / may_include / must_not_include / must_be_silent
        id_fields = (
            "should_inject",
            "should_not_inject",
            "must_include",
            "may_include",
            "must_not_include",
        )
        for label_field in id_fields:
            for nid in expected.get(label_field, []):
                if nid not in seen_ids:
                    raise CorpusError(
                        f"Case '{case_id}' prompt[{i}] references unknown node_id "
                        f"'{nid}' in '{label_field}'"
                    )


def _validate_v2_case_metadata(case: dict, case_id: str) -> None:
    corpus = case.get("corpus")
    if not isinstance(corpus, dict):
        raise CorpusError(f"Case '{case_id}' schema v2 requires a 'corpus' object")
    dataset_version = corpus.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise CorpusError(f"Case '{case_id}' schema v2 requires corpus.dataset_version")
    partition = corpus.get("partition")
    if partition not in CORPUS_PARTITIONS:
        raise CorpusError(
            f"Case '{case_id}' has invalid corpus.partition '{partition}'; "
            f"expected one of {sorted(CORPUS_PARTITIONS)}"
        )


def _validate_v2_prompt(
    prompt: dict,
    expected: dict,
    case_id: str,
    prompt_index: int,
    seen_ids: set[str],
    seen_turn_ids: set[str],
) -> None:
    location = f"Case '{case_id}' prompt[{prompt_index}]"
    text = prompt.get("text")
    if not isinstance(text, str) or not text.strip():
        raise CorpusError(f"{location} schema v2 requires non-empty text")
    if not isinstance(expected, dict):
        raise CorpusError(f"{location} schema v2 requires an expected object")
    turn_id = prompt.get("id")
    if not isinstance(turn_id, str) or not turn_id.strip():
        raise CorpusError(f"{location} schema v2 requires a non-empty prompt id")
    if turn_id in seen_turn_ids:
        raise CorpusError(f"{location} duplicates prompt id '{turn_id}'")
    seen_turn_ids.add(turn_id)
    legacy_fields = {
        "should_inject",
        "should_not_inject",
        "should_suppress",
        "must_be_silent",
    }
    present_legacy = sorted(legacy_fields & expected.keys())
    if present_legacy:
        raise CorpusError(
            f"{location} schema v2 cannot use legacy expectation fields: {present_legacy}"
        )
    target = expected.get("target_decision")
    if target not in TARGET_DECISIONS:
        raise CorpusError(
            f"{location} has invalid target_decision '{target}'; "
            f"expected one of {sorted(TARGET_DECISIONS)}"
        )
    if target == "abstain" and expected.get("must_include"):
        raise CorpusError(f"{location} cannot require nodes when target_decision is 'abstain'")
    if target in {"inject", "ask_qualify"} and not expected.get("must_include"):
        raise CorpusError(f"{location} target_decision '{target}' requires must_include nodes")

    labeled_id_fields = (
        "must_include",
        "may_include",
        "must_not_include",
    )
    for field in labeled_id_fields:
        value = expected.get(field)
        if not isinstance(value, list):
            raise CorpusError(f"{location} schema v2 requires expected.{field} as a list")
        if len(value) != len(set(value)):
            raise CorpusError(f"{location} expected.{field} contains duplicate node IDs")
        if not all(isinstance(node_id, str) and node_id.strip() for node_id in value):
            raise CorpusError(f"{location} expected.{field} node IDs must be non-empty strings")

    eligible = expected.get("eligible_node_ids")
    if not isinstance(eligible, list):
        raise CorpusError(f"{location} schema v2 requires expected.eligible_node_ids")
    if not all(isinstance(node_id, str) and node_id.strip() for node_id in eligible):
        raise CorpusError(
            f"{location} expected.eligible_node_ids must contain non-empty strings"
        )
    eligible_set = set(eligible)
    if len(eligible) != len(eligible_set):
        raise CorpusError(f"{location} expected.eligible_node_ids contains duplicate node IDs")
    unknown_eligible = eligible_set - seen_ids
    if unknown_eligible:
        raise CorpusError(
            f"{location} eligible_node_ids reference unknown nodes: {sorted(unknown_eligible)}"
        )
    required = set(expected["must_include"])
    acceptable = set(expected["may_include"])
    forbidden = set(expected["must_not_include"])
    if required - eligible_set or acceptable - eligible_set:
        raise CorpusError(
            f"{location} required and acceptable nodes must be present in eligible_node_ids"
        )
    if required & acceptable or required & forbidden or acceptable & forbidden:
        raise CorpusError(f"{location} required, acceptable, and forbidden node IDs must be disjoint")

    rationale = expected.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise CorpusError(f"{location} schema v2 requires expected.rationale")
    labels = expected.get("labels")
    if not isinstance(labels, list) or not labels or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise CorpusError(f"{location} schema v2 requires non-empty expected.labels")

    adjudication = expected.get("adjudication")
    if not isinstance(adjudication, dict):
        raise CorpusError(f"{location} schema v2 requires expected.adjudication")
    status = adjudication.get("status")
    reviewer_count = adjudication.get("reviewer_count")
    if status not in ADJUDICATION_STATUSES:
        raise CorpusError(
            f"{location} has invalid adjudication status '{status}'; "
            f"expected one of {sorted(ADJUDICATION_STATUSES)}"
        )
    if not isinstance(reviewer_count, int) or isinstance(reviewer_count, bool) or reviewer_count < 1:
        raise CorpusError(f"{location} adjudication.reviewer_count must be a positive integer")
    if status == "adjudicated" and reviewer_count < 2:
        raise CorpusError(f"{location} adjudicated labels require at least two reviewers")
