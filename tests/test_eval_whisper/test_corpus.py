"""Tests for eval/whisper/corpus.py."""
from __future__ import annotations
import json
import pytest
from eval.whisper.contract import SCHEMA_VERSION
from eval.whisper.corpus import (
    CorpusError,
    VALID_CATEGORIES,
    filter_binding_cases,
    load_corpus,
    validate_case,
)


def _write_jsonl(tmp_path, cases):
    f = tmp_path / "test.jsonl"
    f.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    return f


_MEM = {"node_id": "m-1", "title": "T", "content": "C", "type": "fact", "tier": "working"}
_PROMPT = {"text": "q", "category": "factual", "expected": {"should_inject": ["m-1"], "should_suppress": False}}
_VALID = {"id": "w-001", "memories": [_MEM], "prompts": [_PROMPT]}
_V2_VALID = {
    "schema_version": SCHEMA_VERSION,
    "id": "w-v2-001",
    "corpus": {"dataset_version": "test-1", "partition": "public_regression"},
    "memories": [_MEM],
    "prompts": [{
        "id": "turn-1",
        "text": "q",
        "category": "factual",
        "expected": {
            "target_decision": "inject",
            "eligible_node_ids": ["m-1"],
            "must_include": ["m-1"],
            "may_include": [],
            "must_not_include": [],
            "rationale": "The node answers the question.",
            "labels": ["contract_test"],
            "adjudication": {"status": "adjudicated", "reviewer_count": 2},
        },
    }],
}


class TestLoadCorpus:
    def test_loads_cases(self, tmp_path):
        f = _write_jsonl(tmp_path, [_VALID])
        cases = load_corpus(f)
        assert len(cases) == 1
        assert cases[0]["id"] == "w-001"

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "t.jsonl"
        second = {**_VALID, "id": "w-002"}
        f.write_text(json.dumps(_VALID) + "\n\n" + json.dumps(second) + "\n")
        assert len(load_corpus(f)) == 2

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(CorpusError, match="not found"):
            load_corpus(tmp_path / "missing.jsonl")

    def test_reports_invalid_json_with_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n")
        with pytest.raises(CorpusError, match="Invalid JSON on line 1"):
            load_corpus(path)

    def test_rejects_duplicate_case_ids(self, tmp_path):
        path = _write_jsonl(tmp_path, [_VALID, _VALID])
        with pytest.raises(CorpusError, match="duplicate case id"):
            load_corpus(path)

    def test_rejects_mixed_schema_modes(self, tmp_path):
        path = _write_jsonl(tmp_path, [_VALID, _V2_VALID])
        with pytest.raises(CorpusError, match="cannot mix legacy and schema v2"):
            load_corpus(path)

class TestValidateCase:
    def test_valid_case_passes(self):
        validate_case(_VALID)  # no exception

    def test_valid_v2_case_passes(self):
        validate_case(_V2_VALID)

    def test_public_contract_fixture_loads(self):
        path = (
            __import__("pathlib").Path(__file__).parents[2]
            / "eval/whisper/corpus/public/contract-smoke-v2.jsonl"
        )
        cases = load_corpus(path)
        assert cases[0]["schema_version"] == SCHEMA_VERSION

    @pytest.mark.parametrize("decision", ["inject", "abstain", "ask_qualify"])
    def test_v2_accepts_all_target_decisions(self, decision):
        case = json.loads(json.dumps(_V2_VALID))
        expected = case["prompts"][0]["expected"]
        expected["target_decision"] = decision
        if decision == "abstain":
            expected["must_include"] = []
        validate_case(case)

    def test_v2_rejects_missing_rationale(self):
        case = json.loads(json.dumps(_V2_VALID))
        del case["prompts"][0]["expected"]["rationale"]
        with pytest.raises(CorpusError, match="requires expected.rationale"):
            validate_case(case)

    def test_v2_rejects_single_reviewer_adjudication(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["prompts"][0]["expected"]["adjudication"]["reviewer_count"] = 1
        with pytest.raises(CorpusError, match="at least two reviewers"):
            validate_case(case)

    def test_v2_rejects_unknown_schema_version(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["schema_version"] = "99"
        with pytest.raises(CorpusError, match="unsupported schema_version"):
            validate_case(case)

    def test_v2_rejects_overlapping_node_labels(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["prompts"][0]["expected"]["must_not_include"] = ["m-1"]
        with pytest.raises(CorpusError, match="must be disjoint"):
            validate_case(case)

    def test_v2_requires_required_nodes_to_be_eligible(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["prompts"][0]["expected"]["eligible_node_ids"] = []
        with pytest.raises(CorpusError, match="must be present in eligible_node_ids"):
            validate_case(case)

    def test_v2_rejects_legacy_expectation_aliases(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["prompts"][0]["expected"]["should_inject"] = ["m-1"]
        with pytest.raises(CorpusError, match="cannot use legacy expectation fields"):
            validate_case(case)

    def test_v2_rejects_unhashable_memory_node_id_as_corpus_error(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["memories"][0]["node_id"] = ["not", "an", "id"]
        with pytest.raises(CorpusError, match="node_id must be a non-empty string"):
            validate_case(case)

    def test_missing_node_id_raises(self):
        bad = {"id": "x", "memories": [{"title": "T"}], "prompts": []}
        with pytest.raises(CorpusError, match="missing 'node_id'"):
            validate_case(bad)

    def test_duplicate_node_id_raises(self):
        bad = {
            "id": "x",
            "memories": [
                {"node_id": "dup", "title": "A", "content": "C", "type": "fact", "tier": "working"},
                {"node_id": "dup", "title": "B", "content": "C", "type": "fact", "tier": "working"},
            ],
            "prompts": [],
        }
        with pytest.raises(CorpusError, match="duplicate node_id"):
            validate_case(bad)

    def test_category_is_freeform_string(self):
        ok = {
            "id": "x",
            "memories": [_MEM],
            "prompts": [{"text": "q", "category": "my_custom_bucket", "expected": {"should_inject": ["m-1"]}}],
        }
        validate_case(ok)  # no exception

    def test_empty_category_raises(self):
        bad = {
            "id": "x",
            "memories": [_MEM],
            "prompts": [{"text": "q", "category": "   ", "expected": {}}],
        }
        with pytest.raises(CorpusError, match="category must be a non-empty string"):
            validate_case(bad)

    def test_unknown_node_ref_in_should_inject_raises(self):
        bad = {
            "id": "x",
            "memories": [_MEM],
            "prompts": [{"text": "q", "category": "factual", "expected": {"should_inject": ["unknown-id"]}}],
        }
        with pytest.raises(CorpusError, match="unknown node_id"):
            validate_case(bad)

    def test_new_expectation_fields_are_accepted(self):
        case = {
            "id": "x",
            "memories": [_MEM],
            "prompts": [
                {
                    "text": "q",
                    "category": "factual",
                    "expected": {
                        "must_include": ["m-1"],
                        "may_include": ["m-1"],
                        "must_not_include": [],
                        "must_be_silent": False,
                    },
                }
            ],
        }
        validate_case(case)  # no exception

    def test_unknown_node_ref_in_must_not_include_raises(self):
        bad = {
            "id": "x",
            "memories": [_MEM],
            "prompts": [{"text": "q", "category": "factual", "expected": {"must_not_include": ["unknown-id"]}}],
        }
        with pytest.raises(CorpusError, match="unknown node_id"):
            validate_case(bad)

    def test_unknown_node_ref_in_may_include_raises(self):
        bad = {
            "id": "x",
            "memories": [_MEM],
            "prompts": [{"text": "q", "category": "factual", "expected": {"may_include": ["unknown-id"]}}],
        }
        with pytest.raises(CorpusError, match="unknown node_id"):
            validate_case(bad)

    def test_valid_connections_are_accepted(self):
        case = {
            "id": "x",
            "memories": [
                {**_MEM, "node_id": "m-1", "connections": [{"target": "m-2", "edge": "supports"}]},
                {"node_id": "m-2", "title": "T2", "content": "C2", "type": "fact", "tier": "working"},
            ],
            "prompts": [],
        }
        validate_case(case)  # no exception

    def test_connection_target_must_exist(self):
        bad = {
            "id": "x",
            "memories": [{**_MEM, "connections": [{"target": "missing", "edge": "supports"}]}],
            "prompts": [],
        }
        with pytest.raises(CorpusError, match="references unknown node_id"):
            validate_case(bad)

    def test_connection_edge_must_be_valid(self):
        bad = {
            "id": "x",
            "memories": [
                {**_MEM, "connections": [{"target": "m-2", "edge": "bogus"}]},
                {"node_id": "m-2", "title": "T2", "content": "C2", "type": "fact", "tier": "working"},
            ],
            "prompts": [],
        }
        with pytest.raises(CorpusError, match="invalid edge"):
            validate_case(bad)

    def test_connection_entry_must_be_object(self):
        bad = {
            "id": "x",
            "memories": [{**_MEM, "connections": ["m-2"]}],
            "prompts": [],
        }
        with pytest.raises(CorpusError, match="must be an object"):
            validate_case(bad)

    def test_all_valid_categories_accepted(self):
        for cat in VALID_CATEGORIES:
            case = {
                "id": "x", "memories": [_MEM],
                "prompts": [{"text": "q", "category": cat, "expected": {"should_inject": ["m-1"]}}],
            }
            validate_case(case)  # no exception


class TestBindingLabels:
    def test_draft_v2_prompts_do_not_bind_by_default(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["prompts"][0]["expected"]["adjudication"]["status"] = "draft"
        filtered, skipped = filter_binding_cases([case], include_provisional=False)
        assert filtered == []
        assert skipped == 1

    def test_draft_v2_prompts_can_run_as_explicit_non_binding_smoke(self):
        case = json.loads(json.dumps(_V2_VALID))
        case["prompts"][0]["expected"]["adjudication"]["status"] = "draft"
        filtered, skipped = filter_binding_cases([case], include_provisional=True)
        assert filtered == [case]
        assert skipped == 0
