import math

import pytest

from eval.bench.judge import LONGMEMEVAL_RULES, judge_answer, judge_prompt
from eval.bench.metrics import aggregate, retrieval_metrics
from eval.bench.providers import TextResult
from unittest.mock import Mock


def question(dataset="longmemeval", abstention=False):
    return dict(
        question_id="q",
        dataset=dataset,
        question="Where?",
        gold="London",
        question_type="multi-session",
        question_date="2023-01-01",
        gold_ids=["s1", "s2"],
        abstention=abstention,
    )


def retrieval():
    return {
        "ranked": [
            {"node": {"tags": ["session:s1", "turn:s1"]}, "score": 0.9},
            {"node": {"tags": ["session:s1", "turn:s1"]}, "score": 0.8},
            {"node": {"tags": ["session:noise", "turn:noise"]}, "score": 0.5},
            {"node": {"tags": ["session:s2", "turn:s2"]}, "score": 0.1},
        ],
        "production_gate": 0.35,
        "latency_s": 0.1,
    }


def test_session_recall_ndcg_unique_and_gate():
    metrics = retrieval_metrics(question(), retrieval(), 4)
    assert metrics["recall"] == 1
    assert metrics["ndcg"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert retrieval_metrics(question(), retrieval(), 4, gated=True)["recall"] == 0.5
    assert retrieval_metrics(question(), retrieval(), 1)["recall"] == 0.5


def test_abstention_excluded_and_no_turn_provenance():
    assert retrieval_metrics(question(abstention=True), retrieval(), 4)["recall"] is None
    assert retrieval_metrics(question("locomo"), retrieval(), 4, "extract")["recall"] is None
    assert retrieval_metrics(question("locomo"), retrieval(), 4)["recall"] == 1


def test_locomo_jscore_excludes_category5():
    normal = {
        **question("locomo"),
        "question_type": "4",
        "judge": {"status": "ok", "result": {"correct": True}},
    }
    abstention = {
        **question("locomo", True),
        "question_type": "5",
        "judge": {"status": "ok", "result": {"correct": False}},
    }
    result = aggregate([normal, abstention], 30, "raw")
    assert result["accuracy"] == 1
    assert result["scored"] == 1
    assert result["abstention_accuracy"] == 0


@pytest.mark.parametrize("kind", list(LONGMEMEVAL_RULES))
def test_official_judge_dispatch(kind):
    q = question()
    q["question_type"] = kind
    assert judge_prompt(q, "answer") == LONGMEMEVAL_RULES[kind].format(
        q["question"], q["gold"], "answer"
    )
    q["abstention"] = True
    assert "unanswerable question" in judge_prompt(q, "answer")


def test_invalid_judge_is_error_not_negative():
    provider = Mock()
    provider.complete.return_value = TextResult("maybe yes or no", "fake")
    with pytest.raises(ValueError):
        judge_answer(provider, question(), "answer")


def test_answer_prompt_has_dates_but_no_gold():
    from eval.bench.answer import answer_prompt

    q = question()
    q["gold"] = "HIDDEN GOLD SENTINEL"
    prompt = answer_prompt(
        q,
        [
            {
                "node": {
                    "created": "2000-01-01",
                    "title": "London trip",
                    "content": "Ada visited London.",
                }
            }
        ],
    )
    assert "HIDDEN GOLD SENTINEL" not in prompt
    assert "2023-01-01" in prompt
    assert "2000-01-01" in prompt
    assert "Ada visited London." in prompt


def test_locomo_category5_evidence_counts_for_retrieval_not_jscore():
    q = question("locomo", abstention=True)
    q["question_type"] = "5"
    assert retrieval_metrics(q, retrieval(), 4)["recall"] == 1
