import json

import pytest

from eval.bench.compare import compare_runs


def _run(path, answers):
    path.mkdir()
    rows = []
    for qid, conversation, stratum, correct, chars, silent in answers:
        rows.append(
            {
                "question_id": qid,
                "dataset": "locomo",
                "conversation_id": conversation,
                "question_type": stratum,
                "judge": {"status": "ok", "result": {"correct": correct}},
                "retrieve": {
                    "status": "ok",
                    "result": {
                        "answer_context_chars": chars,
                        "injected_count": 0 if silent else 2,
                        "silent": silent,
                        "latency_s": 0.5,
                    },
                },
            }
        )
    (path / "questions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def test_compare_reports_pairs_abstentions_and_cluster_interval(tmp_path):
    baseline = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _run(
        baseline,
        [
            ("a", "0", "1", True, 100, False),
            ("b", "0", "1", False, 100, False),
            ("c", "1", "5", True, 0, True),
        ],
    )
    _run(
        candidate,
        [
            ("a", "0", "1", False, 150, False),
            ("b", "0", "1", True, 150, False),
            ("c", "1", "5", False, 50, False),
        ],
    )
    result = compare_runs(baseline, candidate, seed=1, samples=100)
    assert result["primary"] == {
        "questions": 2,
        "baseline_correct": 1,
        "candidate_correct": 1,
        "wins": 1,
        "losses": 1,
        "ties": 0,
        "accuracy_delta": 0,
    }
    assert result["abstentions"]["losses"] == 1
    assert result["context"]["candidate_injection_rate"] == 1
    assert result["uncertainty"]["method"] == "conversation-cluster bootstrap"


def test_compare_requires_identical_completed_questions(tmp_path):
    baseline = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _run(baseline, [("a", "0", "1", True, 100, False)])
    _run(candidate, [("b", "0", "1", True, 100, False)])
    with pytest.raises(ValueError, match="identical question IDs"):
        compare_runs(baseline, candidate, samples=10)
