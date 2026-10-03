from eval.bench.answer import answer_prompt
from eval.bench.diagnostics import aggregate_diagnostics, diagnose_question


def row(*, dataset="locomo", mode="raw", retrieved=None, correct=False, evidence=True):
    turn_id = "D1:1" if dataset == "locomo" else "s1:0"
    session_id = "locomo:0:session_1" if dataset == "locomo" else "s1"
    return {
        "question_id": "q",
        "dataset": dataset,
        "question": "What does Ada ride?",
        "gold": "HIDDEN GOLD SENTINEL",
        "question_type": "4" if dataset == "locomo" else "single-session-user",
        "question_date": "2023-01-02T00:00:00+00:00",
        "gold_ids": [turn_id] if dataset == "locomo" else [session_id],
        "abstention": False,
        "source_evidence": ([{
            "session_id": session_id,
            "turn_id": turn_id,
            "speaker": "Ada",
            "text": "SOURCE EVIDENCE SENTINEL",
            "date": "2023-01-01T00:00:00+00:00",
        }] if evidence else []),
        "store": {"status": "ok"},
        "retrieve": {"status": "ok", "result": {
            "ranked": retrieved or [], "answer_context_chars": 0,
        }},
        "answer": {"status": "ok", "result": {"text": "not enough information"}},
        "judge": {"status": "ok", "result": {"correct": correct, "reasoning": "model verdict"}},
    }


def memory(*, content="SOURCE EVIDENCE SENTINEL", turn="D1:1", session="locomo:0:session_1"):
    return {
        "id": "m1", "title": "Ada: source", "content": content,
        "created": "2023-01-01T00:00:00+00:00",
        "tags": [f"turn:{turn}", f"session:{session}"],
    }


def ranked(mem):
    return [{"node": mem, "score": 1.0, "source": "test"}]


def test_synthetic_storage_absence():
    diagnostic = diagnose_question(row(), [], "raw")
    assert diagnostic["failure_stage"] == "storage_absence"
    assert diagnostic["store"]["supporting_turn_coverage"]["recall"] == 0


def test_synthetic_retrieval_miss():
    diagnostic = diagnose_question(row(), [memory()], "raw")
    assert diagnostic["failure_stage"] == "retrieval_miss"
    assert diagnostic["retrieve"]["supporting_turn_coverage"]["recall"] == 0


def test_synthetic_title_only_exposure_loss_and_faithful_context():
    stored = memory()
    exposed = {**stored, "content": ""}
    diagnostic = diagnose_question(row(retrieved=ranked(exposed)), [stored], "raw")
    assert diagnostic["failure_stage"] == "exposure_loss_or_truncation"
    assert diagnostic["exposure"]["supporting_memories"][0]["status"] == "title_only"
    assert "SOURCE EVIDENCE SENTINEL" not in diagnostic["exposure"]["context"]
    assert diagnostic["exposure"]["context"] == answer_prompt(
        row(retrieved=ranked(exposed)), ranked(exposed)
    ).split("Memories:\n", 1)[1].split("\n\nQuestion:", 1)[0]


def test_synthetic_full_evidence_downstream_needs_review():
    stored = memory()
    diagnostic = diagnose_question(row(retrieved=ranked(stored)), [stored], "raw")
    assert diagnostic["failure_stage"] == "downstream_answer_or_judge_needs_review"
    assert diagnostic["exposure"]["supporting_full_content_coverage"]["recall"] == 1


def test_synthetic_extract_provenance_is_unresolvable():
    stored = memory(turn="not-recorded")
    diagnostic = diagnose_question(row(retrieved=ranked(stored)), [stored], "extract")
    assert diagnostic["failure_stage"] == "unresolvable_extracted_provenance"
    assert diagnostic["store"]["supporting_turn_coverage"] is None


def test_unknown_source_and_judge_correct_are_not_overclaimed():
    unresolved = diagnose_question(row(evidence=False), [memory()], "raw")
    assert unresolved["failure_stage"] == "unknown_unresolved_source_evidence"
    marked = diagnose_question(row(retrieved=ranked(memory()), correct=True), [memory()], "raw")
    assert marked["failure_stage"] == "judge_marked_correct_not_adjudicated"


def test_diagnostic_metadata_never_enters_answer_prompt():
    prompt = answer_prompt(row(), [])
    assert "HIDDEN GOLD SENTINEL" not in prompt
    assert "SOURCE EVIDENCE SENTINEL" not in prompt
    assert "D1:1" not in prompt


def test_aggregate_has_explicit_denominators_and_errors():
    stored = memory()
    items = [diagnose_question(row(retrieved=ranked(stored)), [stored], "raw")]
    result = aggregate_diagnostics(items)
    assert result["supporting_turn_retrieval"] == {"questions": 1, "macro_recall": 1}
    assert result["failure_stage_counts"] == {"downstream_answer_or_judge_needs_review": 1}
