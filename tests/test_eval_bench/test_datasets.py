import json

import pytest

from eval.bench.datasets import download, iter_array, load_questions, sha256_file


def test_locomo_categories_and_shared_haystack(tmp_path, locomo):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(locomo))
    questions = list(load_questions(path, "locomo"))
    assert questions[1].gold == "Yes"
    assert questions[2].abstention
    assert questions[0].sessions is questions[1].sessions
    assert questions[0].sessions[0].turns[0].turn_id == "D1:1"
    assert questions[0].sessions[0].date.startswith("2023-01-01")
    assert "sessions" not in questions[0].metadata()


def test_longmemeval(tmp_path):
    fixture = [
        {
            "question_id": "q_abs",
            "question_type": "multi-session",
            "question": "Where?",
            "answer": "Unknown",
            "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_dates": ["2020/01/01 (Wed) 09:00"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[{"role": "user", "content": "Hello"}]],
            "answer_session_ids": [],
        }
    ]
    path = tmp_path / "q.json"
    path.write_text(json.dumps(fixture))
    question = next(load_questions(path, "longmemeval"))
    assert question.abstention
    assert question.sessions[0].turns[0].turn_id == "s1:0"
    assert question.sessions[0].date.startswith("2020-01-01")


def test_stream_and_torn_dataset(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"content": "α" * 50}, {"content": "b" * 50}]))
    assert len(list(iter_array(path, chunk_size=3))) == 2
    path.write_text('[{"x":1}')
    with pytest.raises(ValueError):
        list(iter_array(path, chunk_size=3))


def test_download_manifest_existing_file(tmp_path):
    (tmp_path / "locomo10.json").write_text("[]")
    manifest = download(tmp_path, "locomo")
    assert manifest["locomo"]["sha256"] == sha256_file(tmp_path / "locomo10.json")
    assert manifest["locomo"]["bytes"] == 2
