import json

import pytest

from eval.bench.datasets import sha256_file
from eval.bench.runner import selected_questions, validate
from eval.bench.splits import load_split
from tests.test_eval_bench.test_runner import dataset


def test_locked_split_selection_and_checksum(args, tmp_path, locomo):
    dataset(tmp_path, locomo)
    data_path = tmp_path / "data/locomo10.json"
    manifest = {
        "datasets": {"locomo": {
            "sha256": sha256_file(data_path),
            "splits": {"development": {"question_ids": ["locomo:0:1"]}},
        }}
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    args.split_manifest, args.split = str(path), "development"
    questions = list(selected_questions(args, data_path))
    assert [q.question_id for q in questions] == ["locomo:0:1"]
    locomo[0]["qa"][0]["question"] = "changed"
    data_path.write_text(json.dumps(locomo))
    with pytest.raises(ValueError, match="checksum differs"):
        load_split(path, "locomo", "development", data_path)


def test_split_requires_both_arguments(args):
    args.split = "development"
    with pytest.raises(ValueError, match="used together"):
        validate(args)
