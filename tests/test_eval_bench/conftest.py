from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from eval.bench.providers import TextProvider, TextResult
from eval.settings import RETRIEVAL_EVAL_SETTINGS_OVERRIDES
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, text):
        vec = np.zeros(768, dtype=np.float32)
        for word in text.lower().split():
            vec[int(hashlib.sha256(word.encode()).hexdigest()[:6], 16) % 768] += 1
        return vec / max(np.linalg.norm(vec), 1)

    encode_query = encode

    def encode_batch(self, texts, batch_size=8):
        self.calls += 1
        return np.array([self.encode(text) for text in texts])


@pytest.fixture
def encoder(monkeypatch):
    fake = FakeEncoder()
    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda *a: fake)
    monkeypatch.setattr("eval.bench.store.get_encoder", lambda *a: fake)
    return fake


@pytest.fixture
def bench_engine(tmp_path, encoder):
    settings = Settings(memory_dir=tmp_path / "db", **RETRIEVAL_EVAL_SETTINGS_OVERRIDES)
    engine = MemoryEngine(settings)
    engine.startup()
    yield engine
    engine.shutdown()


class FakeProvider(TextProvider):
    name = "fake"

    def _call(self, prompt, max_tokens):
        if self.phase == "extract":
            text = '{"memories":[{"title":"Ada bikes","content":"Ada rides a blue bike to work.","type":"fact"}]}'
        elif self.phase == "judge":
            text = '{"label":"CORRECT","reasoning":"Matches"}' if "CORRECT" in prompt else "yes"
        else:
            text = "Ada rides a blue bike."
        return TextResult(text, self.model, {"input_tokens": 12, "output_tokens": 5})


@pytest.fixture
def args():
    return SimpleNamespace(
        dataset="locomo",
        mode="raw",
        retrieval="recall",
        k=30,
        phase="free",
        limit=None,
        question_type=None,
        category=None,
        conversation=None,
        workers=2,
        max_usd=5,
        run_id="test",
        resume=False,
        extract_provider="claude-cli",
        extract_model=None,
        answer_provider="claude-cli",
        answer_model=None,
        judge_provider="codex",
        judge_model=None,
    )


@pytest.fixture
def locomo():
    return [
        {
            "conversation": {
                "session_1_date_time": "9:00 am on 1 January, 2023",
                "session_1": [
                    {
                        "speaker": "Ada",
                        "text": "I ride a blue bike to work every day.",
                        "dia_id": "D1:1",
                    },
                    {
                        "speaker": "Bob",
                        "text": "I walk to work instead of cycling.",
                        "dia_id": "D1:2",
                    },
                ],
            },
            "qa": [
                {
                    "question": "What does Ada ride?",
                    "answer": "blue bike",
                    "category": 4,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "Would Ada cycle?",
                    "answer": "Yes; she rides daily",
                    "category": 3,
                    "evidence": ["D1:1"],
                },
                {"question": "What car does Ada drive?", "category": 5, "evidence": []},
            ],
        }
    ]
