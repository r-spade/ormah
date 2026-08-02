"""Issue #87: pair batching — settings, timeout hint, batch module."""
import json

from ormah.background.llm import pair_batch
from ormah.config import Settings


def test_batching_settings_defaults(tmp_path):
    s = Settings(memory_dir=tmp_path)
    assert s.maintenance_pairs_per_call == 1          # K=1 -> legacy flow untouched
    assert s.maintenance_timeout_per_pair_seconds == 10
    # per-job K overrides (council C3): 0 = fall back to the global K
    assert s.auto_link_pairs_per_call == 0
    assert s.duplicate_check_pairs_per_call == 0
    assert s.conflict_check_pairs_per_call == 0
    # caps default to CURRENT-equivalent bounds (council I1): 0 = unbounded
    assert s.auto_link_max_pairs_per_run == 0
    assert s.duplicate_check_max_pairs_per_run == 0
    assert s.conflict_check_max_pairs_per_run == 10000   # today's exact bound


# --- pair_batch.judge_pairs (Task 05) ---

def _settings(tmp_path, k):
    return Settings(memory_dir=tmp_path, maintenance_pairs_per_call=k,
                    llm_timeout_seconds=60, maintenance_timeout_per_pair_seconds=10)


PAIRS = [{"id": i} for i in range(4)]
RENDER = lambda p: f"pair-{p['id']}"       # noqa: E731
INSTR = "JUDGE THE PAIR"


def test_k1_is_a_pure_map_over_judge_single(tmp_path, monkeypatch):
    monkeypatch.setattr(pair_batch, "llm_generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no batch call at K=1")))
    out = pair_batch.judge_pairs(_settings(tmp_path, 1), INSTR, PAIRS, RENDER,
                                 judge_single=lambda p: {"ok": p["id"]})
    assert out == [{"ok": 0}, {"ok": 1}, {"ok": 2}, {"ok": 3}]


def test_explicit_k_overrides_settings(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_generate(settings, prompt, json_mode=True, **kw):
        calls["n"] += 1
        n = prompt.count("### Pair ")
        return json.dumps({"verdicts": [{"pair_id": i, "v": i} for i in range(n)]})

    monkeypatch.setattr(pair_batch, "llm_generate", fake_generate)
    out = pair_batch.judge_pairs(_settings(tmp_path, 1), INSTR, PAIRS, RENDER,
                                 judge_single=lambda p: {"single": True}, k=4)
    assert calls["n"] == 1 and [v["v"] for v in out] == [0, 1, 2, 3]


def test_valid_batch_applies_all_verdicts(tmp_path, monkeypatch):
    prompts = []

    def fake_generate(settings, prompt, json_mode=True, **kw):
        prompts.append((prompt, kw.get("timeout_hint_seconds")))
        return json.dumps({"verdicts": [{"pair_id": i, "v": i} for i in range(4)]})

    monkeypatch.setattr(pair_batch, "llm_generate", fake_generate)
    out = pair_batch.judge_pairs(_settings(tmp_path, 4), INSTR, PAIRS, RENDER,
                                 judge_single=lambda p: {"single": True})
    assert [v["v"] for v in out] == [0, 1, 2, 3]
    assert prompts[0][1] == 60 + 10 * 4          # base + per_pair * K
    assert INSTR in prompts[0][0] and "pair-3" in prompts[0][0]


def test_partial_verdicts_leave_missing_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pair_batch, "llm_generate",
                        lambda *a, **k: json.dumps({"verdicts": [{"pair_id": 1, "v": 1}]}))
    out = pair_batch.judge_pairs(_settings(tmp_path, 4), INSTR, PAIRS, RENDER,
                                 judge_single=lambda p: {"single": True})
    assert out[1] == {"pair_id": 1, "v": 1}
    assert out[0] is None and out[2] is None and out[3] is None


def test_parse_failure_bisects_to_single(tmp_path, monkeypatch):
    monkeypatch.setattr(pair_batch, "llm_generate", lambda *a, **k: "NOT JSON {{{")
    singles = []

    def judge_single(p):
        singles.append(p["id"])
        return {"single": p["id"]}

    out = pair_batch.judge_pairs(_settings(tmp_path, 4), INSTR, PAIRS, RENDER, judge_single)
    assert singles == [0, 1, 2, 3]               # ladder bottomed out per pair
    assert [v["single"] for v in out] == [0, 1, 2, 3]


def test_llm_unavailable_aborts_remaining_chunks(tmp_path, monkeypatch):
    """Council C1: an outage must not iterate the whole collected list."""
    calls = {"n": 0}

    def fake_generate(*a, **k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(pair_batch, "llm_generate", fake_generate)
    out = pair_batch.judge_pairs(_settings(tmp_path, 2), INSTR, PAIRS, RENDER,
                                 judge_single=lambda p: {"single": True})
    assert out == [None, None, None, None]
    assert calls["n"] == 1                        # chunk 1 fails -> chunk 2 never attempted
