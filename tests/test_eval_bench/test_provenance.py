import eval.bench.provenance as provenance


def test_fingerprint_detects_dirty_source_and_prompt_changes(tmp_path, monkeypatch):
    source = tmp_path / "ormah"
    source.mkdir()
    engine = source / "memory_engine.py"
    engine.write_text("before")
    bench = tmp_path / "eval/bench"
    bench.mkdir(parents=True)
    prompt = bench / "answer.py"
    prompt.write_text("prompt before")
    monkeypatch.setattr(provenance.ormah, "__file__", str(source / "__init__.py"))
    monkeypatch.setattr(provenance, "__file__", str(bench / "provenance.py"))
    first = provenance.experiment_fingerprint({}, {})
    engine.write_text("after")
    second = provenance.experiment_fingerprint({}, {})
    prompt.write_text("prompt after")
    third = provenance.experiment_fingerprint({}, {})
    assert len({p["source_sha256"] for p in (first, second, third)}) == 3
    (bench / "README.md").write_text("documentation-only edit")
    assert provenance.experiment_fingerprint({}, {}) == third


def test_fingerprint_does_not_expose_config_secrets(monkeypatch):
    monkeypatch.setenv("ORMAH_ACCOUNT_TOKEN", "private-account-secret")
    result = provenance.experiment_fingerprint({}, {})
    assert "private-account-secret" not in str(result)
