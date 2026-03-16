"""Tests for the hippocampus file-watching & auto-ingestion layer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ormah.background.hippocampus import (
    HippocampusHandler,
    _detect_space,
    _ingest_file,
    _load_state,
    _save_state,
    _scan_directory,
    run_hippocampus_scan,
    start_hippocampus,
    stop_hippocampus,
)
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine

_LLM_PATCH = "ormah.background.llm_client.llm_generate"

_LLM_RESPONSE = json.dumps({"memories": [
    {
        "content": "Using PostgreSQL for the database.",
        "type": "decision",
        "title": "Database choice",
        "tags": ["database"],
    },
]})

_LLM_RESPONSE_2 = json.dumps({"memories": [
    {
        "content": "Using Redis for caching layer with TTL-based eviction.",
        "type": "decision",
        "title": "Cache choice",
        "tags": ["caching"],
    },
]})

_SAMPLE_MD = (
    "# Session Notes\n\n"
    "We decided to use PostgreSQL for the database.\n"
    "The main reasons are JSON support and PostGIS.\n"
)


def test_initial_scan_ingests_existing_files(engine, tmp_path):
    """Files present before watcher starts get ingested on catch-up scan."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "notes.md").write_text(_SAMPLE_MD)

    engine.settings.hippocampus_watch_dirs = [watch_dir]
    engine.settings.hippocampus_enabled = True

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        observers = start_hippocampus(engine)

    try:
        state = _load_state(watch_dir)
        assert "notes.md" in state
    finally:
        stop_hippocampus(observers)


def test_new_file_triggers_ingestion(engine, tmp_path):
    """A file created after watcher starts gets ingested."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()

    engine.settings.hippocampus_watch_dirs = [watch_dir]
    engine.settings.hippocampus_enabled = True
    engine.settings.hippocampus_debounce_seconds = 0.1

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        observers = start_hippocampus(engine)
        try:
            (watch_dir / "new_session.md").write_text(_SAMPLE_MD)
            # Wait for debounce + processing
            time.sleep(0.5)
            state = _load_state(watch_dir)
            assert "new_session.md" in state
        finally:
            stop_hippocampus(observers)


def test_modified_file_re_ingested(engine, tmp_path):
    """Changing a file's content triggers re-ingestion."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    md_file = watch_dir / "notes.md"
    md_file.write_text(_SAMPLE_MD)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        _ingest_file(engine, md_file, state, watch_dir)

    old_hash = state["notes.md"]["hash"]

    # Modify the file
    md_file.write_text(_SAMPLE_MD + "\nAlso using Redis for caching.\n")

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        result = _ingest_file(engine, md_file, state, watch_dir)

    assert result is True
    assert state["notes.md"]["hash"] != old_hash


def test_unchanged_file_skipped(engine, tmp_path):
    """Same hash means the file is not re-ingested."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    md_file = watch_dir / "notes.md"
    md_file.write_text(_SAMPLE_MD)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_file(engine, md_file, state, watch_dir) is True
        assert _ingest_file(engine, md_file, state, watch_dir) is False


def test_non_md_files_ignored(engine, tmp_path):
    """Only .md files are picked up by scan."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "data.txt").write_text("Some text data that should be ignored")
    (watch_dir / "config.json").write_text('{"key": "value"}')
    (watch_dir / "notes.md").write_text(_SAMPLE_MD)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        count = _scan_directory(engine, watch_dir)

    assert count == 1
    state = _load_state(watch_dir)
    assert "notes.md" in state
    assert "data.txt" not in state
    assert "config.json" not in state


def test_debounce_prevents_duplicate_ingestion(engine, tmp_path):
    """Rapid writes result in a single ingestion call."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()

    handler = HippocampusHandler(engine, watch_dir, debounce_seconds=0.3)
    md_file = watch_dir / "rapid.md"

    call_count = 0
    original_ingest = _ingest_file

    def counting_ingest(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_ingest(*args, **kwargs)

    with patch("ormah.background.hippocampus._ingest_file", side_effect=counting_ingest):
        # Simulate rapid writes
        for i in range(5):
            md_file.write_text(f"{_SAMPLE_MD}\nEdit {i}\n")
            from watchdog.events import FileModifiedEvent
            handler.on_modified(FileModifiedEvent(str(md_file)))
            time.sleep(0.05)

        # Wait for debounce
        time.sleep(0.5)

    assert call_count == 1


def test_state_file_persists(engine, tmp_path):
    """State file survives across load/save cycles."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()

    state = {"notes.md": {"hash": "abc123", "last_ingested": "2024-01-01T00:00:00", "size": 100}}
    _save_state(watch_dir, state)

    loaded = _load_state(watch_dir)
    assert loaded == state
    assert loaded["notes.md"]["hash"] == "abc123"


def test_space_detection(tmp_path):
    """Git repo path produces correct space name."""
    # For a non-git directory, falls back to parent dir name
    test_dir = tmp_path / "my-project"
    test_dir.mkdir()
    test_file = test_dir / "notes.md"
    test_file.write_text("test")

    space = _detect_space(test_file)
    assert space == "my-project"


def test_disabled_does_nothing(engine, tmp_path):
    """hippocampus_enabled=False returns no observers."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "notes.md").write_text(_SAMPLE_MD)

    engine.settings.hippocampus_watch_dirs = [watch_dir]
    engine.settings.hippocampus_enabled = False

    observers = start_hippocampus(engine)
    assert observers == []


def test_manual_scan(engine, tmp_path):
    """run_hippocampus_scan ingests files from configured dirs."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "session.md").write_text(_SAMPLE_MD)

    engine.settings.hippocampus_watch_dirs = [watch_dir]
    engine.settings.hippocampus_enabled = True

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        run_hippocampus_scan(engine)

    state = _load_state(watch_dir)
    assert "session.md" in state


def test_ingest_stores_node_ids(engine, tmp_path):
    """Ingesting a file records node_ids from created memories in state."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    md_file = watch_dir / "notes.md"
    md_file.write_text(_SAMPLE_MD)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_file(engine, md_file, state, watch_dir) is True

    entry = state["notes.md"]
    assert "node_ids" in entry
    assert len(entry["node_ids"]) == 1  # _LLM_RESPONSE has 1 memory

    # Verify provenance query returns the same ids
    provenance = MemoryEngine.get_file_provenance(str(watch_dir), "notes.md")
    assert provenance == entry["node_ids"]


def test_reingest_appends_node_ids(engine, tmp_path):
    """Re-ingesting a modified file appends new node_ids to existing ones."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    md_file = watch_dir / "notes.md"
    md_file.write_text(_SAMPLE_MD)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        _ingest_file(engine, md_file, state, watch_dir)

    first_ids = list(state["notes.md"]["node_ids"])
    assert len(first_ids) == 1

    # Modify the file to trigger re-ingestion (use different LLM response to avoid dedup)
    md_file.write_text(_SAMPLE_MD + "\nAlso using Redis for caching.\n")

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE_2):
        assert _ingest_file(engine, md_file, state, watch_dir) is True

    all_ids = state["notes.md"]["node_ids"]
    assert len(all_ids) == 2
    # First id is preserved
    assert all_ids[0] == first_ids[0]


def test_ignore_patterns_skip_files(engine, tmp_path):
    """Files matching ignore patterns are skipped by scan and ingest."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    subdir = watch_dir / "node_modules" / "pkg"
    subdir.mkdir(parents=True)
    (subdir / "readme.md").write_text(_SAMPLE_MD)
    (watch_dir / "notes.md").write_text(_SAMPLE_MD)
    (watch_dir / "CHANGELOG.md").write_text(_SAMPLE_MD)

    ignore = ["**/node_modules/**", "**/CHANGELOG.md"]

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        count = _scan_directory(engine, watch_dir, ignore_patterns=ignore)

    assert count == 1  # only notes.md
    state = _load_state(watch_dir)
    assert "notes.md" in state
    assert "node_modules/pkg/readme.md" not in state
    assert "CHANGELOG.md" not in state


def test_scan_cleans_stale_state(engine, tmp_path):
    """Deleted files have their state entries removed on next scan."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    md_file = watch_dir / "ephemeral.md"
    md_file.write_text(_SAMPLE_MD)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        _scan_directory(engine, watch_dir)

    state = _load_state(watch_dir)
    assert "ephemeral.md" in state

    # Delete the file
    md_file.unlink()

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        _scan_directory(engine, watch_dir)

    state = _load_state(watch_dir)
    assert "ephemeral.md" not in state
