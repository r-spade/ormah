"""Tests for the durable write buffer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ormah.store.write_buffer import WriteBuffer, buffer_path_for


@pytest.fixture
def buf(tmp_path):
    return WriteBuffer(tmp_path / "pending_writes.jsonl")


class TestWriteBufferBasics:
    def test_is_empty_when_file_missing(self, buf):
        assert buf.is_empty()

    def test_append_creates_file(self, buf):
        buf.append(args={"content": "test"}, params={})
        assert buf.path.exists()

    def test_append_multiple_entries(self, buf):
        buf.append(args={"content": "one"}, params={})
        buf.append(args={"content": "two"}, params={})
        entries = buf.load()
        assert len(entries) == 2
        assert entries[0]["args"]["content"] == "one"
        assert entries[1]["args"]["content"] == "two"

    def test_clear_removes_file(self, buf):
        buf.append(args={"content": "x"}, params={})
        buf.clear()
        assert not buf.path.exists()

    def test_clear_on_missing_file_is_safe(self, buf):
        buf.clear()  # should not raise

    def test_load_empty_when_no_file(self, buf):
        assert buf.load() == []

    def test_is_empty_after_clear(self, buf):
        buf.append(args={"content": "x"}, params={})
        buf.clear()
        assert buf.is_empty()


class TestWriteBufferRewrite:
    def test_rewrite_replaces_contents(self, buf):
        buf.append(args={"content": "old"}, params={})
        new_entries = [{"args": {"content": "new"}, "params": {}}]
        buf.rewrite(new_entries)
        loaded = buf.load()
        assert len(loaded) == 1
        assert loaded[0]["args"]["content"] == "new"

    def test_rewrite_empty_list_clears_file(self, buf):
        buf.append(args={"content": "x"}, params={})
        buf.rewrite([])
        assert not buf.path.exists()

    def test_rewrite_atomic_via_tmp(self, buf):
        """tmp file should not persist after rewrite."""
        buf.rewrite([{"args": {"content": "a"}, "params": {}}])
        tmp = buf.path.with_suffix(".tmp")
        assert not tmp.exists()


class TestWriteBufferRoundtrip:
    def test_full_args_preserved(self, buf):
        args = {
            "content": "my memory",
            "type": "decision",
            "tier": "core",
            "title": "My Title",
            "space": "project-x",
            "tags": ["tag1", "tag2"],
            "confidence": 0.8,
        }
        params = {"default_space": "project-x"}
        buf.append(args=args, params=params)
        entries = buf.load()
        assert entries[0]["args"] == args
        assert entries[0]["params"] == params

    def test_malformed_line_skipped(self, buf, tmp_path):
        buf.path.parent.mkdir(parents=True, exist_ok=True)
        with buf.path.open("w") as f:
            f.write('{"args": {"content": "good"}, "params": {}}\n')
            f.write("NOT_VALID_JSON\n")
            f.write('{"args": {"content": "also good"}, "params": {}}\n')
        entries = buf.load()
        assert len(entries) == 2
        assert entries[0]["args"]["content"] == "good"
        assert entries[1]["args"]["content"] == "also good"


class TestBufferPathHelper:
    def test_buffer_path_is_sibling_of_memory_dir(self, tmp_path):
        memory_dir = tmp_path / "memory"
        path = buffer_path_for(memory_dir)
        assert path == tmp_path / "pending_writes.jsonl"
