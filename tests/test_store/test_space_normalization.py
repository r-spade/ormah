"""Space normalization: placeholder strings persist as None, not literal 'null' (#22).

A handful of stored nodes carried the literal string ``space='null'`` (not SQL NULL),
landing them in a phantom "null" space group. Root cause: no write-path guard coerced
the placeholder to None. The fix normalizes at the MemoryNode boundary (every persisted
node) plus the update path.
"""

import pytest

from ormah.index.db import Database
from ormah.models.node import MemoryNode, normalize_space
from ormah.store.file_store import FileStore
from ormah.store.markdown import parse_node, serialize_node
from ormah.store.migrations import migrate_null_space


@pytest.mark.parametrize("raw", ["null", "none", "NULL", "None", " null ", "", "   "])
def test_placeholder_space_becomes_none(raw):
    assert MemoryNode(type="fact", space=raw).space is None


@pytest.mark.parametrize("raw", ["work", "AndreMartins", "global", "ormah"])
def test_real_space_is_preserved(raw):
    assert MemoryNode(type="fact", space=raw).space == raw


def test_none_space_stays_none():
    assert MemoryNode(type="fact", space=None).space is None


def test_real_space_is_stripped():
    assert MemoryNode(type="fact", space="  work  ").space == "work"


def test_normalize_space_helper():
    assert normalize_space("null") is None
    assert normalize_space("") is None
    assert normalize_space(None) is None
    assert normalize_space("work") == "work"


def test_markdown_roundtrip_drops_placeholder_space():
    """A file corrupted with the literal string 'null' parses back to None."""
    node = MemoryNode(type="fact", space="work", content="x")
    corrupted = serialize_node(node).replace("space: work", "space: 'null'")
    assert parse_node(corrupted).space is None


def test_migration_cleans_placeholder_space_in_files_and_index(tmp_path):
    """End-to-end migration on a throwaway store: files and index both cleaned."""
    nodes_dir = tmp_path / "nodes"
    fs = FileStore(nodes_dir)

    fs.save(MemoryNode(type="fact", space="work", content="clean"))  # untouched
    dirty = MemoryNode(type="fact", space="work", content="dirty")
    path = fs.save(dirty)
    path.write_text(
        serialize_node(dirty).replace("space: work", "space: 'null'"), encoding="utf-8"
    )
    assert "space: 'null'" in path.read_text(encoding="utf-8")  # sanity

    fixed, reindexed = migrate_null_space(nodes_dir, tmp_path / "index.db")

    assert fixed == 1
    assert reindexed == 2
    assert FileStore(nodes_dir).load(dirty.id).space is None  # source of truth cleaned
    db = Database(tmp_path / "index.db")
    spaces = {r[0] for r in db.conn.execute("SELECT space FROM nodes").fetchall()}
    assert "null" not in spaces
    assert spaces == {None, "work"}
