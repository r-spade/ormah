"""space_locked: user-curated spaces survive auto_cluster (#22 council follow-up).

A node marked space_locked is global/curated on purpose; the persistence layer must
round-trip the flag through markdown and the SQLite index.
"""

from ormah.index.builder import IndexBuilder
from ormah.index.db import Database
from ormah.models.node import MemoryNode
from ormah.store.file_store import FileStore
from ormah.store.markdown import parse_node, serialize_node


def test_model_default_is_unlocked():
    assert MemoryNode(type="fact").space_locked is False


def test_markdown_roundtrip_preserves_lock():
    node = MemoryNode(type="fact", content="x", space_locked=True)
    text = serialize_node(node)
    assert "space_locked: true" in text
    assert parse_node(text).space_locked is True


def test_markdown_omits_flag_when_unlocked():
    node = MemoryNode(type="fact", content="x")
    assert "space_locked" not in serialize_node(node)


def test_index_persists_space_locked(tmp_path):
    fs = FileStore(tmp_path / "nodes")
    locked = MemoryNode(type="fact", content="locked", space_locked=True)
    fs.save(locked)
    fs.save(MemoryNode(type="fact", content="plain"))

    db = Database(tmp_path / "index.db")
    db.init_schema()
    IndexBuilder(db, fs).full_rebuild()

    row = db.conn.execute(
        "SELECT space_locked FROM nodes WHERE id = ?", (locked.id,)
    ).fetchone()
    assert row[0] == 1
