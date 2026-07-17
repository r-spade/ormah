"""startup() must not embed synchronously anymore (#32)."""
from __future__ import annotations

from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine


def test_startup_does_not_call_reindex(tmp_memory_dir, monkeypatch):
    called = {"reindex": False, "embed_rows": False}

    monkeypatch.setattr(
        MemoryEngine, "_reindex_all_embeddings",
        lambda self: called.__setitem__("reindex", True),
    )
    monkeypatch.setattr(
        MemoryEngine, "_embed_node_rows",
        lambda self, nodes: (called.__setitem__("embed_rows", True), ([], []))[1],
    )

    eng = MemoryEngine(Settings(memory_dir=tmp_memory_dir))
    eng.startup()
    try:
        assert called["reindex"] is False
        assert called["embed_rows"] is False
    finally:
        eng.shutdown()
