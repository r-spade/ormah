"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine
from ormah.index.db import Database
from ormah.store.file_store import FileStore


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Temporary memory directory."""
    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    return tmp_path


@pytest.fixture
def settings(tmp_memory_dir):
    return Settings(memory_dir=tmp_memory_dir)


@pytest.fixture
def file_store(tmp_memory_dir):
    return FileStore(tmp_memory_dir / "nodes")


@pytest.fixture
def db(tmp_memory_dir):
    database = Database(tmp_memory_dir / "index.db")
    database.init_schema()
    yield database
    database.close()


@pytest.fixture
def engine(settings):
    eng = MemoryEngine(settings)
    eng.startup()
    yield eng
    eng.shutdown()
