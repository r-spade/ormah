"""Shared test fixtures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine
from ormah.index.db import Database
from ormah.store.file_store import FileStore


_REAL_HOME = Path.home()
_REAL_ORMAH_PATHS = (
    _REAL_HOME / ".local" / "bin" / "ormah",
    _REAL_HOME / ".local" / "share" / "uv" / "tools" / "ormah",
    _REAL_HOME / ".local" / "share" / "ormah",
    _REAL_HOME / ".cache" / "ormah",
    _REAL_HOME / ".config" / "ormah",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_real_ormah_path(path: Path) -> bool:
    expanded = path.expanduser()
    candidates = {expanded, expanded.resolve(strict=False)}

    for protected in _REAL_ORMAH_PATHS:
        protected_candidates = {protected, protected.resolve(strict=False)}
        for candidate in candidates:
            for protected_candidate in protected_candidates:
                if candidate == protected_candidate or _is_relative_to(
                    candidate, protected_candidate
                ):
                    return True
    return False


@pytest.fixture(autouse=True)
def prevent_tests_mutating_real_ormah_install(monkeypatch):
    """Fail fast if a test tries to delete the developer's real Ormah install."""
    real_unlink = Path.unlink
    real_rmtree = shutil.rmtree

    def assert_safe_to_delete(path: Path) -> None:
        if _is_real_ormah_path(path):
            raise AssertionError(
                "Test attempted to delete the real Ormah install/config path "
                f"{path}. Patch Path.home() to tmp_path or mock the destructive helper."
            )

    def guarded_unlink(self, *args, **kwargs):
        assert_safe_to_delete(self)
        return real_unlink(self, *args, **kwargs)

    def guarded_rmtree(path, *args, **kwargs):
        assert_safe_to_delete(Path(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    monkeypatch.setattr(shutil, "rmtree", guarded_rmtree)


@pytest.fixture(autouse=True)
def _isolate_settings_from_global_env(monkeypatch, tmp_path):
    """Stop the global ~/.config/ormah/.env and stray ORMAH_* OS vars from
    leaking into bare Settings() during tests (env pollution, not regressions).
    """
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    monkeypatch.setitem(Settings.model_config, "env_file", str(empty_env))
    for key in list(os.environ):
        if key.startswith("ORMAH_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session", autouse=True)
def isolate_fastembed_cache(tmp_path_factory):
    """Keep model download/cache mutations outside the real Ormah install.

    The deletion guard above intentionally protects ``~/.local/share/ormah``,
    which is also FastEmbed's production cache location.  A fresh test runner
    must therefore use a separate cache so Hugging Face can atomically replace
    its ``*.incomplete`` downloads without tripping the guard.
    """
    cache_dir = tmp_path_factory.getbasetemp().parent / "ormah-fastembed-cache"
    cache_dir.mkdir(exist_ok=True)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_dir))
        yield cache_dir


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
