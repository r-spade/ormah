"""Tests for the FastAPI app shell."""

import logging
import re
from types import SimpleNamespace

import pytest

from ormah import main
from ormah.main import _LOCAL_CORS_ORIGIN_REGEX, _is_reserved_api_path


def test_spa_fallback_excludes_api_prefixes():
    assert _is_reserved_api_path("agent/context") is True
    assert _is_reserved_api_path("admin/healthz") is True
    assert _is_reserved_api_path("ingest/missing") is True
    assert _is_reserved_api_path("ui/missing") is True


def test_spa_fallback_allows_frontend_routes():
    assert _is_reserved_api_path("") is False
    assert _is_reserved_api_path("graph") is False
    assert _is_reserved_api_path("projects/ormah") is False


def test_cors_regex_allows_only_loopback_origins():
    assert re.match(_LOCAL_CORS_ORIGIN_REGEX, "http://localhost:5173")
    assert re.match(_LOCAL_CORS_ORIGIN_REGEX, "http://127.0.0.1:8787")
    assert re.match(_LOCAL_CORS_ORIGIN_REGEX, "https://[::1]:8787")
    assert not re.match(_LOCAL_CORS_ORIGIN_REGEX, "https://evil.example")


@pytest.mark.parametrize("error", [RuntimeError("corrupt"), OSError("unwritable")])
def test_local_admin_failure_disables_only_sensitive_routes(monkeypatch, caplog, error):
    app = SimpleNamespace(state=SimpleNamespace())

    def fail_closed():
        raise error

    monkeypatch.setattr(main, "load_or_create_local_admin_token", fail_closed)
    with caplog.at_level(logging.WARNING):
        main._initialize_local_admin(app)

    assert app.state.local_admin_token is None
    assert "routes are disabled" in caplog.text
    assert str(error) not in caplog.text


# --- one server per memory directory ---------------------------------------------------


def _lifespan_settings(monkeypatch, tmp_path, port=18787):
    from ormah.config import Settings

    lifespan_settings = Settings(memory_dir=tmp_path / "memory", port=port)
    monkeypatch.setattr(main, "settings", lifespan_settings)
    return lifespan_settings


def _stub_server_body(monkeypatch, events):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_run_server(app):
        events.append("started")
        yield
        events.append("stopped")

    monkeypatch.setattr(main, "_run_server", fake_run_server)


def test_lifespan_refuses_to_start_when_directory_is_locked(monkeypatch, tmp_path):
    import asyncio

    from ormah.server_lock import ServerLock, ServerLockHeld

    lifespan_settings = _lifespan_settings(monkeypatch, tmp_path)
    events: list[str] = []
    _stub_server_body(monkeypatch, events)

    other = ServerLock(lifespan_settings.memory_dir, port=9999)
    other.acquire()
    try:

        async def attempt():
            async with main.lifespan(main.app):
                pass

        with pytest.raises(ServerLockHeld) as exc_info:
            asyncio.run(attempt())
    finally:
        other.release()

    assert "port 9999" in str(exc_info.value)
    assert events == []


def test_lifespan_holds_lock_while_running_and_releases_on_shutdown(monkeypatch, tmp_path):
    import asyncio

    from ormah.server_lock import ServerLock, ServerLockHeld, check_available

    lifespan_settings = _lifespan_settings(monkeypatch, tmp_path, port=18788)
    events: list[str] = []
    _stub_server_body(monkeypatch, events)
    seen: dict = {}

    async def run():
        async with main.lifespan(main.app):
            try:
                ServerLock(lifespan_settings.memory_dir).acquire()
            except ServerLockHeld as exc:
                seen["error"] = exc

    asyncio.run(run())

    assert events == ["started", "stopped"]
    assert seen["error"].holder["port"] == 18788
    check_available(lifespan_settings.memory_dir)


def test_lifespan_releases_lock_when_engine_startup_fails(monkeypatch, tmp_path):
    import asyncio

    from ormah.server_lock import check_available

    lifespan_settings = _lifespan_settings(monkeypatch, tmp_path)

    class ExplodingEngine:
        def __init__(self, settings):
            pass

        def startup(self):
            raise RuntimeError("index is corrupt")

    monkeypatch.setattr(main, "MemoryEngine", ExplodingEngine)
    monkeypatch.setattr(main, "_initialize_local_admin", lambda app: None)

    async def run():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="index is corrupt"):
        asyncio.run(run())

    check_available(lifespan_settings.memory_dir)


def test_lifespan_takes_lock_before_constructing_the_engine(monkeypatch, tmp_path):
    import asyncio

    from ormah.server_lock import ServerLock, ServerLockHeld

    lifespan_settings = _lifespan_settings(monkeypatch, tmp_path)
    constructed: list[bool] = []

    class RecordingEngine:
        def __init__(self, settings):
            constructed.append(True)

    monkeypatch.setattr(main, "MemoryEngine", RecordingEngine)
    monkeypatch.setattr(main, "_initialize_local_admin", lambda app: None)

    other = ServerLock(lifespan_settings.memory_dir)
    other.acquire()
    try:

        async def run():
            async with main.lifespan(main.app):
                pass

        with pytest.raises(ServerLockHeld):
            asyncio.run(run())
    finally:
        other.release()

    assert constructed == []
