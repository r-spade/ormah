"""Tests for the one-server-per-memory-directory lock."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from ormah.cloud.store_lock import StoreLock
from ormah.server_lock import ServerLock, ServerLockHeld, check_available

_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from ormah.server_lock import ServerLock

    lock = ServerLock(Path(sys.argv[1]), port=int(sys.argv[2]))
    lock.acquire()
    print("held", flush=True)
    time.sleep(120)
    """
)


@pytest.fixture
def holder(tmp_path):
    """Start a subprocess that holds the server lock on tmp_path; yield the Popen."""
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(tmp_path), "9123"],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout.readline().strip() == "held"
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()


def test_acquire_succeeds_on_fresh_directory(tmp_path):
    memory_dir = tmp_path / "does" / "not" / "exist"
    lock = ServerLock(memory_dir, port=8800)
    lock.acquire()
    try:
        assert lock.held
    finally:
        lock.release()


def test_second_lock_object_in_same_process_fails_immediately(tmp_path):
    first = ServerLock(tmp_path, port=8800)
    first.acquire()
    try:
        second = ServerLock(tmp_path, port=8801)
        started = time.monotonic()
        with pytest.raises(ServerLockHeld) as exc_info:
            second.acquire()
        assert time.monotonic() - started < 1.0
        assert str(tmp_path.resolve()) in str(exc_info.value)
        assert exc_info.value.memory_dir == tmp_path.resolve()
        assert not second.held
    finally:
        first.release()


def test_failed_acquire_does_not_clobber_holder_record(tmp_path):
    first = ServerLock(tmp_path, port=8800)
    first.acquire()
    try:
        with pytest.raises(ServerLockHeld):
            ServerLock(tmp_path, port=8801).acquire()
        with pytest.raises(ServerLockHeld) as exc_info:
            ServerLock(tmp_path, port=8802).acquire()
        assert exc_info.value.holder["port"] == 8800
    finally:
        first.release()


def test_symlinked_path_is_the_same_lock(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    first = ServerLock(real)
    first.acquire()
    try:
        with pytest.raises(ServerLockHeld):
            ServerLock(link).acquire()
    finally:
        first.release()


def test_acquiring_twice_on_one_object_is_a_programming_error(tmp_path):
    lock = ServerLock(tmp_path)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            lock.acquire()
    finally:
        lock.release()


def test_release_lets_a_new_acquire_succeed(tmp_path):
    first = ServerLock(tmp_path, port=8800)
    first.acquire()
    first.release()
    second = ServerLock(tmp_path, port=8801)
    second.acquire()
    second.release()


def test_context_manager_releases_on_error(tmp_path):
    with pytest.raises(ValueError):
        with ServerLock(tmp_path):
            raise ValueError("boom")
    check_available(tmp_path)


def test_subprocess_holder_blocks_with_pid_and_port_in_message(tmp_path, holder):
    with pytest.raises(ServerLockHeld) as exc_info:
        ServerLock(tmp_path, port=8800).acquire()
    message = str(exc_info.value)
    assert f"pid {holder.pid}" in message
    assert "port 9123" in message
    assert str(tmp_path.resolve()) in message
    assert "ORMAH_MEMORY_DIR" in message


def test_sigkilled_holder_leaves_no_stale_lock(tmp_path, holder):
    with pytest.raises(ServerLockHeld):
        check_available(tmp_path)

    holder.send_signal(signal.SIGKILL)
    holder.wait(timeout=10)

    lock = ServerLock(tmp_path, port=8800)
    lock.acquire()
    lock.release()


def test_stale_sidecar_is_harmless(tmp_path):
    """A leftover server.json from a dead process neither blocks nor misleads."""
    (tmp_path / ".ormah").mkdir()
    (tmp_path / ".ormah" / "server.json").write_text(
        '{"pid": 2147483646, "port": 1234, "role": "server"}'
    )
    lock = ServerLock(tmp_path, port=8800)
    lock.acquire()
    try:
        with pytest.raises(ServerLockHeld) as exc_info:
            ServerLock(tmp_path).acquire()
        assert exc_info.value.holder["pid"] == os.getpid()
        assert exc_info.value.holder["port"] == 8800
    finally:
        lock.release()


def test_corrupt_sidecar_still_refuses_with_directory_in_message(tmp_path):
    lock = ServerLock(tmp_path)
    lock.acquire()
    try:
        (tmp_path / ".ormah" / "server.json").write_text("{not json")
        with pytest.raises(ServerLockHeld) as exc_info:
            ServerLock(tmp_path).acquire()
        assert exc_info.value.holder is None
        assert str(tmp_path.resolve()) in str(exc_info.value)
    finally:
        lock.release()


def test_check_available_does_not_keep_the_lock(tmp_path):
    check_available(tmp_path)
    lock = ServerLock(tmp_path)
    lock.acquire()
    lock.release()


def test_store_lock_is_independent_of_server_lock(tmp_path):
    server_lock = ServerLock(tmp_path, port=8800)
    server_lock.acquire()
    try:
        with StoreLock(tmp_path, timeout=1.0):
            pass
    finally:
        server_lock.release()


def test_store_lock_held_does_not_block_server_lock(tmp_path):
    with StoreLock(tmp_path, timeout=1.0):
        lock = ServerLock(tmp_path)
        lock.acquire()
        lock.release()


def test_store_lock_works_from_another_thread_while_server_lock_held(tmp_path):
    import threading

    server_lock = ServerLock(tmp_path, port=8800)
    server_lock.acquire()
    errors: list[BaseException] = []

    def worker():
        try:
            with StoreLock(tmp_path, timeout=2.0):
                pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
    finally:
        server_lock.release()


def test_restore_role_message_does_not_say_server(tmp_path):
    lock = ServerLock(tmp_path, role="restore")
    lock.acquire()
    try:
        with pytest.raises(ServerLockHeld) as exc_info:
            ServerLock(tmp_path, port=8800).acquire()
        assert "restore" in str(exc_info.value)
    finally:
        lock.release()
