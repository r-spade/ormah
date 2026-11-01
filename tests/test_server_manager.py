"""Tests for server lifecycle helpers: port-conflict detection and launchd plist."""

from __future__ import annotations

import socket
from unittest import mock

import pytest

from ormah import server_manager


def test_is_port_in_use_true_when_socket_listening():
    """A bound, listening socket is reported as in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        host, port = srv.getsockname()
        assert server_manager.is_port_in_use(host, port) is True


def test_is_port_in_use_false_when_nothing_listening():
    """A port with no listener is reported as free."""
    # Grab an ephemeral port, then close it so nothing is listening.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    assert server_manager.is_port_in_use(host, port) is False


def test_is_port_in_use_false_for_unused_ipv6_literal():
    """An IPv6 host literal must not fail the pre-flight probe."""
    assert server_manager.is_port_in_use("::1", 0) is False


def test_is_port_in_use_true_when_ipv6_socket_listening():
    """An IPv6 listening socket is reported as in use when IPv6 is available."""
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as srv:
        try:
            srv.bind(("::1", 0))
        except OSError as exc:
            pytest.skip(f"IPv6 loopback unavailable: {exc}")

        srv.listen(1)
        _host, port, *_rest = srv.getsockname()
        assert server_manager.is_port_in_use("::1", port) is True


def test_terminate_pid_allows_graceful_shutdown_window():
    pid = 1234
    with mock.patch("ormah.server_manager.os.kill") as kill, \
         mock.patch("ormah.server_manager._wait_for_pid_exit", return_value=True) as wait:
        result = server_manager._terminate_pid(pid)

    assert result is True
    kill.assert_called_once_with(pid, server_manager.signal.SIGTERM)
    wait.assert_called_once_with(
        pid,
        timeout=server_manager._GRACEFUL_STOP_TIMEOUT_SECONDS,
    )


def test_terminate_pid_escalates_after_graceful_timeout():
    pid = 1234
    with mock.patch("ormah.server_manager.os.kill") as kill, \
         mock.patch(
             "ormah.server_manager._wait_for_pid_exit",
             side_effect=[False, True],
         ) as wait:
        result = server_manager._terminate_pid(pid)

    assert result is True
    assert kill.call_args_list == [
        mock.call(pid, server_manager.signal.SIGTERM),
        mock.call(pid, server_manager.signal.SIGKILL),
    ]
    assert wait.call_args_list == [
        mock.call(pid, timeout=server_manager._GRACEFUL_STOP_TIMEOUT_SECONDS),
        mock.call(pid, timeout=server_manager._FORCED_STOP_TIMEOUT_SECONDS),
    ]


def test_terminate_pid_fails_when_process_survives_sigkill():
    pid = 1234
    with mock.patch("ormah.server_manager.os.kill") as kill, \
         mock.patch("ormah.server_manager._wait_for_pid_exit", return_value=False):
        result = server_manager._terminate_pid(pid)

    assert result is False
    assert kill.call_args_list[-1] == mock.call(pid, server_manager.signal.SIGKILL)


def test_plist_keepalive_only_on_unsuccessful_exit():
    """KeepAlive must not be unconditional: a clean exit (port already taken)
    must not trigger a launchd respawn storm."""
    plist = server_manager.PLIST_TEMPLATE.format(
        label="com.ormah.server",
        wrapper_path="/tmp/wrapper",
        bin_dir="/tmp/bin",
    )
    assert "<key>SuccessfulExit</key><false/>" in plist.replace(" ", "").replace("\n", "")
    # The old unconditional form must be gone.
    assert "<key>KeepAlive</key><true/>" not in plist.replace(" ", "").replace("\n", "")


def test_plist_has_throttle_interval():
    """A ThrottleInterval backstops genuine crash loops."""
    plist = server_manager.PLIST_TEMPLATE.format(
        label="com.ormah.server",
        wrapper_path="/tmp/wrapper",
        bin_dir="/tmp/bin",
    )
    compact = plist.replace(" ", "").replace("\n", "")
    assert "<key>ThrottleInterval</key>" in compact


def test_server_start_exits_clean_when_port_in_use():
    """A healthy Ormah listener makes a duplicate foreground start a no-op."""
    from ormah import cli

    args = mock.Mock(daemon=False, reload=False)
    with mock.patch("ormah.server_manager.is_port_in_use", return_value=True), \
         mock.patch("ormah.server_manager.is_server_running", return_value=True), \
         mock.patch("uvicorn.run") as run:
        cli._cmd_server_start(args)
    run.assert_not_called()


def test_server_start_fails_when_foreign_process_owns_port():
    """A foreign listener must make the supervisor retry instead of going dormant."""
    from ormah import cli

    args = mock.Mock(daemon=False, reload=False)
    with mock.patch("ormah.server_manager.is_port_in_use", return_value=True), \
         mock.patch("ormah.server_manager.is_server_running", return_value=False), \
         mock.patch("uvicorn.run") as run, \
         pytest.raises(SystemExit) as exc_info:
        cli._cmd_server_start(args)

    assert exc_info.value.code == 1
    run.assert_not_called()


def test_server_start_runs_uvicorn_when_port_free():
    """When the port is free, uvicorn is launched as normal."""
    from ormah import cli

    args = mock.Mock(daemon=False, reload=False)
    with mock.patch("ormah.server_manager.is_port_in_use", return_value=False), \
         mock.patch("uvicorn.run") as run:
        cli._cmd_server_start(args)
    run.assert_called_once()


def test_restart_with_autostart_transfers_ownership_and_waits_for_health():
    stop_result = server_manager._StopServerResult(found=True, stopped=True)
    with mock.patch("ormah.server_manager._stop_running_server", return_value=stop_result) as stop, \
         mock.patch("ormah.server_manager.is_port_in_use", return_value=False) as port_in_use, \
         mock.patch("ormah.server_manager.install_autostart") as install, \
         mock.patch("ormah.server_manager.wait_for_server", return_value=True) as wait:
        result = server_manager.restart_with_autostart(
            "/abs/path/ormah",
            wrapper_path="/abs/path/ormah-server",
            show_progress=True,
        )

    assert result is True
    stop.assert_called_once()
    port_in_use.assert_called_once_with(server_manager.settings.host, server_manager.settings.port)
    install.assert_called_once_with("/abs/path/ormah", wrapper_path="/abs/path/ormah-server")
    wait.assert_called_once_with(show_progress=True)


def test_restart_with_autostart_fails_closed_when_stop_fails():
    stop_result = server_manager._StopServerResult(found=True, failed=True)
    with mock.patch("ormah.server_manager._stop_running_server", return_value=stop_result), \
         mock.patch("ormah.server_manager.install_autostart") as install, \
         mock.patch("ormah.server_manager.wait_for_server") as wait:
        result = server_manager.restart_with_autostart("/abs/path/ormah")

    assert result is False
    install.assert_not_called()
    wait.assert_not_called()


def test_restart_with_autostart_fails_closed_when_port_remains_occupied():
    stop_result = server_manager._StopServerResult(found=True, stopped=True)
    with mock.patch("ormah.server_manager._stop_running_server", return_value=stop_result), \
         mock.patch("ormah.server_manager.is_port_in_use", return_value=True), \
         mock.patch("ormah.server_manager.install_autostart") as install, \
         mock.patch("ormah.server_manager.wait_for_server") as wait:
        result = server_manager.restart_with_autostart("/abs/path/ormah")

    assert result is False
    install.assert_not_called()
    wait.assert_not_called()


def test_restart_with_autostart_fails_closed_when_install_fails():
    stop_result = server_manager._StopServerResult()
    with mock.patch("ormah.server_manager._stop_running_server", return_value=stop_result), \
         mock.patch("ormah.server_manager.is_port_in_use", return_value=False), \
         mock.patch("ormah.server_manager.install_autostart", side_effect=OSError("denied")), \
         mock.patch("ormah.server_manager.wait_for_server") as wait:
        result = server_manager.restart_with_autostart("/abs/path/ormah")

    assert result is False
    wait.assert_not_called()


def test_health_check_ignores_proxy_settings(monkeypatch):
    """A proxy can't reach this machine's localhost, so it must never be used for it."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(server_manager.settings, "port", server.server_port)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    try:
        assert server_manager.is_server_running()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8787", False),
        ("http://127.0.0.1:8787", False),
        ("http://[::1]:8787", False),
        ("https://localhost:8787", True),
        ("https://ormah.example.com", True),
        ("http://server.internal:8787", True),
    ],
)
def test_only_local_http_skips_env_proxy(url, expected):
    from ormah.config import uses_env_proxy

    assert uses_env_proxy(url) is expected
