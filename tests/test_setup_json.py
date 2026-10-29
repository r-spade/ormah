"""Tests for the non-interactive JSON setup path used by the Mac app."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import ormah.setup as setup


@pytest.fixture(autouse=True)
def _isolate_claude_home(tmp_path, monkeypatch):
    """Structurally block every test in this file from touching the real
    ~/.claude — regardless of what the test under it does or forgets to do.

    _claude_code_wire()'s plugin-guard branch calls Path.home() directly, but
    its plain-wiring branch (configure_claude_hooks / configure_claude_code_mcp)
    calls os.path.expanduser("~/...") instead — patching Path.home alone leaves
    that path unguarded. Both resolve "~" via the HOME env var (pathlib.Path.home
    delegates to os.path.expanduser internally), so pinning HOME structurally
    blocks every call site in one move, regardless of which one a test reaches.
    A test that reaches either without this fixture would rewrite the
    developer's real ~/.claude/settings.json and ~/.claude.json. conftest.py's
    guard does not cover this (it only intercepts unlink/rmtree, and
    _REAL_ORMAH_PATHS does not list these files), so this file provides its
    own blanket isolation.
    """
    monkeypatch.setenv("HOME", str(tmp_path))


def set_detected(monkeypatch, *agent_ids: str) -> None:
    detected = set(agent_ids)
    for agent in setup.AGENT_REGISTRY:
        monkeypatch.setattr(agent, "detect_fn", lambda agent_id=agent.id: agent_id in detected)


def test_detect_clients_none(tmp_path):
    with (
        patch("ormah.setup._find_binary", return_value=None),
        patch("ormah.setup.Path.home", return_value=tmp_path),
        patch("platform.system", return_value="Darwin"),
    ):
        d = setup.detect_clients()
    assert {"claude_code", "codex", "claude_desktop", "pi"} <= d.keys()
    assert not any(d.values())


def test_detect_clients_claude_code(tmp_path):
    with (
        patch("ormah.setup._find_binary", side_effect=lambda n: "/bin/claude" if n == "claude" else None),
        patch("ormah.setup.Path.home", return_value=tmp_path),
        patch("platform.system", return_value="Linux"),
    ):
        d = setup.detect_clients()
    assert d["claude_code"] is True
    assert d["codex"] is False


def test_detect_clients_codex_via_dir(tmp_path):
    (tmp_path / ".codex").mkdir()
    with (
        patch("ormah.setup._find_binary", return_value=None),
        patch("ormah.setup.Path.home", return_value=tmp_path),
        patch("platform.system", return_value="Linux"),
    ):
        d = setup.detect_clients()
    assert d["codex"] is True


def test_run_setup_json_wires_detected(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(setup, "get_ormah_bin_path", lambda: "/bin/ormah")
    monkeypatch.setattr(setup, "_preload_local_models", lambda: None)
    set_detected(monkeypatch, "claude_code")
    for name in (
        "configure_claude_hooks", "configure_claude_code_mcp",
        "install_claude_md", "install_claude_agents", "install_claude_commands",
    ):
        monkeypatch.setattr(setup, name, lambda *a, _n=name: calls.append(_n))

    result = setup.run_setup_json()

    assert result["detected"] == ["claude_code"]
    assert result["wired"] == ["claude_code"]
    assert result["errors"] == {}
    assert result["warnings"] == {}
    assert "configure_claude_hooks" in calls


def test_run_setup_json_captures_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "get_ormah_bin_path", lambda: "/bin/ormah")
    monkeypatch.setattr(setup, "_preload_local_models", lambda: None)
    set_detected(monkeypatch, "claude_code")

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(setup, "configure_claude_hooks", boom)
    for name in (
        "configure_claude_code_mcp", "install_claude_md",
        "install_claude_agents", "install_claude_commands",
    ):
        monkeypatch.setattr(setup, name, lambda *a: None)

    result = setup.run_setup_json()

    assert result["wired"] == []
    assert "claude_code" in result["errors"]
    assert result["warnings"] == {}
    assert "RuntimeError" in result["errors"]["claude_code"]


def test_run_setup_json_wires_pi_from_registry(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(setup, "get_ormah_bin_path", lambda: "/bin/ormah")
    monkeypatch.setattr(setup, "_preload_local_models", lambda: None)
    set_detected(monkeypatch, "pi")
    monkeypatch.setattr(setup._get_agent("pi"), "wire_fn", lambda: calls.append("pi"))

    result = setup.run_setup_json()

    assert result["detected"] == ["pi"]
    assert result["wired"] == ["pi"]
    assert result["errors"] == {}
    assert calls == ["pi"]


def test_run_setup_json_preloads_models_and_keeps_stdout_clean(monkeypatch, capsys):
    calls: list[str] = []
    monkeypatch.setattr(setup, "get_ormah_bin_path", lambda: "/bin/ormah")
    for agent in setup.AGENT_REGISTRY:
        monkeypatch.setattr(agent, "detect_fn", lambda: False)

    def preload():
        print("preload progress")
        calls.append("preload")

    monkeypatch.setattr(setup, "_preload_local_models", preload)

    result = setup.run_setup_json()

    captured = capsys.readouterr()
    assert calls == ["preload"]
    assert captured.out == ""
    assert "preload progress" in captured.err
    assert result["errors"] == {}
    assert result["warnings"] == {}


def test_run_setup_json_preload_failure_is_warning_not_error(monkeypatch):
    monkeypatch.setattr(setup, "get_ormah_bin_path", lambda: "/bin/ormah")
    for agent in setup.AGENT_REGISTRY:
        monkeypatch.setattr(agent, "detect_fn", lambda: False)

    def preload():
        raise RuntimeError("model host unavailable")

    monkeypatch.setattr(setup, "_preload_local_models", preload)

    result = setup.run_setup_json()

    assert result["detected"] == []
    assert result["wired"] == []
    assert result["errors"] == {}
    assert result["warnings"]["models"] == "RuntimeError: model host unavailable"
