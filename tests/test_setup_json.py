"""Tests for the non-interactive JSON setup path used by the Mac app."""

from __future__ import annotations

from unittest.mock import patch

import ormah.setup as setup


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
    assert d == {
        "claude_code": False,
        "codex": False,
        "claude_desktop": False,
        "pi": False,
    }


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


def test_run_setup_json_wires_detected(monkeypatch):
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


def test_run_setup_json_captures_errors(monkeypatch):
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
