"""Tests for ormah setup and server manager."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomllib

from ormah.server_manager import (
    LAUNCHD_LABEL,
    PLIST_TEMPLATE,
    SYSTEMD_TEMPLATE,
    get_ormah_bin_path,
    is_server_running,
)
from ormah.setup import (
    CloudRecoveryPreflightError,
    CODEX_AGENTS_SENTINEL_END,
    CODEX_AGENTS_SENTINEL_START,
    CLAUDE_MD_SENTINEL_END,
    CLAUDE_MD_SENTINEL_START,
    PI_AGENTS_MD_SENTINEL_END,
    PI_AGENTS_MD_SENTINEL_START,
    _get_agent,
    _atomic_write,
    _is_ormah_hook,
    _merge_hooks,
    _merge_json_file,
    _pi_is_wired,
    _preload_local_models,
    _print_setup_summary,
    _prepare_cloud_recovery,
    _remove_codex_hooks,
    _remove_codex_md_block,
    _remove_codex_mcp_config,
    _remove_config_preserving_cloud_recovery,
    _read_env_file,
    _remove_codex_agents,
    _remove_claude_hooks,
    _remove_claude_md_block,
    _remove_fastembed_cache,
    _remove_mcp_from_json,
    _remove_pi_agents,
    _remove_pi_extension,
    _remove_pi_md_block,
    _strip_ormah_hooks,
    _write_env_file,
    configure_claude_hooks,
    configure_claude_code_mcp,
    configure_claude_desktop,
    configure_agent_maintenance,
    configure_codex_hooks,
    configure_codex_mcp,
    configure_llm,
    configure_pi_extension,
    generate_server_wrapper,
    install_claude_md,
    install_codex_agents,
    install_codex_md,
    install_pi_agents,
    install_pi_md,
    run_setup,
    run_uninstall,
)


# --- server_manager tests ---


class TestGetOrmahBinPath:
    def test_returns_which_result(self):
        with patch("shutil.which", return_value="/usr/local/bin/ormah"):
            assert get_ormah_bin_path() == "/usr/local/bin/ormah"

    def test_fallback_to_sys_executable_dir(self, tmp_path):
        ormah_bin = tmp_path / "ormah"
        ormah_bin.touch()
        with (
            patch("shutil.which", return_value=None),
            patch("sys.executable", str(tmp_path / "python")),
        ):
            assert get_ormah_bin_path() == str(ormah_bin)

    def test_fallback_to_bare_name(self, tmp_path):
        with (
            patch("shutil.which", return_value=None),
            patch("sys.executable", str(tmp_path / "python")),
        ):
            assert get_ormah_bin_path() == "ormah"


class TestIsServerRunning:
    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with patch("ormah.server_manager.httpx.Client", return_value=mock_client):
            assert is_server_running() is True

    def test_returns_false_on_connection_error(self):
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")

        with patch("ormah.server_manager.httpx.Client", return_value=mock_client):
            assert is_server_running() is False


class TestPlistTemplate:
    def test_template_renders(self):
        rendered = PLIST_TEMPLATE.format(
            label=LAUNCHD_LABEL,
            wrapper_path="/home/user/.config/ormah/ormah-server",
            bin_dir="/usr/local/bin",
        )
        assert "<string>com.ormah.server</string>" in rendered
        assert "<string>/home/user/.config/ormah/ormah-server</string>" in rendered
        assert "<key>RunAtLoad</key><true/>" in rendered
        # KeepAlive must only respawn on failure — a clean exit (e.g. the port
        # is already owned by another server) must not trigger a respawn loop.
        compact = rendered.replace(" ", "").replace("\n", "")
        assert "<key>KeepAlive</key><true/>" not in compact
        assert "<key>SuccessfulExit</key><false/>" in compact
        assert "<key>ThrottleInterval</key>" in compact
        assert "StandardOutPath" not in rendered
        assert "StandardErrorPath" not in rendered

    def test_template_includes_path(self):
        rendered = PLIST_TEMPLATE.format(
            label=LAUNCHD_LABEL,
            wrapper_path="/home/user/.config/ormah/ormah-server",
            bin_dir="/home/user/.local/bin",
        )
        assert "<key>PATH</key><string>/home/user/.local/bin:" in rendered


class TestSystemdTemplate:
    def test_template_renders(self):
        rendered = SYSTEMD_TEMPLATE.format(
            wrapper_path="/home/user/.config/ormah/ormah-server",
            bin_dir="/usr/local/bin",
        )
        assert "ExecStart=/home/user/.config/ormah/ormah-server" in rendered
        assert "Restart=on-failure" in rendered
        assert "WantedBy=default.target" in rendered
        assert "After=network.target" in rendered
        assert 'Environment="PATH=/usr/local/bin:' in rendered
        assert "EnvironmentFile" not in rendered
        assert "StandardOutput" not in rendered
        assert "StandardError" not in rendered

    def test_template_renders_with_spaces_in_path(self):
        rendered = SYSTEMD_TEMPLATE.format(
            wrapper_path="/home/user/.config/ormah/ormah-server",
            bin_dir="/home/user/my apps",
        )
        assert "ExecStart=/home/user/.config/ormah/ormah-server" in rendered


class TestInstallAutostart:
    def test_linux_systemd_failure_falls_back_to_background(self, tmp_path, capsys):
        from ormah.server_manager import install_autostart

        unit_path = tmp_path / "ormah.service"
        error = subprocess.CalledProcessError(
            1,
            ["systemctl", "--user", "daemon-reload"],
            stderr=b"Failed to connect to bus: No medium found\n",
        )

        def fake_install_systemd(*args, **kwargs):
            unit_path.write_text("partial unit")
            raise error

        with (
            patch("platform.system", return_value="Linux"),
            patch("shutil.which", return_value="/usr/bin/systemctl"),
            patch("ormah.server_manager.SYSTEMD_UNIT", unit_path),
            patch("ormah.server_manager.install_systemd_service", side_effect=fake_install_systemd),
            patch("ormah.server_manager._start_server_background") as mock_start,
        ):
            install_autostart("/usr/local/bin/ormah", wrapper_path="/tmp/ormah-server")

        mock_start.assert_called_once_with("/tmp/ormah-server")
        assert not unit_path.exists()
        out = capsys.readouterr().out
        assert "User systemd is unavailable" in out
        assert "Failed to connect to bus" in out


# --- setup tests ---


class TestMergeJsonFile:
    def test_creates_new_file(self, tmp_path):
        path = str(tmp_path / "config.json")
        _merge_json_file(path, {"key": "value"})

        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}

    def test_merges_with_existing(self, tmp_path):
        path = str(tmp_path / "config.json")
        with open(path, "w") as f:
            json.dump({"existing": True, "nested": {"a": 1}}, f)

        _merge_json_file(path, {"nested": {"b": 2}, "new": "val"})

        with open(path) as f:
            data = json.load(f)
        assert data["existing"] is True
        assert data["nested"] == {"a": 1, "b": 2}
        assert data["new"] == "val"

    def test_handles_corrupt_file(self, tmp_path):
        path = str(tmp_path / "config.json")
        with open(path, "w") as f:
            f.write("not json{{{")

        _merge_json_file(path, {"key": "value"})

        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}


class TestConfigureClaudeHooks:
    def test_writes_hooks_with_absolute_path(self, tmp_path):
        settings_path = str(tmp_path / ".claude" / "settings.json")

        with patch("ormah.setup.os.path.expanduser", return_value=settings_path):
            configure_claude_hooks("/abs/path/ormah")

        with open(settings_path) as f:
            data = json.load(f)

        hooks = data["hooks"]
        assert "UserPromptSubmit" in hooks
        cmd = hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert cmd == "/abs/path/ormah whisper inject"

    def test_merges_with_existing_settings(self, tmp_path):
        settings_path = str(tmp_path / ".claude" / "settings.json")
        os.makedirs(os.path.dirname(settings_path))
        with open(settings_path, "w") as f:
            json.dump({"allowedTools": ["bash"]}, f)

        with patch("ormah.setup.os.path.expanduser", return_value=settings_path):
            configure_claude_hooks("/abs/path/ormah")

        with open(settings_path) as f:
            data = json.load(f)

        assert data["allowedTools"] == ["bash"]
        assert "hooks" in data

    def test_non_object_hooks_section_left_unchanged(self, tmp_path, capsys):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({"theme": "dark", "hooks": []}) + "\n")
        before = settings_path.read_text()

        with patch("ormah.setup.os.path.expanduser", return_value=str(settings_path)):
            configure_claude_hooks("/abs/ormah")

        assert settings_path.read_text() == before
        assert "Whisper hooks installed" not in capsys.readouterr().out

    def test_preserves_top_level_keys(self, tmp_path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(
            json.dumps({"theme": "dark", "permissions": {"allow": ["x"]}}) + "\n"
        )

        with patch("ormah.setup.os.path.expanduser", return_value=str(settings_path)):
            configure_claude_hooks("/abs/ormah")

        data = json.loads(settings_path.read_text())
        assert data["theme"] == "dark"
        assert data["permissions"] == {"allow": ["x"]}
        assert "UserPromptSubmit" in data["hooks"]


class TestConfigureClaudeCodeMcp:
    def test_writes_mcp_config_to_claude_json(self, tmp_path):
        config_path = str(tmp_path / ".claude.json")

        with patch("ormah.setup.shutil.which", return_value=None), \
             patch("ormah.setup.subprocess.run") as mock_run, \
             patch("ormah.setup.os.path.expanduser", return_value=config_path):
            configure_claude_code_mcp("/abs/path/ormah")

        mock_run.assert_not_called()

        with open(config_path) as f:
            data = json.load(f)

        assert data["mcpServers"]["ormah"]["command"] == "/abs/path/ormah"
        assert data["mcpServers"]["ormah"]["args"] == ["mcp"]

    def test_merges_with_existing_mcp_servers(self, tmp_path):
        config_path = str(tmp_path / ".claude.json")
        with open(config_path, "w") as f:
            json.dump({"mcpServers": {"fetch": {"command": "uvx"}}}, f)

        with patch("ormah.setup.shutil.which", return_value=None), \
             patch("ormah.setup.subprocess.run") as mock_run, \
             patch("ormah.setup.os.path.expanduser", return_value=config_path):
            configure_claude_code_mcp("/abs/path/ormah")

        mock_run.assert_not_called()

        with open(config_path) as f:
            data = json.load(f)

        assert "fetch" in data["mcpServers"]
        assert "ormah" in data["mcpServers"]

    def test_uses_claude_cli_when_available(self):
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("ormah.setup.shutil.which", return_value="/usr/local/bin/claude"), \
             patch("ormah.setup.subprocess.run", return_value=mock_result) as mock_run:
            configure_claude_code_mcp("/usr/local/bin/ormah")

        mock_run.assert_called_once_with(
            ["/usr/local/bin/claude", "mcp", "add", "ormah", "--scope", "user",
             "--", "/usr/local/bin/ormah", "mcp"],
            capture_output=True, text=True, timeout=10,
        )

    def test_cli_already_exists_removes_and_readds(self):
        first_result = MagicMock()
        first_result.returncode = 1
        first_result.stderr = "already exists"
        first_result.stdout = ""

        second_result = MagicMock()
        second_result.returncode = 0

        with patch("ormah.setup.shutil.which", return_value="/usr/local/bin/claude"), \
             patch("ormah.setup.subprocess.run", side_effect=[
                 first_result, second_result, second_result,
             ]) as mock_run:
            configure_claude_code_mcp("/usr/local/bin/ormah")

        assert mock_run.call_count == 3
        # First: add attempt
        # Second: remove
        assert mock_run.call_args_list[1][0][0] == [
            "/usr/local/bin/claude", "mcp", "remove", "ormah", "--scope", "user",
        ]
        # Third: re-add
        assert mock_run.call_args_list[2][0][0] == [
            "/usr/local/bin/claude", "mcp", "add", "ormah", "--scope", "user",
            "--", "/usr/local/bin/ormah", "mcp",
        ]


class TestConfigureClaudeDesktop:
    def test_skips_if_no_claude_desktop(self, tmp_path, capsys):
        config_dir = str(tmp_path / "nonexistent")

        with patch("ormah.setup.os.path.expanduser", return_value=config_dir):
            configure_claude_desktop("/abs/path/ormah")

        captured = capsys.readouterr()
        # Should silently skip (no output)
        assert "Connected to Claude Desktop" not in captured.out

    def test_writes_config_if_dir_exists(self, tmp_path):
        config_dir = tmp_path / "Claude"
        config_dir.mkdir()

        with (
            patch("ormah.setup.os.path.expanduser", return_value=str(config_dir)),
            patch("platform.system", return_value="Darwin"),
        ):
            configure_claude_desktop("/abs/path/ormah")

        config_path = config_dir / "claude_desktop_config.json"
        with open(config_path) as f:
            data = json.load(f)

        assert data["mcpServers"]["ormah"]["command"] == "/abs/path/ormah"


class TestConfigureCodexMcp:
    def test_writes_mcp_config_to_codex_toml(self, tmp_path):
        with (
            patch("ormah.setup.shutil.which", return_value=None),
            patch("ormah.setup.subprocess.run") as mock_run,
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            configure_codex_mcp("/abs/path/ormah")

        mock_run.assert_not_called()

        config_path = tmp_path / ".codex" / "config.toml"
        content = config_path.read_text()
        assert '[mcp_servers.ormah]' in content
        assert 'command = "/abs/path/ormah"' in content
        assert 'args = ["mcp"]' in content

    def test_preserves_existing_toml_content(self, tmp_path):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_path = config_dir / "config.toml"
        config_path.write_text(
            '[projects."/tmp/demo"]\n'
            'trust_level = "trusted"\n'
        )

        with (
            patch("ormah.setup.shutil.which", return_value=None),
            patch("ormah.setup.subprocess.run") as mock_run,
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            configure_codex_mcp("/abs/path/ormah")

        mock_run.assert_not_called()
        content = config_path.read_text()
        assert '[projects."/tmp/demo"]' in content
        assert 'trust_level = "trusted"' in content
        assert '[mcp_servers.ormah]' in content

    def test_replaces_existing_ormah_block(self, tmp_path):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_path = config_dir / "config.toml"
        config_path.write_text(
            '[mcp_servers.ormah]\n'
            'command = "/old/path/ormah"\n'
            'args = ["mcp"]\n\n'
            '[projects."/tmp/demo"]\n'
            'trust_level = "trusted"\n'
        )

        with (
            patch("ormah.setup.shutil.which", return_value=None),
            patch("ormah.setup.subprocess.run") as mock_run,
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            configure_codex_mcp("/new/path/ormah")

        mock_run.assert_not_called()
        content = config_path.read_text()
        assert 'command = "/new/path/ormah"' in content
        assert 'command = "/old/path/ormah"' not in content
        assert content.count('[mcp_servers.ormah]') == 1

    def test_uses_codex_cli_when_available(self):
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("ormah.setup.shutil.which", return_value="/usr/local/bin/codex"),
            patch("ormah.setup.subprocess.run", return_value=mock_result) as mock_run,
        ):
            configure_codex_mcp("/usr/local/bin/ormah")

        mock_run.assert_called_once_with(
            ["/usr/local/bin/codex", "mcp", "add", "ormah", "--",
             "/usr/local/bin/ormah", "mcp"],
            capture_output=True, text=True, timeout=10,
        )

    def test_cli_already_exists_removes_and_readds(self):
        first_result = MagicMock()
        first_result.returncode = 1
        first_result.stderr = "already exists"
        first_result.stdout = ""

        second_result = MagicMock()
        second_result.returncode = 0

        with (
            patch("ormah.setup.shutil.which", return_value="/usr/local/bin/codex"),
            patch("ormah.setup.subprocess.run", side_effect=[
                first_result, second_result, second_result,
            ]) as mock_run,
        ):
            configure_codex_mcp("/usr/local/bin/ormah")

        assert mock_run.call_count == 3
        assert mock_run.call_args_list[1][0][0] == [
            "/usr/local/bin/codex", "mcp", "remove", "ormah",
        ]
        assert mock_run.call_args_list[2][0][0] == [
            "/usr/local/bin/codex", "mcp", "add", "ormah", "--",
            "/usr/local/bin/ormah", "mcp",
        ]


class TestConfigureCodexHooks:
    def test_writes_hooks_and_enables_feature(self, tmp_path):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            configure_codex_hooks("/abs/path/ormah")

        hooks_path = tmp_path / ".codex" / "hooks.json"
        config_path = tmp_path / ".codex" / "config.toml"

        hooks_data = json.loads(hooks_path.read_text())
        assert hooks_data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == "/abs/path/ormah whisper inject"
        assert hooks_data["hooks"]["Stop"][0]["hooks"][0]["command"] == "/abs/path/ormah whisper store"

        content = config_path.read_text()
        assert "[features]" in content
        assert "hooks = true" in content
        assert "codex_hooks" not in content

    def test_preserves_existing_features_and_projects(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            '[projects."/tmp/demo"]\n'
            'trust_level = "trusted"\n\n'
            '[features]\n'
            'foo = true\n'
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            configure_codex_hooks("/abs/path/ormah")

        content = config_path.read_text()
        assert '[projects."/tmp/demo"]' in content
        assert 'trust_level = "trusted"' in content
        assert "[features]" in content
        assert "foo = true" in content
        assert "hooks = true" in content
        assert "codex_hooks" not in content

    def test_removes_deprecated_codex_hooks_feature(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            "[features]\n"
            "codex_hooks = true\n"
            "foo = true\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            configure_codex_hooks("/abs/path/ormah")

        content = config_path.read_text()
        assert "[features]" in content
        assert "foo = true" in content
        assert "hooks = true" in content
        assert "codex_hooks" not in content

    def test_merges_with_existing_hooks(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        hooks_path = codex_dir / "hooks.json"
        hooks_path.write_text(json.dumps({
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": "/bin/other"}]}
                ]
            }
        }, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            configure_codex_hooks("/abs/path/ormah")

        hooks_data = json.loads(hooks_path.read_text())
        assert "UserPromptSubmit" in hooks_data["hooks"]
        stop_cmds = [h["command"] for m in hooks_data["hooks"]["Stop"] for h in m["hooks"]]
        assert "/abs/path/ormah whisper store" in stop_cmds
        assert "/bin/other" in stop_cmds  # co-tenant preserved

    def test_non_object_hooks_section_no_false_success(self, tmp_path, capsys):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        hooks_path = codex_dir / "hooks.json"
        hooks_path.write_text(json.dumps({"hooks": "bad"}) + "\n")
        before = hooks_path.read_text()

        with patch("ormah.setup.Path.home", return_value=tmp_path), \
             patch("ormah.setup._enable_codex_feature") as enable:
            configure_codex_hooks("/abs/ormah")

        assert hooks_path.read_text() == before
        enable.assert_not_called()
        assert "Codex hooks installed" not in capsys.readouterr().out

    def test_non_list_event_no_false_success(self, tmp_path, capsys):
        """Non-list value on a claimed event (e.g. Stop) must leave file unchanged."""
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        hooks_path = codex_dir / "hooks.json"
        hooks_path.write_text(json.dumps({"hooks": {"Stop": "bad"}}) + "\n")
        before = hooks_path.read_text()

        with patch("ormah.setup.Path.home", return_value=tmp_path), \
             patch("ormah.setup._enable_codex_feature") as enable:
            configure_codex_hooks("/abs/ormah")

        assert hooks_path.read_text() == before
        enable.assert_not_called()
        assert "Codex hooks installed" not in capsys.readouterr().out


class TestRunSetup:
    def test_server_timeout_exits_nonzero_without_success_summary(self, tmp_path, capsys):
        with ExitStack() as stack:
            stack.enter_context(patch("ormah.setup.get_ormah_bin_path", return_value="/abs/path/ormah"))
            stack.enter_context(patch("ormah.setup.shutil.which", return_value=None))
            stack.enter_context(
                patch("ormah.setup.generate_server_wrapper", return_value=tmp_path / "ormah-server")
            )
            stack.enter_context(patch("ormah.setup.configure_llm"))
            stack.enter_context(patch("ormah.setup._preload_local_models"))
            stack.enter_context(patch("ormah.setup.is_server_running", return_value=False))
            mock_install = stack.enter_context(patch("ormah.setup.install_autostart"))
            stack.enter_context(patch("ormah.setup.wait_for_server", return_value=False))
            mock_diagnose = stack.enter_context(patch("ormah.setup._diagnose_server_failure"))
            mock_backfill = stack.enter_context(patch("ormah.setup.backfill_transcripts"))
            mock_finale = stack.enter_context(patch("ormah.setup.play_finale"))
            mock_summary = stack.enter_context(patch("ormah.setup._print_setup_summary"))
            mock_browser = stack.enter_context(patch("ormah.setup.webbrowser.open"))

            with pytest.raises(SystemExit) as exc_info:
                run_setup(skip_client_setup=True)

        assert exc_info.value.code == 1
        mock_install.assert_called_once_with("/abs/path/ormah", wrapper_path=str(tmp_path / "ormah-server"))
        mock_diagnose.assert_called_once()
        mock_backfill.assert_not_called()
        mock_finale.assert_not_called()
        mock_summary.assert_not_called()
        mock_browser.assert_not_called()

        out = capsys.readouterr().out
        assert "Setup incomplete" in out
        assert "Setup complete" not in out
        assert "Ormah is ready." not in out

    def test_skip_client_setup_avoids_client_wiring(self, tmp_path):
        def which(binary: str) -> str | None:
            return {
                "claude": "/usr/local/bin/claude",
                "codex": "/usr/local/bin/codex",
            }.get(binary)

        with ExitStack() as stack:
            stack.enter_context(patch("ormah.setup.get_ormah_bin_path", return_value="/abs/path/ormah"))
            stack.enter_context(patch("ormah.setup.shutil.which", side_effect=which))
            stack.enter_context(patch("ormah.setup.Path.home", return_value=tmp_path))
            stack.enter_context(
                patch("ormah.setup.generate_server_wrapper", return_value=tmp_path / "ormah-server")
            )
            stack.enter_context(patch("ormah.setup._preload_local_models"))
            stack.enter_context(patch("ormah.setup.is_server_running", return_value=True))
            mock_maintenance_prompt = stack.enter_context(patch("ormah.setup.configure_agent_maintenance"))
            mock_configure_llm = stack.enter_context(patch("ormah.setup.configure_llm"))
            mock_claude_hooks = stack.enter_context(patch("ormah.setup.configure_claude_hooks"))
            mock_claude_mcp = stack.enter_context(patch("ormah.setup.configure_claude_code_mcp"))
            mock_claude_md = stack.enter_context(patch("ormah.setup.install_claude_md"))
            mock_claude_agents = stack.enter_context(patch("ormah.setup.install_claude_agents"))
            mock_claude_commands = stack.enter_context(patch("ormah.setup.install_claude_commands"))
            mock_codex_hooks = stack.enter_context(patch("ormah.setup.configure_codex_hooks"))
            mock_codex_mcp = stack.enter_context(patch("ormah.setup.configure_codex_mcp"))
            mock_codex_md = stack.enter_context(patch("ormah.setup.install_codex_md"))
            mock_codex_agents = stack.enter_context(patch("ormah.setup.install_codex_agents"))
            mock_claude_desktop = stack.enter_context(patch("ormah.setup.configure_claude_desktop"))
            mock_pi_extension = stack.enter_context(patch("ormah.setup.configure_pi_extension"))
            mock_pi_md = stack.enter_context(patch("ormah.setup.install_pi_md"))
            mock_pi_agents = stack.enter_context(patch("ormah.setup.install_pi_agents"))
            stack.enter_context(patch("ormah.setup.backfill_transcripts"))
            stack.enter_context(patch("ormah.setup.play_finale"))
            stack.enter_context(patch("ormah.setup._print_setup_summary"))
            stack.enter_context(patch("ormah.setup.webbrowser.open"))
            run_setup(skip_client_setup=True)

        mock_maintenance_prompt.assert_not_called()
        mock_configure_llm.assert_called_once()
        mock_claude_hooks.assert_not_called()
        mock_claude_mcp.assert_not_called()
        mock_claude_md.assert_not_called()
        mock_claude_agents.assert_not_called()
        mock_claude_commands.assert_not_called()
        mock_codex_hooks.assert_not_called()
        mock_codex_mcp.assert_not_called()
        mock_codex_md.assert_not_called()
        mock_codex_agents.assert_not_called()
        mock_claude_desktop.assert_not_called()
        mock_pi_extension.assert_not_called()
        mock_pi_md.assert_not_called()
        mock_pi_agents.assert_not_called()

    def test_update_restarts_existing_server(self, tmp_path, capsys):
        from ormah.server_manager import _StopServerResult

        wrapper = tmp_path / "ormah-server"
        with ExitStack() as stack:
            stack.enter_context(patch("ormah.setup.get_ormah_bin_path", return_value="/abs/path/ormah"))
            stack.enter_context(patch("ormah.setup.shutil.which", return_value=None))
            stack.enter_context(patch("ormah.setup._read_env_file", return_value={}))
            stack.enter_context(patch("ormah.setup.generate_server_wrapper", return_value=wrapper))
            stack.enter_context(patch("ormah.setup._preload_local_models"))
            stack.enter_context(patch("ormah.setup.is_server_running", return_value=True))
            mock_stop = stack.enter_context(
                patch(
                    "ormah.setup._stop_running_server",
                    return_value=_StopServerResult(found=True, stopped=True),
                )
            )
            mock_install = stack.enter_context(patch("ormah.setup.install_autostart"))
            mock_wait = stack.enter_context(patch("ormah.setup.wait_for_server", return_value=True))
            stack.enter_context(patch("ormah.setup.backfill_transcripts"))
            stack.enter_context(patch("ormah.setup.play_finale"))
            stack.enter_context(patch("ormah.setup._print_setup_summary"))
            stack.enter_context(patch("ormah.setup.webbrowser.open"))

            run_setup(update=True, skip_client_setup=True)

        mock_stop.assert_called_once()
        mock_install.assert_called_once_with("/abs/path/ormah", wrapper_path=str(wrapper))
        mock_wait.assert_called_once_with(show_progress=True)

        out = capsys.readouterr().out
        assert "Restarting server" in out
        assert "Server already running" not in out


class TestClaudePluginManifest:
    def test_plugin_manifest_version_matches_project_version(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        plugin_manifest = json.loads(
            (root / "integrations" / "claude-plugin" / ".claude-plugin" / "plugin.json").read_text()
        )

        assert plugin_manifest["version"] == pyproject["project"]["version"]


class TestPiPluginPackage:
    def test_package_json_declares_pi_extension(self):
        root = Path(__file__).resolve().parents[1]
        pkg = json.loads((root / "integrations" / "pi-plugin" / "package.json").read_text())
        assert pkg["name"] == "ormah-pi"
        assert "./ormah-pi.ts" in pkg["pi"]["extensions"]

    def test_pi_resources_shipped(self):
        root = Path(__file__).resolve().parents[1]
        assert (root / "src" / "ormah" / "pi_instructions.md").exists()
        assert (root / "src" / "ormah" / "agents" / "ormah-pi-maintenance.md").exists()

    def test_entry_file_exists(self):
        root = Path(__file__).resolve().parents[1]
        assert (root / "integrations" / "pi-plugin" / "ormah-pi.ts").exists()


class TestClaudePluginDocs:
    def test_setup_command_requires_plugin_safe_flag_or_upgrade(self):
        root = Path(__file__).resolve().parents[1]
        content = (
            root / "integrations" / "claude-plugin" / "commands" / "setup.md"
        ).read_text()

        assert "command -v ormah" in content
        assert "ormah --version" in content
        assert "ormah setup --help" in content
        assert "ormah setup --skip-client-setup" in content
        assert "/ormah:upgrade" in content
        assert "ormah claude-md install" in content
        assert "CLAUDE.local.md" in content
        assert "Do not treat `ormah setup --update` as equivalent" in content

    def test_setup_playbook_matches_plugin_safe_upgrade_flow(self):
        root = Path(__file__).resolve().parents[1]
        content = (root / "integrations" / "claude-plugin" / "SETUP.md").read_text()

        assert "ormah --version" in content
        assert "ormah setup --help" in content
        assert "ormah setup --skip-client-setup" in content
        assert "/ormah:upgrade" in content
        assert "ormah claude-md install" in content
        assert "CLAUDE.local.md" in content
        assert "installed runtime is too old for plugin mode" in content

    def test_status_command_reports_installed_version(self):
        root = Path(__file__).resolve().parents[1]
        content = (
            root / "integrations" / "claude-plugin" / "commands" / "status.md"
        ).read_text()

        assert "command -v ormah" in content
        assert "ormah --version" in content
        assert "ormah server status" in content

    def test_upgrade_command_exists_with_plugin_safe_installer_flow(self):
        root = Path(__file__).resolve().parents[1]
        content = (
            root / "integrations" / "claude-plugin" / "commands" / "upgrade.md"
        ).read_text()

        assert "command -v ormah" in content
        assert "ormah --version" in content
        assert "ormah setup --help" in content
        assert "bash <(curl -fsSL https://ormah.me/install.sh) --no-setup" in content
        assert "Do not substitute `ormah setup --update`" in content

    def test_maintenance_command_exists(self):
        root = Path(__file__).resolve().parents[1]
        content = (
            root / "integrations" / "claude-plugin" / "commands" / "maintenance.md"
        ).read_text()

        assert 'subagent_type="ormah-maintenance"' in content
        assert "run_in_background=True" in content


# --- CLI tests ---


def test_setup_summary_prints_install_locations(capsys):
    _print_setup_summary("/abs/path/ormah")

    out = capsys.readouterr().out
    assert "Ormah is ready." in out
    assert 'What do you know about me?' not in out
    assert "CLI: /abs/path/ormah" in out
    assert "Config:" in out
    assert "Memory:" in out
    assert "Graph UI:" in out


class TestCliEntryPoint:
    def test_no_args_shows_help(self, capsys):
        from ormah.cli import main

        with patch("sys.argv", ["ormah"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_setup_calls_run_setup(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "setup"]),
            patch("ormah.setup.run_setup") as mock_setup,
        ):
            main()
            mock_setup.assert_called_once_with(
                ci=False,
                update=False,
                skip_client_setup=False,
            )

    def test_setup_ci_flag(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "setup", "--ci"]),
            patch("ormah.setup.run_setup") as mock_setup,
        ):
            main()
            mock_setup.assert_called_once_with(
                ci=True,
                update=False,
                skip_client_setup=False,
            )

    def test_setup_skip_client_setup_flag(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "setup", "--skip-client-setup"]),
            patch("ormah.setup.run_setup") as mock_setup,
        ):
            main()
            mock_setup.assert_called_once_with(
                ci=False,
                update=False,
                skip_client_setup=True,
            )

    def test_server_start_daemon_exits_nonzero_on_timeout(self, tmp_path):
        from ormah.cli import main

        wrapper = tmp_path / "ormah-server"
        with (
            patch("sys.argv", ["ormah", "server", "start", "-d"]),
            patch("ormah.setup.WRAPPER_PATH", wrapper),
            patch("ormah.setup.generate_server_wrapper", return_value=wrapper),
            patch("ormah.server_manager.get_ormah_bin_path", return_value="/abs/path/ormah"),
            patch("ormah.server_manager.install_autostart"),
            patch("ormah.server_manager.wait_for_server", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_claude_md_install_defaults_to_auto_scope(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "claude-md", "install"]),
            patch("ormah.setup.install_claude_md") as mock_install,
        ):
            main()
            mock_install.assert_called_once_with(scope="auto", cwd=Path.cwd())

    def test_claude_md_install_user_scope(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "claude-md", "install", "--scope", "user"]),
            patch("ormah.setup.install_claude_md") as mock_install,
        ):
            main()
            mock_install.assert_called_once_with(scope="user", cwd=Path.cwd())

    def test_pi_md_install_defaults_to_user_scope(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "pi-md", "install"]),
            patch("ormah.setup.install_pi_md") as mock_install,
        ):
            main()
            mock_install.assert_called_once_with(scope="user", cwd=Path.cwd())

    def test_pi_md_install_project_scope(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "pi-md", "install", "--scope", "project"]),
            patch("ormah.setup.install_pi_md") as mock_install,
        ):
            main()
            mock_install.assert_called_once_with(scope="project", cwd=Path.cwd())

    def test_server_status_when_not_running(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "server", "status"]),
            patch("ormah.server_manager.is_server_running", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    def test_mcp_delegates_to_adapter(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "mcp"]),
            patch("ormah.adapters.mcp_adapter.main") as mock_mcp,
        ):
            main()
            mock_mcp.assert_called_once()


# --- env file tests ---


class TestEnvFile:
    def test_write_and_read(self, tmp_path):
        env_path = tmp_path / ".env"
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"FOO": "bar", "BAZ": "123"})
            result = _read_env_file()
        assert result == {"FOO": "bar", "BAZ": "123"}

    def test_read_skips_comments_and_blanks(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("# comment\n\nKEY=val\n")
        with patch("ormah.setup.ENV_PATH", env_path):
            result = _read_env_file()
        assert result == {"KEY": "val"}

    def test_read_nonexistent_returns_empty(self, tmp_path):
        env_path = tmp_path / "nope"
        with patch("ormah.setup.ENV_PATH", env_path):
            assert _read_env_file() == {}

    def test_write_sets_600_permissions(self, tmp_path):
        env_path = tmp_path / ".env"
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"SECRET": "value"})
        file_mode = stat.S_IMODE(env_path.stat().st_mode)
        assert file_mode == 0o600


class TestWriteEnvPreservation:
    def test_atomic_write_preserves_relative_symlink(self, tmp_path):
        target = tmp_path / "managed.env"
        target.write_text("A=old\n")
        target.chmod(0o644)
        link = tmp_path / ".env"
        link.symlink_to(target.name)

        _atomic_write(str(link), "A=new\n", mode=0o600)

        assert link.is_symlink()
        assert link.read_text() == "A=new\n"
        assert target.read_text() == "A=new\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_preserves_comments_and_manual_key(self, tmp_path):
        from ormah.setup import _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("# header comment\nMANUAL_KEY=keep\n\nORMAH_X=old\n")
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"MANUAL_KEY": "keep", "ORMAH_X": "new"})
        text = env_path.read_text()
        assert "# header comment" in text
        assert "MANUAL_KEY=keep" in text
        assert "ORMAH_X=new" in text
        assert "ORMAH_X=old" not in text

    def test_removed_key_dropped_comments_kept(self, tmp_path):
        from ormah.setup import _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("# keep me\nDROP=1\nKEEP=2\n")
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"KEEP": "2"})
        text = env_path.read_text()
        assert "# keep me" in text
        assert "KEEP=2" in text
        assert "DROP" not in text

    def test_new_key_appended(self, tmp_path):
        from ormah.setup import _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("# c\nA=1\n")
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"A": "1", "B": "2"})
        lines = [ln for ln in env_path.read_text().splitlines() if ln.strip()]
        assert lines[-1] == "B=2"
        assert "# c" in env_path.read_text()

    def test_nonexistent_file_writes_dict_order(self, tmp_path):
        from ormah.setup import _write_env_file

        env_path = tmp_path / ".env"
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"A": "1", "B": "2"})
        assert env_path.read_text() == "A=1\nB=2\n"

    def test_untouched_key_with_inline_comment_preserved(self, tmp_path):
        from ormah.setup import _read_env_file, _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("MANUAL=val  # keep this note\n")
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            env = _read_env_file()
            _write_env_file(env)
        assert "# keep this note" in env_path.read_text()

    def test_configure_llm_flow_preserves_block_comment(self, tmp_path):
        from ormah.setup import _read_env_file, _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("# my ormah config\nORMAH_LLM_PROVIDER=none\n")
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            env = _read_env_file()
            env["ORMAH_LLM_PROVIDER"] = "ollama"
            _write_env_file(env)
        text = env_path.read_text()
        assert "# my ormah config" in text
        assert "ORMAH_LLM_PROVIDER=ollama" in text

    def test_duplicate_keys_collapsed(self, tmp_path):
        from ormah.setup import _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("DUP=1\nDUP=2\n")
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"DUP": "2"})
        text = env_path.read_text()
        assert text.count("DUP=") == 1
        assert "DUP=2" in text

    def test_existing_file_mode_forced_to_600(self, tmp_path):
        from ormah.setup import _write_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("A=1\n")
        env_path.chmod(0o644)
        with patch("ormah.setup.ENV_PATH", env_path), patch("ormah.setup.ENV_DIR", tmp_path):
            _write_env_file({"A": "1"})
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


# --- Server wrapper tests ---


class TestGenerateServerWrapper:
    def test_creates_wrapper_file(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            result = generate_server_wrapper("/usr/local/bin/ormah")
        assert result == wrapper
        assert wrapper.exists()

    def test_wrapper_has_700_permissions(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        file_mode = stat.S_IMODE(wrapper.stat().st_mode)
        assert file_mode == 0o700

    def test_wrapper_contains_ormah_bin(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert "exec /usr/local/bin/ormah server start" in content

    def test_wrapper_uses_explicit_api_key_policy(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert "ORMAH_LLM_API_KEY_ENV_VAR" in content
        assert "ORMAH_LLM_INHERIT_API_KEY" in content
        assert "ANTHROPIC_API_KEY" in content
        assert "OPENAI_API_KEY" in content
        assert "AWS_SECRET_ACCESS_KEY" not in content
        assert "grep -E" not in content

    def test_wrapper_no_hardcoded_secrets(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert "sk-ant-" not in content
        assert "sk-" not in content.replace("#!/", "")  # ignore shebang

    def test_idempotent(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
            first_content = wrapper.read_text()
            generate_server_wrapper("/usr/local/bin/ormah")
            second_content = wrapper.read_text()
        assert first_content == second_content

    def test_sources_env_file(self, tmp_path):
        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert '.config/ormah/.env' in content
        assert "ORMAH_*=*" in content
        assert "set -a" not in content

    def test_wrapper_imports_only_selected_api_key_at_runtime(self, tmp_path):
        home = tmp_path / "home"
        config_dir = home / ".config" / "ormah"
        config_dir.mkdir(parents=True)
        config_dir.joinpath(".env").write_text(
            "ORMAH_LLM_PROVIDER=litellm\n"
            "ORMAH_LLM_MODEL=claude-haiku-4-5-20251001\n"
            "ORMAH_LLM_API_KEY_ENV_VAR=ANTHROPIC_API_KEY\n"
            "ORMAH_LLM_INHERIT_API_KEY=true\n"
        )

        fake_shell = tmp_path / "fake-shell"
        fake_shell.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' 'ANTHROPIC_API_KEY=sk-ant-selected-secret'\n"
            "printf '%s\\n' 'OPENAI_API_KEY=sk-other-secret'\n"
            "printf '%s\\n' 'AWS_SECRET_ACCESS_KEY=aws-secret'\n"
        )
        fake_shell.chmod(0o700)

        capture = tmp_path / "env-capture.txt"
        fake_ormah = tmp_path / "ormah"
        fake_ormah.write_text("#!/usr/bin/env bash\nenv > \"$ENV_CAPTURE\"\n")
        fake_ormah.chmod(0o700)

        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", config_dir):
            generate_server_wrapper(str(fake_ormah))

        subprocess.run(
            [str(wrapper)],
            check=True,
            env={
                "HOME": str(home),
                "SHELL": str(fake_shell),
                "ENV_CAPTURE": str(capture),
                "PATH": os.environ.get("PATH", ""),
            },
        )

        captured = capture.read_text()
        assert "ANTHROPIC_API_KEY=sk-ant-selected-secret" in captured
        assert "OPENAI_API_KEY=" not in captured
        assert "AWS_SECRET_ACCESS_KEY=" not in captured

    def test_wrapper_imports_no_api_key_when_llm_disabled(self, tmp_path):
        home = tmp_path / "home"
        config_dir = home / ".config" / "ormah"
        config_dir.mkdir(parents=True)
        config_dir.joinpath(".env").write_text(
            "ORMAH_LLM_PROVIDER=none\n"
            "ORMAH_LLM_API_KEY_ENV_VAR=ANTHROPIC_API_KEY\n"
            "ORMAH_LLM_INHERIT_API_KEY=true\n"
        )

        fake_shell = tmp_path / "fake-shell"
        fake_shell.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' 'ANTHROPIC_API_KEY=sk-ant-selected-secret'\n"
        )
        fake_shell.chmod(0o700)

        capture = tmp_path / "env-capture.txt"
        fake_ormah = tmp_path / "ormah"
        fake_ormah.write_text("#!/usr/bin/env bash\nenv > \"$ENV_CAPTURE\"\n")
        fake_ormah.chmod(0o700)

        wrapper = tmp_path / "ormah-server"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", config_dir):
            generate_server_wrapper(str(fake_ormah))

        subprocess.run(
            [str(wrapper)],
            check=True,
            env={
                "HOME": str(home),
                "SHELL": str(fake_shell),
                "ENV_CAPTURE": str(capture),
                "PATH": os.environ.get("PATH", ""),
            },
        )

        assert "ANTHROPIC_API_KEY=" not in capture.read_text()


# --- LLM configuration tests ---


class TestConfigureLlm:
    def _clear_all_api_keys(self, monkeypatch):
        """Remove known API keys from env so provider setup sees no key."""
        for key in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
            "AZURE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_anthropic_no_key_in_env_disables_llm(self, tmp_path, monkeypatch):
        """Selecting Anthropic without a key keeps server-side LLM disabled."""
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        inputs = iter(["y", "1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "none"
        assert "ORMAH_LLM_MODEL" not in result
        assert "ANTHROPIC_API_KEY" not in result

    def test_openai_no_key_in_env_disables_llm(self, tmp_path, monkeypatch):
        """Selecting OpenAI without a key keeps server-side LLM disabled."""
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        inputs = iter(["y", "3"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "none"
        assert "ORMAH_LLM_MODEL" not in result
        assert "OPENAI_API_KEY" not in result

    def test_ollama_no_key_needed(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        inputs = iter(["y", "5"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "ollama"
        assert result["ORMAH_LLM_MODEL"] == "llama3.2"
        assert "ANTHROPIC_API_KEY" not in result

    def test_none_sets_provider_none(self, tmp_path, monkeypatch, capsys):
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        inputs = iter(["y", "6"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "none"
        assert "ORMAH_LLM_MODEL" not in result

        captured = capsys.readouterr()
        assert "No LLM configured" in captured.out
        assert "Run 'ormah setup' again to add an LLM later" in captured.out

    def test_explicit_key_inheritance_does_not_store_key_value(self, tmp_path, monkeypatch):
        """Opt-in stores key policy but never the API key value."""
        env_path = tmp_path / ".env"
        inputs = iter(["y", "1", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "litellm"
        assert result["ORMAH_LLM_MODEL"] == "claude-haiku-4-5-20251001"
        assert result["ORMAH_LLM_API_KEY_ENV_VAR"] == "ANTHROPIC_API_KEY"
        assert result["ORMAH_LLM_INHERIT_API_KEY"] == "true"
        assert "ANTHROPIC_API_KEY" not in result
        assert "sk-ant-from-env" not in env_path.read_text()

    def test_declining_key_inheritance_disables_llm(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        inputs = iter(["y", "1", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "none"
        assert "ORMAH_LLM_API_KEY_ENV_VAR" not in result
        assert "sk-ant-from-env" not in env_path.read_text()

    def test_preserves_existing_env_values(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text(
            "ORMAH_PORT=9999\n"
            "ORMAH_LLM_API_KEY_ENV_VAR=ANTHROPIC_API_KEY\n"
            "ORMAH_LLM_INHERIT_API_KEY=true\n"
        )
        self._clear_all_api_keys(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _: "")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_PORT"] == "9999"
        assert result["ORMAH_LLM_PROVIDER"] == "none"
        assert "ORMAH_LLM_API_KEY_ENV_VAR" not in result
        assert "ORMAH_LLM_INHERIT_API_KEY" not in result


# --- CLAUDE.md installation tests ---


class TestInstallClaudeMd:
    def test_creates_new_file(self, tmp_path, capsys):
        claude_md = tmp_path / ".claude" / "CLAUDE.md"

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_claude_md()

        content = claude_md.read_text()
        assert CLAUDE_MD_SENTINEL_START in content
        assert CLAUDE_MD_SENTINEL_END in content
        assert "# Ormah Memory System" in content
        assert "remember" in content

        captured = capsys.readouterr()
        assert "Instructions added to ~/.claude/CLAUDE.md" in captured.out

    def test_appends_to_existing_content(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text("# My existing instructions\n\nDo things my way.\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_claude_md()

        content = claude_md.read_text()
        assert content.startswith("# My existing instructions\n\nDo things my way.\n")
        assert CLAUDE_MD_SENTINEL_START in content
        assert "# Ormah Memory System" in content

    def test_idempotent_replace(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"

        # Run twice
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_claude_md()
            first_content = claude_md.read_text()
            install_claude_md()
            second_content = claude_md.read_text()

        assert first_content == second_content

    def test_preserves_content_around_sentinels(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text(
            "# Before\n\n"
            f"{CLAUDE_MD_SENTINEL_START}\nold content\n{CLAUDE_MD_SENTINEL_END}\n"
            "\n# After\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_claude_md()

        content = claude_md.read_text()
        assert content.startswith("# Before\n\n")
        assert content.endswith("\n# After\n")
        assert "old content" not in content
        assert "# Ormah Memory System" in content

    def test_project_scope_writes_to_project_claude_md(self, tmp_path, capsys):
        with patch("ormah.setup.Path.cwd", return_value=tmp_path):
            install_claude_md(scope="project")

        project_claude_md = tmp_path / "CLAUDE.md"
        content = project_claude_md.read_text()
        assert CLAUDE_MD_SENTINEL_START in content
        assert CLAUDE_MD_SENTINEL_END in content
        assert "# Ormah Memory System" in content

        captured = capsys.readouterr()
        assert "Instructions added to ./CLAUDE.md" in captured.out

    def test_local_scope_writes_to_project_local_claude_md(self, tmp_path, capsys):
        with patch("ormah.setup.Path.cwd", return_value=tmp_path):
            install_claude_md(scope="local")

        local_claude_md = tmp_path / "CLAUDE.local.md"
        content = local_claude_md.read_text()
        assert CLAUDE_MD_SENTINEL_START in content
        assert CLAUDE_MD_SENTINEL_END in content
        assert "# Ormah Memory System" in content

        captured = capsys.readouterr()
        assert "Instructions added to ./CLAUDE.local.md" in captured.out

    def test_auto_scope_uses_local_plugin_settings_when_present(self, tmp_path, capsys):
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        (settings_dir / "settings.local.json").write_text(
            json.dumps({"enabledPlugins": {"ormah@claude-plugins-official": True}})
        )

        install_claude_md(scope="auto", cwd=tmp_path)

        local_claude_md = tmp_path / "CLAUDE.local.md"
        assert local_claude_md.exists()

        captured = capsys.readouterr()
        assert "Instructions added to ./CLAUDE.local.md" in captured.out

    def test_auto_scope_uses_project_plugin_settings_when_present(self, tmp_path, capsys):
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        (settings_dir / "settings.json").write_text(
            json.dumps({"enabledPlugins": {"ormah@claude-plugins-official": True}})
        )

        install_claude_md(scope="auto", cwd=tmp_path)

        project_claude_md = tmp_path / "CLAUDE.md"
        assert project_claude_md.exists()

        captured = capsys.readouterr()
        assert "Instructions added to ./CLAUDE.md" in captured.out

    def test_auto_scope_uses_user_plugin_settings_when_present(self, tmp_path, capsys):
        home_claude_dir = tmp_path / ".claude"
        home_claude_dir.mkdir()
        (home_claude_dir / "settings.json").write_text(
            json.dumps({"enabledPlugins": {"ormah@claude-plugins-official": True}})
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_claude_md(scope="auto", cwd=tmp_path / "repo")

        user_claude_md = home_claude_dir / "CLAUDE.md"
        assert user_claude_md.exists()

        captured = capsys.readouterr()
        assert "Instructions added to ~/.claude/CLAUDE.md" in captured.out


class TestInstallCodexMd:
    def test_creates_agents_md(self, tmp_path, capsys):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_codex_md()

        agents_md = tmp_path / ".codex" / "AGENTS.md"
        content = agents_md.read_text()
        assert CODEX_AGENTS_SENTINEL_START in content
        assert CODEX_AGENTS_SENTINEL_END in content
        assert "# Ormah Memory System" in content

        captured = capsys.readouterr()
        assert "Instructions added to" in captured.out
        assert "AGENTS.md" in captured.out

    def test_uses_override_file_when_present(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        override_md = codex_dir / "AGENTS.override.md"
        override_md.write_text("# Existing override\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_codex_md()

        content = override_md.read_text()
        assert content.startswith("# Existing override\n")
        assert CODEX_AGENTS_SENTINEL_START in content
        assert not (codex_dir / "AGENTS.md").exists()

    def test_idempotent_replace(self, tmp_path):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_codex_md()
            first = (tmp_path / ".codex" / "AGENTS.md").read_text()
            install_codex_md()
            second = (tmp_path / ".codex" / "AGENTS.md").read_text()

        assert first == second


class TestInstallCodexAgents:
    def test_creates_agent_file(self, tmp_path, capsys):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_codex_agents()

        agent_file = tmp_path / ".codex" / "agents" / "ormah-maintenance.toml"
        content = agent_file.read_text()
        assert 'name = "ormah-maintenance"' in content
        assert 'mcp_servers = ["ormah"]' not in content
        assert 'sandbox_mode = "read-only"' in content

        captured = capsys.readouterr()
        assert "Codex" in captured.out

    def test_overwrites_existing_agent_file(self, tmp_path):
        agent_dir = tmp_path / ".codex" / "agents"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "ormah-maintenance.toml"
        agent_file.write_text('name = "old"\n')

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_codex_agents()

        content = agent_file.read_text()
        assert 'name = "ormah-maintenance"' in content
        assert 'name = "old"' not in content


class TestConfigureAgentMaintenance:
    def test_enables_claude_code_maintenance(self, tmp_path, monkeypatch, capsys):
        env_path = tmp_path / ".env"
        monkeypatch.setattr("builtins.input", lambda _: "")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            result = configure_agent_maintenance([_get_agent("claude_code")])

        assert result is True
        assert "ORMAH_CLAUDE_MAINTENANCE_ENABLED=true" in env_path.read_text()
        assert not (tmp_path / ".codex" / "config.toml").exists()

        captured = capsys.readouterr()
        assert "Claude Code" in captured.out

    def test_enables_codex_multi_agent_without_clobbering_existing_config(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            '[projects."/tmp/demo"]\n'
            'trust_level = "trusted"\n\n'
            '[features]\n'
            'multi_agent = false\n'
            'foo = true\n'
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            result = configure_agent_maintenance([_get_agent("codex")])

        assert result is True
        assert "ORMAH_CLAUDE_MAINTENANCE_ENABLED=true" in env_path.read_text()
        content = config_path.read_text()
        assert '[projects."/tmp/demo"]' in content
        assert 'trust_level = "trusted"' in content
        assert 'foo = true' in content
        assert 'multi_agent = true' in content
        assert 'multi_agent = false' not in content

    def test_mentions_both_clients_when_available(self, tmp_path, monkeypatch, capsys):
        env_path = tmp_path / ".env"
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            result = configure_agent_maintenance(
                [_get_agent("claude_code"), _get_agent("codex")]
            )

        assert result is False
        assert not env_path.exists()

        captured = capsys.readouterr()
        assert "Claude Code or Codex" in captured.out
        assert "Skipped automatic maintenance" in captured.out

    def test_enables_pi_maintenance(self, tmp_path, monkeypatch, capsys):
        env_path = tmp_path / ".env"
        monkeypatch.setattr("builtins.input", lambda _: "")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
            patch("ormah.setup.Path.home", return_value=tmp_path),
        ):
            result = configure_agent_maintenance([_get_agent("pi")])

        assert result is True
        env = env_path.read_text()
        assert "ORMAH_CLAUDE_MAINTENANCE_ENABLED=true" in env
        assert "ORMAH_PI_MAINTENANCE_ENABLED" not in env

        captured = capsys.readouterr()
        assert "Pi" in captured.out


class TestInstallPiMd:
    def test_respects_pi_agent_dir_override(self, tmp_path, monkeypatch):
        pi_dir = tmp_path / "custom-pi-agent"
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))

        install_pi_md()
        install_pi_agents()

        assert (pi_dir / "AGENTS.md").exists()
        assert (pi_dir / "agents" / "ormah-maintenance.md").exists()

    def test_creates_new_file(self, tmp_path, capsys):
        agents_md = tmp_path / ".pi" / "agent" / "AGENTS.md"

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_pi_md()

        content = agents_md.read_text()
        assert PI_AGENTS_MD_SENTINEL_START in content
        assert PI_AGENTS_MD_SENTINEL_END in content
        assert "# Ormah Memory System" in content
        assert "ormah_remember" in content

        captured = capsys.readouterr()
        assert "Instructions added to ~/.pi/agent/AGENTS.md" in captured.out

    def test_appends_to_existing_content(self, tmp_path):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)
        agents_md = pi_dir / "AGENTS.md"
        agents_md.write_text("# My existing instructions\n\nDo things my way.\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_pi_md()

        content = agents_md.read_text()
        assert content.startswith("# My existing instructions\n\nDo things my way.\n")
        assert PI_AGENTS_MD_SENTINEL_START in content
        assert "# Ormah Memory System" in content

    def test_idempotent_replace(self, tmp_path):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)
        agents_md = pi_dir / "AGENTS.md"

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_pi_md()
            first = agents_md.read_text()
            install_pi_md()
            second = agents_md.read_text()

        assert first == second

    def test_project_scope_writes_to_project_agents_md(self, tmp_path, capsys):
        with patch("ormah.setup.Path.cwd", return_value=tmp_path):
            install_pi_md(scope="project")

        project_agents_md = tmp_path / "AGENTS.md"
        content = project_agents_md.read_text()
        assert PI_AGENTS_MD_SENTINEL_START in content
        assert PI_AGENTS_MD_SENTINEL_END in content
        assert "# Ormah Memory System" in content

        captured = capsys.readouterr()
        assert "Instructions added to ./AGENTS.md" in captured.out


class TestInstallPiAgents:
    def test_creates_agent_file(self, tmp_path, capsys):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_pi_agents()

        agent_file = tmp_path / ".pi" / "agent" / "agents" / "ormah-maintenance.md"
        content = agent_file.read_text()
        assert "ormah_run_maintenance" in content
        assert "name: ormah-maintenance" in content

        captured = capsys.readouterr()
        assert "Pi" in captured.out

    def test_overwrites_existing_agent_file(self, tmp_path):
        agent_dir = tmp_path / ".pi" / "agent" / "agents"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "ormah-maintenance.md"
        agent_file.write_text("# old\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            install_pi_agents()

        content = agent_file.read_text()
        assert "ormah_run_maintenance" in content
        assert "# old" not in content


class TestConfigurePiExtension:
    def test_partial_wiring_never_reports_connected(self, tmp_path):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)
        (pi_dir / "settings.json").write_text(
            json.dumps({"packages": ["npm:ormah-pi"]})
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            assert _pi_is_wired() is False

            (pi_dir / "AGENTS.md").write_text(
                f"{PI_AGENTS_MD_SENTINEL_START}\n{PI_AGENTS_MD_SENTINEL_END}\n"
            )
            assert _pi_is_wired() is False

            agents_dir = pi_dir / "agents"
            agents_dir.mkdir()
            (agents_dir / "ormah-maintenance.md").write_text(
                "Use ormah_run_maintenance."
            )
            assert _pi_is_wired() is True

    def test_detects_extension_via_settings_packages(self, tmp_path, capsys):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)
        (pi_dir / "settings.json").write_text(json.dumps({"packages": ["npm:ormah-pi"]}))

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            configure_pi_extension("/abs/path/ormah")

        captured = capsys.readouterr()
        assert "ormah-pi extension detected" in captured.out

    def test_installs_extension_when_missing(self, tmp_path, capsys):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)

        def install(*_args, **_kwargs):
            (pi_dir / "settings.json").write_text(
                json.dumps({"packages": ["npm:ormah-pi"]})
            )
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.setup._find_binary", return_value="/usr/bin/pi"),
            patch("ormah.setup.subprocess.run", side_effect=install) as mock_run,
        ):
            configure_pi_extension("/abs/path/ormah")

        captured = capsys.readouterr()
        assert "ormah-pi extension installed" in captured.out
        mock_run.assert_called_once_with(
            ["/usr/bin/pi", "install", "npm:ormah-pi"],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_install_failure_is_reported(self, tmp_path):
        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.setup._find_binary", return_value="/usr/bin/pi"),
            patch(
                "ormah.setup.subprocess.run",
                return_value=MagicMock(returncode=1, stdout="", stderr="not found"),
            ),
            pytest.raises(RuntimeError, match="not found"),
        ):
            configure_pi_extension("/abs/path/ormah")


# --- Uninstall tests ---


class TestRemoveClaudeHooks:
    def _make_settings(self, tmp_path: Path, data: dict) -> Path:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(data, indent=2) + "\n")
        return settings_path

    def test_removes_inject_and_store_hooks(self, tmp_path):
        data = {
            "hooks": {
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "/usr/bin/ormah whisper inject", "timeout": 10}]}
                ],
                "SessionEnd": [
                    {"hooks": [{"type": "command", "command": "/usr/bin/ormah whisper store", "timeout": 300}]}
                ],
            }
        }
        self._make_settings(tmp_path, data)

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            (tmp_path / ".claude").mkdir()
            (tmp_path / ".claude" / "settings.json").write_text(json.dumps(data, indent=2) + "\n")
            _remove_claude_hooks()
            result = json.loads((tmp_path / ".claude" / "settings.json").read_text())

        assert "hooks" not in result

    def test_preserves_non_ormah_hooks(self, tmp_path):
        data = {
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {"type": "command", "command": "/usr/bin/ormah whisper inject"},
                            {"type": "command", "command": "/usr/bin/other-tool run"},
                        ]
                    }
                ],
            }
        }
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps(data, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()
            result = json.loads((claude_dir / "settings.json").read_text())

        hooks = result["hooks"]["UserPromptSubmit"][0]["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["command"] == "/usr/bin/other-tool run"

    def test_preserves_untouched_empty_and_missing_hooks_matchers(self, tmp_path):
        data = {
            "hooks": {
                "UserPromptSubmit": [
                    {"matcher": "empty", "hooks": []},
                    {"hooks": [{"command": "/usr/bin/ormah whisper inject"}]},
                ],
                "PreToolUse": [{"matcher": "Write"}],
            }
        }
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(json.dumps(data, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()

        result = json.loads(settings_path.read_text())
        assert result["hooks"] == {
            "UserPromptSubmit": [{"matcher": "empty", "hooks": []}],
            "PreToolUse": [{"matcher": "Write"}],
        }

    def test_no_settings_file_is_noop(self, tmp_path, capsys):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()  # must not raise

        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_invalid_json_is_noop(self, tmp_path, capsys):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("not json{{{")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()

        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_no_hooks_section_is_noop(self, tmp_path, capsys):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({"theme": "dark"}) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()

        # File unchanged
        result = json.loads((claude_dir / "settings.json").read_text())
        assert result == {"theme": "dark"}

    def test_removes_empty_event_key_after_cleanup(self, tmp_path):
        data = {
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "/bin/ormah whisper store"}]}
                ],
            }
        }
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps(data, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()
            result = json.loads((claude_dir / "settings.json").read_text())

        assert "hooks" not in result


class TestRemoveMcpFromJson:
    def test_removes_ormah_entry(self, tmp_path):
        config = tmp_path / "claude.json"
        config.write_text(json.dumps({
            "mcpServers": {
                "ormah": {"command": "/bin/ormah", "args": ["mcp"]},
                "other": {"command": "/bin/other"},
            }
        }, indent=2) + "\n")

        _remove_mcp_from_json(config)
        result = json.loads(config.read_text())
        assert "ormah" not in result["mcpServers"]
        assert "other" in result["mcpServers"]

    def test_removes_mcpservers_key_when_empty(self, tmp_path):
        config = tmp_path / "claude.json"
        config.write_text(json.dumps({
            "mcpServers": {"ormah": {"command": "/bin/ormah", "args": ["mcp"]}}
        }, indent=2) + "\n")

        _remove_mcp_from_json(config)
        result = json.loads(config.read_text())
        assert "mcpServers" not in result

    def test_noop_when_file_missing(self, tmp_path):
        config = tmp_path / "nonexistent.json"
        _remove_mcp_from_json(config)  # must not raise

    def test_noop_when_ormah_not_present(self, tmp_path):
        config = tmp_path / "claude.json"
        original = {"mcpServers": {"other": {"command": "/bin/other"}}}
        config.write_text(json.dumps(original, indent=2) + "\n")

        _remove_mcp_from_json(config)
        result = json.loads(config.read_text())
        assert result == original


class TestRemoveCodexMcpConfig:
    def test_removes_ormah_block(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        config = codex_dir / "config.toml"
        config.write_text(
            '[projects."/tmp/demo"]\n'
            'trust_level = "trusted"\n\n'
            '[mcp_servers.ormah]\n'
            'command = "/bin/ormah"\n'
            'args = ["mcp"]\n'
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_mcp_config()

        content = config.read_text()
        assert '[mcp_servers.ormah]' not in content
        assert '[projects."/tmp/demo"]' in content

    def test_noop_when_missing(self, tmp_path):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_mcp_config()


class TestRemoveCodexHooks:
    def test_removes_ormah_hooks_only(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        hooks_path = codex_dir / "hooks.json"
        hooks_path.write_text(json.dumps({
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {"type": "command", "command": "/usr/bin/ormah whisper inject"},
                            {"type": "command", "command": "/usr/bin/other-tool run"},
                        ]
                    }
                ],
                "Stop": [
                    {"hooks": [{"type": "command", "command": "/usr/bin/ormah whisper store"}]}
                ],
            }
        }, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_hooks()

        result = json.loads(hooks_path.read_text())
        assert result["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == "/usr/bin/other-tool run"
        assert "Stop" not in result["hooks"]

    def test_preserves_untouched_empty_and_missing_hooks_matchers(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        hooks_path = codex_dir / "hooks.json"
        hooks_path.write_text(json.dumps({
            "hooks": {
                "Stop": [
                    {"matcher": "empty", "hooks": []},
                    {"hooks": [{"command": "/usr/bin/ormah whisper store"}]},
                ],
                "PreToolUse": [{"matcher": "Write"}],
            }
        }, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_hooks()

        result = json.loads(hooks_path.read_text())
        assert result["hooks"] == {
            "Stop": [{"matcher": "empty", "hooks": []}],
            "PreToolUse": [{"matcher": "Write"}],
        }

    def test_noop_when_missing(self, tmp_path):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_hooks()


class TestRemoveCodexAgents:
    def test_removes_agent_file(self, tmp_path):
        agent_dir = tmp_path / ".codex" / "agents"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "ormah-maintenance.toml"
        agent_file.write_text('name = "ormah-maintenance"\n')

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_agents()

        assert not agent_file.exists()

    def test_noop_when_missing(self, tmp_path):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_agents()


class TestRemoveCodexMdBlock:
    def test_removes_sentinel_block_from_agents_md(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        agents_md = codex_dir / "AGENTS.md"
        agents_md.write_text(
            "# Before\n\n"
            f"{CODEX_AGENTS_SENTINEL_START}\ncontent\n{CODEX_AGENTS_SENTINEL_END}\n"
            "\n# After\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_md_block()

        content = agents_md.read_text()
        assert CODEX_AGENTS_SENTINEL_START not in content
        assert CODEX_AGENTS_SENTINEL_END not in content
        assert "content" not in content
        assert "# Before" in content
        assert "# After" in content

    def test_uses_override_file_when_present(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        override_md = codex_dir / "AGENTS.override.md"
        override_md.write_text(
            "# Before\n\n"
            f"{CODEX_AGENTS_SENTINEL_START}\ncontent\n{CODEX_AGENTS_SENTINEL_END}\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_codex_md_block()

        content = override_md.read_text()
        assert CODEX_AGENTS_SENTINEL_START not in content


class TestRemoveClaudeMdBlock:
    def test_removes_sentinel_block(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text(
            "# Before\n\n"
            f"{CLAUDE_MD_SENTINEL_START}\normah instructions\n{CLAUDE_MD_SENTINEL_END}\n"
            "\n# After\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_md_block()

        content = claude_md.read_text()
        assert CLAUDE_MD_SENTINEL_START not in content
        assert CLAUDE_MD_SENTINEL_END not in content
        assert "ormah instructions" not in content
        assert "# Before" in content
        assert "# After" in content

    def test_no_triple_newlines_after_removal(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text(
            "# Before\n\n"
            f"{CLAUDE_MD_SENTINEL_START}\ncontent\n{CLAUDE_MD_SENTINEL_END}\n"
            "\n# After\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_md_block()

        content = claude_md.read_text()
        assert "\n\n\n" not in content

    def test_noop_when_file_missing(self, tmp_path, capsys):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_md_block()

        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_noop_when_no_sentinels(self, tmp_path, capsys):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text("# Just some content\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_md_block()

        assert claude_md.read_text() == "# Just some content\n"
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()


class TestRemovePiMdBlock:
    def test_removes_sentinel_block(self, tmp_path):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)
        agents_md = pi_dir / "AGENTS.md"
        agents_md.write_text(
            "# Before\n\n"
            f"{PI_AGENTS_MD_SENTINEL_START}\normah instructions\n{PI_AGENTS_MD_SENTINEL_END}\n"
            "\n# After\n"
        )

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_pi_md_block()

        content = agents_md.read_text()
        assert PI_AGENTS_MD_SENTINEL_START not in content
        assert PI_AGENTS_MD_SENTINEL_END not in content
        assert "ormah instructions" not in content
        assert "# Before" in content
        assert "# After" in content

    def test_noop_when_file_missing(self, tmp_path, capsys):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_pi_md_block()

        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_noop_when_no_sentinels(self, tmp_path, capsys):
        pi_dir = tmp_path / ".pi" / "agent"
        pi_dir.mkdir(parents=True)
        agents_md = pi_dir / "AGENTS.md"
        agents_md.write_text("# Just some content\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_pi_md_block()

        assert agents_md.read_text() == "# Just some content\n"
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()


class TestRemovePiAgents:
    def test_removes_agent_file(self, tmp_path, capsys):
        agent_dir = tmp_path / ".pi" / "agent" / "agents"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "ormah-maintenance.md"
        agent_file.write_text("# old\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_pi_agents()

        assert not agent_file.exists()
        captured = capsys.readouterr()
        assert "Removed" in captured.out

    def test_noop_when_missing(self, tmp_path, capsys):
        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_pi_agents()

        captured = capsys.readouterr()
        assert "Removed" not in captured.out


class TestRemovePiExtension:
    def test_removes_only_ormah_entries(self, tmp_path):
        settings_path = tmp_path / ".pi" / "agent" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps(
                {
                    "packages": ["npm:ormah-pi", "npm:other-package"],
                    "extensions": [{"source": "/tmp/custom-extension.ts"}],
                    "theme": "dark",
                }
            )
        )

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.setup._find_binary", return_value="/usr/bin/pi"),
            patch(
                "ormah.setup.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="", stderr=""),
            ) as mock_run,
        ):
            _remove_pi_extension()

        assert json.loads(settings_path.read_text()) == {
            "packages": ["npm:other-package"],
            "extensions": [{"source": "/tmp/custom-extension.ts"}],
            "theme": "dark",
        }
        mock_run.assert_called_once_with(
            ["/usr/bin/pi", "remove", "npm:ormah-pi"],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_scrubs_settings_when_pi_binary_is_missing(self, tmp_path):
        settings_path = tmp_path / ".pi" / "agent" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"extensions": ["/checkout/integrations/pi-plugin/ormah-pi.ts"]})
        )

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.setup._find_binary", return_value=None),
        ):
            _remove_pi_extension()

        assert json.loads(settings_path.read_text()) == {}


class TestRunUninstall:
    @pytest.fixture(autouse=True)
    def _isolate_uninstall_from_real_home(self, tmp_path):
        """Keep uninstall tests from touching the developer's real Ormah install."""
        fake_settings = MagicMock()
        fake_settings.memory_dir = tmp_path / ".local" / "share" / "ormah" / "memory"
        fake_settings.embedding_model = "BAAI/bge-base-en-v1.5"
        fake_settings.whisper_reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.config.settings", fake_settings),
            patch("ormah.setup._get_running_server_data_dir", return_value=None),
        ):
            yield

    def _patch_all(self, mock_uninstall_autostart, mock_hooks, mock_mcp, mock_md, mock_rmtree, mock_run):
        """Shared patcher helper — not used directly, see individual tests."""

    def test_cancels_on_first_no(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch("ormah.server_manager.uninstall_autostart") as mock_daemon:
            run_uninstall(yes=False)
            mock_daemon.assert_not_called()

        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()

    def test_cancels_on_wrong_confirmation(self, monkeypatch, capsys):
        inputs = iter(["y", "nope"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with patch("ormah.server_manager.uninstall_autostart") as mock_daemon:
            run_uninstall(yes=False)
            mock_daemon.assert_not_called()

        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()

    def test_proceeds_with_yes_flag(self, tmp_path, capsys):
        with (
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_codex_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_pi_extension"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_codex_md_block"),
            patch("ormah.setup._remove_codex_agents"),
            patch("ormah.setup._remove_claude_agents"),
            patch("ormah.setup._remove_claude_commands"),
            patch("ormah.setup._remove_pi_md_block"),
            patch("ormah.setup._remove_pi_agents"),
            patch("shutil.rmtree"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_uninstall(yes=True)

        captured = capsys.readouterr()
        assert "uninstalled" in captured.out.lower()

    def test_deletes_data_directories(self, tmp_path, capsys):
        share_dir = tmp_path / ".local" / "share" / "ormah"
        cache_dir = tmp_path / ".cache" / "ormah"
        config_dir = tmp_path / ".config" / "ormah"
        for d in (share_dir, cache_dir, config_dir):
            d.mkdir(parents=True)

        fake_settings = MagicMock()
        # memory_dir inside the XDG tree so it doesn't get added as an extra candidate
        fake_settings.memory_dir = share_dir / "memory"
        fake_settings.embedding_model = "BAAI/bge-base-en-v1.5"
        fake_settings.whisper_reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.config.settings", fake_settings),
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_codex_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_pi_extension"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_codex_md_block"),
            patch("ormah.setup._remove_codex_agents"),
            patch("ormah.setup._remove_claude_agents"),
            patch("ormah.setup._remove_claude_commands"),
            patch("ormah.setup._remove_pi_md_block"),
            patch("ormah.setup._remove_pi_agents"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_uninstall(yes=True)

        assert not share_dir.exists()
        assert not cache_dir.exists()
        assert not config_dir.exists()

    @pytest.mark.parametrize("filename", ["cloud.key", "ormah-recovery-kit.md"])
    def test_config_cleanup_preserves_each_cloud_recovery_file(self, tmp_path, filename):
        config_dir = tmp_path / ".config" / "ormah"
        config_dir.mkdir(parents=True)
        recovery_file = config_dir / filename
        recovery_file.write_text("recovery material\n")
        recovery_file.chmod(0o600)
        (config_dir / ".env").write_text("ORMAH_ACCOUNT_TOKEN=secret\n")
        nested = config_dir / "generated"
        nested.mkdir()
        (nested / "state.json").write_text("{}\n")

        preserved = _remove_config_preserving_cloud_recovery(config_dir)

        assert preserved == (recovery_file,)
        assert recovery_file.read_text() == "recovery material\n"
        assert stat.S_IMODE(recovery_file.stat().st_mode) == 0o600
        assert list(config_dir.iterdir()) == [recovery_file]

    def test_uninstall_preserves_cloud_recovery_material_with_yes(self, tmp_path, capsys):
        share_dir = tmp_path / ".local" / "share" / "ormah"
        cache_dir = tmp_path / ".cache" / "ormah"
        config_dir = tmp_path / ".config" / "ormah"
        for directory in (share_dir, cache_dir, config_dir):
            directory.mkdir(parents=True)

        from ormah.cloud.keys import get_or_create_store_id, init_key, write_recovery_kit

        key_path = config_dir / "cloud.key"
        kit_path = config_dir / "ormah-recovery-kit.md"
        memory_dir = share_dir / "memory"
        init_key(key_path)
        store_id = get_or_create_store_id(memory_dir)
        write_recovery_kit(store_id, key_path=key_path, kit_path=kit_path)
        key_content = key_path.read_text()
        kit_content = kit_path.read_text()
        (config_dir / ".env").write_text("ORMAH_ACCOUNT_TOKEN=secret\n")

        with (
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_codex_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_pi_extension"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_codex_md_block"),
            patch("ormah.setup._remove_codex_agents"),
            patch("ormah.setup._remove_claude_agents"),
            patch("ormah.setup._remove_claude_commands"),
            patch("ormah.setup._remove_pi_md_block"),
            patch("ormah.setup._remove_pi_agents"),
            patch("ormah.setup._remove_fastembed_cache"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_uninstall(yes=True)

        assert not share_dir.exists()
        assert not cache_dir.exists()
        assert key_path.read_text() == key_content
        assert kit_path.read_text() == kit_content
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(kit_path.stat().st_mode) == 0o600
        assert {path.name for path in config_dir.iterdir()} == {
            "cloud.key",
            "ormah-recovery-kit.md",
        }
        output = capsys.readouterr().out.lower()
        assert "preserved cloud recovery material" in output
        assert "permanently unreadable" in output

    def test_recovery_preflight_regenerates_missing_kit(self, tmp_path):
        from ormah.cloud.keys import (
            extract_store_id,
            get_or_create_store_id,
            init_key,
            load_identity_strings,
        )

        config_dir = tmp_path / ".config" / "ormah"
        key_path = config_dir / "cloud.key"
        kit_path = config_dir / "ormah-recovery-kit.md"
        memory_dir = tmp_path / "memory"
        init_key(key_path)
        store_id = get_or_create_store_id(memory_dir)

        result = _prepare_cloud_recovery(config_dir, [memory_dir])

        assert result.kit_regenerated is True
        assert result.paths == (key_path, kit_path)
        assert load_identity_strings(kit_path) == load_identity_strings(key_path)
        assert extract_store_id(str(kit_path)) == store_id
        assert stat.S_IMODE(kit_path.stat().st_mode) == 0o600

    def test_recovery_preflight_refreshes_stale_kit_after_rotation(self, tmp_path):
        from ormah.cloud.keys import (
            get_or_create_store_id,
            init_key,
            load_identity_strings,
            rotate_key,
            write_recovery_kit,
        )

        config_dir = tmp_path / ".config" / "ormah"
        key_path = config_dir / "cloud.key"
        kit_path = config_dir / "ormah-recovery-kit.md"
        memory_dir = tmp_path / "memory"
        init_key(key_path)
        store_id = get_or_create_store_id(memory_dir)
        write_recovery_kit(store_id, key_path=key_path, kit_path=kit_path)
        rotate_key(key_path)

        result = _prepare_cloud_recovery(config_dir, [memory_dir])

        assert result.kit_regenerated is True
        assert load_identity_strings(kit_path) == load_identity_strings(key_path)

    def test_recovery_preflight_accepts_complete_kit_without_key_file(self, tmp_path):
        from ormah.cloud.keys import get_or_create_store_id, init_key, write_recovery_kit

        config_dir = tmp_path / ".config" / "ormah"
        key_path = config_dir / "cloud.key"
        kit_path = config_dir / "ormah-recovery-kit.md"
        memory_dir = tmp_path / "memory"
        init_key(key_path)
        store_id = get_or_create_store_id(memory_dir)
        write_recovery_kit(store_id, key_path=key_path, kit_path=kit_path)
        original = kit_path.read_bytes()
        key_path.unlink()

        result = _prepare_cloud_recovery(config_dir, [memory_dir])

        assert result.paths == (kit_path,)
        assert result.kit_regenerated is False
        assert kit_path.read_bytes() == original

    def test_recovery_preflight_refuses_key_without_store_id(self, tmp_path):
        from ormah.cloud.keys import init_key

        config_dir = tmp_path / ".config" / "ormah"
        init_key(config_dir / "cloud.key")

        with pytest.raises(CloudRecoveryPreflightError, match="no store ID"):
            _prepare_cloud_recovery(config_dir, [tmp_path / "memory"])

    def test_recovery_preflight_refuses_mismatched_store(self, tmp_path):
        from ormah.cloud.keys import get_or_create_store_id, init_key, write_recovery_kit

        config_dir = tmp_path / ".config" / "ormah"
        key_path = config_dir / "cloud.key"
        kit_path = config_dir / "ormah-recovery-kit.md"
        memory_a = tmp_path / "memory-a"
        memory_b = tmp_path / "memory-b"
        init_key(key_path)
        store_a = get_or_create_store_id(memory_a)
        store_b = get_or_create_store_id(memory_b)
        write_recovery_kit(store_b, key_path=key_path, kit_path=kit_path)
        original = kit_path.read_bytes()

        with pytest.raises(CloudRecoveryPreflightError, match="does not match"):
            _prepare_cloud_recovery(config_dir, [memory_a])

        assert store_a != store_b
        assert kit_path.read_bytes() == original

    def test_recovery_preflight_refuses_multiple_store_ids(self, tmp_path):
        from ormah.cloud.keys import get_or_create_store_id, init_key

        config_dir = tmp_path / ".config" / "ormah"
        init_key(config_dir / "cloud.key")
        memory_a = tmp_path / "memory-a"
        memory_b = tmp_path / "memory-b"
        get_or_create_store_id(memory_a)
        get_or_create_store_id(memory_b)

        with pytest.raises(CloudRecoveryPreflightError, match="Multiple cloud store IDs"):
            _prepare_cloud_recovery(config_dir, [memory_a, memory_b])

    def test_uninstall_aborts_before_changes_when_recovery_is_incomplete(
        self, tmp_path, capsys
    ):
        from ormah.cloud.keys import init_key

        config_dir = tmp_path / ".config" / "ormah"
        key_path = config_dir / "cloud.key"
        init_key(key_path)

        with patch("ormah.server_manager.uninstall_autostart") as mock_daemon:
            run_uninstall(yes=True)

        mock_daemon.assert_not_called()
        assert key_path.is_file()
        output = capsys.readouterr().out
        assert "Uninstall cancelled before removing any data or integrations" in output

    def test_warns_about_cloud_key_before_interactive_confirmation(
        self, tmp_path, monkeypatch, capsys
    ):
        config_dir = tmp_path / ".config" / "ormah"
        config_dir.mkdir(parents=True)
        key_path = config_dir / "cloud.key"
        key_path.write_text("AGE-SECRET-KEY-TEST\n")
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch("ormah.server_manager.uninstall_autostart") as mock_daemon:
            run_uninstall(yes=False)

        mock_daemon.assert_not_called()
        assert key_path.exists()
        output = capsys.readouterr().out.lower()
        assert "uninstall will not delete it" in output
        assert "permanently unreadable" in output

    def test_graceful_uv_failure(self, capsys):
        with (
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_codex_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_pi_extension"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_codex_md_block"),
            patch("ormah.setup._remove_codex_agents"),
            patch("ormah.setup._remove_claude_agents"),
            patch("ormah.setup._remove_claude_commands"),
            patch("ormah.setup._remove_pi_md_block"),
            patch("ormah.setup._remove_pi_agents"),
            patch("shutil.rmtree"),
            patch("ormah.setup._remove_uv_tool_install_files", return_value=False),
            patch("subprocess.run", side_effect=Exception("uv not found")),
        ):
            run_uninstall(yes=True)  # must not raise

        captured = capsys.readouterr()
        assert "uv tool uninstall ormah" in captured.out

    def test_uv_failure_removes_desktop_tool_install_files(self, tmp_path):
        shim = tmp_path / ".local" / "bin" / "ormah"
        shim.parent.mkdir(parents=True)
        shim.write_text("#!/bin/sh\n")
        shim.chmod(0o755)

        tool_dir = tmp_path / ".local" / "share" / "uv" / "tools" / "ormah"
        (tool_dir / "bin").mkdir(parents=True)
        (tool_dir / "bin" / "ormah").write_text("#!/bin/sh\n")

        fake_settings = MagicMock()
        fake_settings.memory_dir = tmp_path / ".local" / "share" / "ormah" / "memory"

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.config.settings", fake_settings),
            patch("ormah.setup._get_running_server_data_dir", return_value=None),
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_codex_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_codex_md_block"),
            patch("ormah.setup._remove_codex_agents"),
            patch("ormah.setup._remove_claude_agents"),
            patch("ormah.setup._remove_claude_commands"),
            patch("ormah.setup._remove_fastembed_cache"),
            patch("subprocess.run", side_effect=FileNotFoundError("uv")),
        ):
            run_uninstall(yes=True)

        assert not shim.exists()
        assert not tool_dir.exists()

    def test_successful_uv_uninstall_still_removes_stale_command_shim(self, tmp_path):
        shim = tmp_path / ".local" / "bin" / "ormah"
        shim.parent.mkdir(parents=True)
        shim.write_text("#!/bin/sh\n")
        shim.chmod(0o755)

        fake_settings = MagicMock()
        fake_settings.memory_dir = tmp_path / ".local" / "share" / "ormah" / "memory"

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.config.settings", fake_settings),
            patch("ormah.setup._get_running_server_data_dir", return_value=None),
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_codex_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_codex_md_block"),
            patch("ormah.setup._remove_codex_agents"),
            patch("ormah.setup._remove_claude_agents"),
            patch("ormah.setup._remove_claude_commands"),
            patch("ormah.setup._remove_fastembed_cache"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_uninstall(yes=True)

        assert not shim.exists()

    def test_eof_on_first_prompt_cancels(self, monkeypatch, capsys):
        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)

        with patch("ormah.server_manager.uninstall_autostart") as mock_daemon:
            run_uninstall(yes=False)
            mock_daemon.assert_not_called()


class TestRemoveFastembedCache:
    def test_deletes_known_model_dirs(self, tmp_path, monkeypatch, capsys):
        # Simulate a fastembed cache with two model directories
        model_a = tmp_path / "models--qdrant--bge-base-en-v1.5-onnx-q"
        model_b = tmp_path / "models--Xenova--ms-marco-MiniLM-L-6-v2"
        model_a.mkdir()
        model_b.mkdir()

        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))

        fake_embed_models = [{"model": "BAAI/bge-base-en-v1.5", "sources": {"hf": "qdrant/bge-base-en-v1.5-onnx-q"}}]
        fake_rerank_models = [{"model": "Xenova/ms-marco-MiniLM-L-6-v2", "sources": {"hf": "Xenova/ms-marco-MiniLM-L-6-v2"}}]

        with (
            patch("fastembed.TextEmbedding.list_supported_models", return_value=fake_embed_models),
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder.list_supported_models", return_value=fake_rerank_models),
        ):
            _remove_fastembed_cache()

        assert not model_a.exists()
        assert not model_b.exists()

    def test_noop_when_cache_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "nonexistent"))
        _remove_fastembed_cache()  # must not raise
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_warns_when_registry_unavailable(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
        with (
            patch("fastembed.TextEmbedding.list_supported_models", side_effect=Exception("no fastembed")),
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder.list_supported_models", side_effect=Exception("no fastembed")),
        ):
            _remove_fastembed_cache()  # must not raise
        captured = capsys.readouterr()
        assert "manually" in captured.out.lower()

    def test_removes_cache_dir_when_empty_after_cleanup(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "models--qdrant--bge-base-en-v1.5-onnx-q"
        model_dir.mkdir()

        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))

        fake_embed_models = [{"model": "BAAI/bge-base-en-v1.5", "sources": {"hf": "qdrant/bge-base-en-v1.5-onnx-q"}}]

        with (
            patch("fastembed.TextEmbedding.list_supported_models", return_value=fake_embed_models),
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder.list_supported_models", return_value=[]),
        ):
            _remove_fastembed_cache()

        # cache_dir itself is removed when empty
        assert not tmp_path.exists()

    def test_uses_default_fastembed_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FASTEMBED_CACHE_PATH")
        cache_dir = tmp_path / ".local" / "share" / "ormah" / "models"
        model_dir = cache_dir / "models--qdrant--bge-base-en-v1.5-onnx-q"
        model_dir.mkdir(parents=True)

        fake_embed_models = [{"model": "BAAI/bge-base-en-v1.5", "sources": {"hf": "qdrant/bge-base-en-v1.5-onnx-q"}}]

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("fastembed.TextEmbedding.list_supported_models", return_value=fake_embed_models),
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder.list_supported_models", return_value=[]),
        ):
            _remove_fastembed_cache()

        assert not cache_dir.exists()


class TestPreloadLocalModels:
    def test_preloads_embedding_and_reranker_into_shared_cache(self, tmp_path):
        fake_settings = MagicMock()
        fake_settings.embedding_provider = "local"
        fake_settings.embedding_model = "BAAI/bge-base-en-v1.5"
        fake_settings.whisper_reranker_enabled = True
        fake_settings.whisper_reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        with (
            patch("ormah.setup.settings", fake_settings),
            patch("ormah.setup.get_fastembed_cache_dir", return_value=tmp_path),
            patch("fastembed.TextEmbedding") as embed_cls,
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as reranker_cls,
        ):
            _preload_local_models()

        embed_cls.assert_called_once_with("BAAI/bge-base-en-v1.5", cache_dir=str(tmp_path))
        reranker_cls.assert_called_once_with("Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=str(tmp_path))

    def test_skips_reranker_preload_when_disabled(self, tmp_path):
        fake_settings = MagicMock()
        fake_settings.embedding_provider = "local"
        fake_settings.embedding_model = "BAAI/bge-base-en-v1.5"
        fake_settings.whisper_reranker_enabled = False
        fake_settings.whisper_reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        with (
            patch("ormah.setup.settings", fake_settings),
            patch("ormah.setup.get_fastembed_cache_dir", return_value=tmp_path),
            patch("fastembed.TextEmbedding") as embed_cls,
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as reranker_cls,
        ):
            _preload_local_models()

        embed_cls.assert_called_once()
        reranker_cls.assert_not_called()

    def test_skips_fastembed_embedding_preload_for_ollama(self, tmp_path):
        fake_settings = MagicMock()
        fake_settings.embedding_provider = "ollama"
        fake_settings.embedding_model = "bge-m3:latest"
        fake_settings.whisper_reranker_enabled = True
        fake_settings.whisper_reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        with (
            patch("ormah.setup.settings", fake_settings),
            patch("ormah.setup.get_fastembed_cache_dir", return_value=tmp_path),
            patch("fastembed.TextEmbedding") as embed_cls,
            patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as reranker_cls,
        ):
            _preload_local_models()

        embed_cls.assert_not_called()
        reranker_cls.assert_called_once_with(
            "Xenova/ms-marco-MiniLM-L-6-v2",
            cache_dir=str(tmp_path),
        )


class TestUninstallMemoryDirResolution:
    """Verify that run_uninstall deletes the actual memory directory regardless of
    whether it is a relative path (old default) or absolute (new default / custom)."""

    def _run_uninstall_with_mem_dir(self, tmp_path, mem_dir: Path):
        """Helper: run uninstall with a faked settings.memory_dir."""
        fake_settings = MagicMock()
        fake_settings.memory_dir = mem_dir
        fake_settings.embedding_model = "BAAI/bge-base-en-v1.5"
        fake_settings.whisper_reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        with (
            patch("ormah.setup.Path.home", return_value=tmp_path),
            patch("ormah.config.settings", fake_settings),
            patch("ormah.server_manager.uninstall_autostart"),
            patch("ormah.setup._remove_claude_hooks"),
            patch("ormah.setup._remove_mcp_registration"),
            patch("ormah.setup._remove_pi_extension"),
            patch("ormah.setup._remove_claude_md_block"),
            patch("ormah.setup._remove_fastembed_cache"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_uninstall(yes=True)

    def test_relative_memory_dir_resolved_from_home(self, tmp_path):
        """Old ormah used Path('memory') — server runs from ~, so data is at ~/memory."""
        fake_mem = tmp_path / "memory"
        fake_mem.mkdir()
        (fake_mem / "index.db").touch()

        self._run_uninstall_with_mem_dir(tmp_path, Path("memory"))

        assert not fake_mem.exists()

    def test_absolute_memory_dir_outside_xdg_is_deleted(self, tmp_path):
        """Custom absolute path outside XDG dirs is also cleaned up."""
        custom_mem = tmp_path / "custom_memories"
        custom_mem.mkdir()

        self._run_uninstall_with_mem_dir(tmp_path, custom_mem)

        assert not custom_mem.exists()

    def test_memory_dir_inside_xdg_not_double_deleted(self, tmp_path):
        """memory_dir under ~/.local/share/ormah is already covered by XDG cleanup."""
        xdg_share = tmp_path / ".local" / "share" / "ormah"
        xdg_share.mkdir(parents=True)
        mem_dir = xdg_share / "memory"
        mem_dir.mkdir()

        # Should not raise even though the parent dir covers the mem_dir
        self._run_uninstall_with_mem_dir(tmp_path, mem_dir)

        assert not xdg_share.exists()


class TestUninstallCli:
    def test_uninstall_calls_run_uninstall(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "uninstall", "--yes"]),
            patch("ormah.setup.run_uninstall") as mock_uninstall,
        ):
            main()
            mock_uninstall.assert_called_once_with(yes=True)

    def test_uninstall_no_yes_flag(self):
        from ormah.cli import main

        with (
            patch("sys.argv", ["ormah", "uninstall"]),
            patch("ormah.setup.run_uninstall") as mock_uninstall,
        ):
            main()
            mock_uninstall.assert_called_once_with(yes=False)


class TestStopRunningServer:
    def test_no_process_found_returns_false(self, capsys):
        from ormah.server_manager import stop_running_server

        with (
            patch("platform.system", return_value="Linux"),
            patch("shutil.which", return_value=None),  # no systemctl
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            result = stop_running_server()

        assert result is False
        out = capsys.readouterr().out
        assert "No running Ormah server found." in out

    def test_ps_finds_pid_sends_sigterm(self, capsys):
        import signal as _signal

        from ormah.server_manager import stop_running_server

        fake_pid = 99999
        ps_output = f"    {fake_pid} 1000 ormah server start\n"

        def fake_run(cmd, **kwargs):
            m = MagicMock()
            if cmd[0] == "ps":
                m.stdout = ps_output
                m.returncode = 0
            else:
                m.stdout = ""
                m.returncode = 1
            return m

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError

        with (
            patch("platform.system", return_value="Linux"),
            patch("shutil.which", return_value=None),
            patch("subprocess.run", side_effect=fake_run),
            patch("os.getpid", return_value=0),
            patch("os.getuid", return_value=1000),
            patch("os.kill", side_effect=fake_kill) as mock_kill,
        ):
            result = stop_running_server()

        assert mock_kill.call_args_list[0] == ((fake_pid, _signal.SIGTERM),)
        assert result is True
        out = capsys.readouterr().out
        assert "Stopped Ormah server" in out

    def test_ps_ignores_other_users_processes(self, capsys):
        from ormah.server_manager import stop_running_server

        ps_output = (
            "    111 2000 /home/r2205/.local/share/uv/tools/ormah/bin/python3 "
            "/home/r2205/.local/bin/ormah server start\n"
        )

        with (
            patch("platform.system", return_value="Linux"),
            patch("shutil.which", return_value=None),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=ps_output)),
            patch("os.getpid", return_value=0),
            patch("os.getuid", return_value=1000),
            patch("os.kill") as mock_kill,
        ):
            result = stop_running_server()

        assert result is False
        mock_kill.assert_not_called()
        out = capsys.readouterr().out
        assert "No running Ormah server found." in out

    def test_false_positive_not_matched(self):
        from ormah.server_manager import _is_ormah_server_start_command

        # These should NOT match
        assert not _is_ormah_server_start_command("grep ormah server start")
        assert not _is_ormah_server_start_command("bash -c 'ormah server start'")
        assert not _is_ormah_server_start_command("python3 worker.py ormah server start")
        assert not _is_ormah_server_start_command("ormah server stop")
        assert not _is_ormah_server_start_command("ormah server status")
        assert not _is_ormah_server_start_command("")

        # These SHOULD match
        assert _is_ormah_server_start_command("ormah server start")
        assert _is_ormah_server_start_command("/usr/local/bin/ormah server start")
        assert _is_ormah_server_start_command("ormah server start --reload")
        assert _is_ormah_server_start_command(
            "/home/r2205/.local/share/uv/tools/ormah/bin/python3 "
            "/home/r2205/.local/bin/ormah server start"
        )

    def test_systemd_active_calls_stop(self, capsys):
        from ormah.server_manager import stop_running_server

        def fake_run(cmd, **kwargs):
            m = MagicMock()
            if "is-active" in cmd:
                m.stdout = "active\n"
            elif cmd[0] == "pgrep":
                m.stdout = ""
            else:
                m.stdout = ""
            m.returncode = 0
            return m

        with (
            patch("platform.system", return_value="Linux"),
            patch("shutil.which", return_value="/usr/bin/systemctl"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = stop_running_server()

        assert result is True
        out = capsys.readouterr().out
        assert "Stopped Ormah server (systemd)." in out

    def test_server_stop_cli_calls_both(self):
        from ormah.cli import main
        from ormah.server_manager import _StopServerResult

        with (
            patch("sys.argv", ["ormah", "server", "stop"]),
            patch("ormah.server_manager._stop_running_server", return_value=_StopServerResult()) as mock_stop,
            patch("ormah.server_manager.uninstall_autostart") as mock_uninstall,
        ):
            main()

        mock_stop.assert_called_once()
        mock_uninstall.assert_called_once()

    def test_server_stop_exits_nonzero_on_failure(self):
        from ormah.cli import main
        from ormah.server_manager import _StopServerResult

        failed_result = _StopServerResult(found=True, stopped=False, failed=True)

        with (
            patch("sys.argv", ["ormah", "server", "stop"]),
            patch("ormah.server_manager._stop_running_server", return_value=failed_result),
            patch("ormah.server_manager.uninstall_autostart"),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1


class TestMergeHooks:
    ORMAH = {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "/x/ormah whisper inject"}]}]
    }

    def test_preserves_cotenant_under_same_event(self):
        existing = {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "other-tool"}]}]}
        merged = _merge_hooks(existing, self.ORMAH)
        cmds = [h["command"] for m in merged["UserPromptSubmit"] for h in m["hooks"]]
        assert "other-tool" in cmds
        assert "/x/ormah whisper inject" in cmds

    def test_idempotent_no_duplicate_ormah(self):
        once = _merge_hooks({}, self.ORMAH)
        twice = _merge_hooks(once, self.ORMAH)
        cmds = [h["command"] for m in twice["UserPromptSubmit"] for h in m["hooks"]]
        assert cmds.count("/x/ormah whisper inject") == 1

    def test_leaves_unclaimed_events_untouched(self):
        existing = {"PreToolUse": [{"hooks": [{"type": "command", "command": "rtk hook claude"}]}]}
        merged = _merge_hooks(existing, self.ORMAH)
        assert merged["PreToolUse"] == existing["PreToolUse"]

    def test_substring_collision_not_stripped(self):
        existing = {"UserPromptSubmit": [{"hooks": [
            {"type": "command", "command": "/opt/whisper inject-backup run"}]}]}
        merged = _merge_hooks(existing, self.ORMAH)
        cmds = [h["command"] for m in merged["UserPromptSubmit"] for h in m["hooks"]]
        assert "/opt/whisper inject-backup run" in cmds

    def test_preserves_matcher_without_hooks_key(self):
        # matcher dict with no "hooks" key must survive the merge unchanged
        existing = {"UserPromptSubmit": [{"matcher": "Write"}]}
        merged = _merge_hooks(existing, self.ORMAH)
        # user's hooks-less matcher is still present
        assert {"matcher": "Write"} in merged["UserPromptSubmit"]
        # ormah's matcher is also appended
        cmds = [
            h["command"]
            for m in merged["UserPromptSubmit"]
            if isinstance(m, dict)
            for h in m.get("hooks", [])
        ]
        assert "/x/ormah whisper inject" in cmds

    def test_preserves_matcher_with_only_nonormah_hooks(self):
        # regression guard: a matcher whose hooks are all non-ormah survives unchanged
        existing = {"UserPromptSubmit": [{"hooks": [{"command": "/bin/other"}]}]}
        merged = _merge_hooks(existing, self.ORMAH)
        cmds = [h["command"] for m in merged["UserPromptSubmit"] for h in m.get("hooks", [])]
        assert "/bin/other" in cmds
        assert "/x/ormah whisper inject" in cmds

    def test_preserves_malformed_non_dict_hook_entry(self):
        # a matcher whose hooks list contains a malformed (non-dict) entry must not crash
        existing = {"UserPromptSubmit": [{"hooks": ["malformed-string-entry"]}]}
        merged = _merge_hooks(existing, self.ORMAH)  # must NOT raise
        # the malformed entry is preserved
        all_hooks = [h for m in merged["UserPromptSubmit"] if isinstance(m, dict) for h in m.get("hooks", [])]
        assert "malformed-string-entry" in all_hooks
        # ormah's hook is also appended
        cmds = [h["command"] for m in merged["UserPromptSubmit"] if isinstance(m, dict) for h in m.get("hooks", []) if isinstance(h, dict)]
        assert "/x/ormah whisper inject" in cmds

    def test_non_string_command_preserved(self):
        # a hook with a non-string command is neither Ormah nor crash-worthy —
        # it must be preserved and the merge must succeed
        existing = {"UserPromptSubmit": [{"hooks": [{"command": 123}]}]}
        merged = _merge_hooks(existing, self.ORMAH)  # must not raise
        preserved = [
            h
            for m in merged["UserPromptSubmit"]
            if isinstance(m, dict)
            for h in m.get("hooks", [])
        ]
        assert {"command": 123} in preserved

    def test_drops_matcher_emptied_of_only_ormah_hooks(self):
        # a matcher that held ONLY ormah hooks should be dropped after stripping,
        # not left as {"hooks": []} — ormah's own fresh matcher is then appended
        existing = {
            "UserPromptSubmit": [{"hooks": [{"command": "/x/ormah whisper inject"}]}]
        }
        merged = _merge_hooks(existing, self.ORMAH)
        # exactly one occurrence of the inject command (from ormah's appended matcher)
        cmds = [h["command"] for m in merged["UserPromptSubmit"] for h in m.get("hooks", [])]
        assert cmds.count("/x/ormah whisper inject") == 1
        # no matcher left with an empty hooks list
        empty_hook_matchers = [
            m for m in merged["UserPromptSubmit"]
            if isinstance(m, dict) and m.get("hooks") == []
        ]
        assert empty_hook_matchers == []


class TestStripOrmahHooks:
    def test_preserves_malformed_inner_hooks_verbatim(self):
        existing = {
            "UserPromptSubmit": [
                {"hooks": 5},
                {"hooks": "not-a-list"},
                {"matcher": "missing"},
            ]
        }

        cleaned, changed = _strip_ormah_hooks(existing)

        assert changed is False
        assert cleaned == existing

    def test_removes_ormah_hook_without_rewriting_untouched_matchers(self):
        untouched = {"matcher": "empty", "hooks": []}
        existing = {
            "UserPromptSubmit": [
                untouched,
                {
                    "hooks": [
                        {"command": "/x/ormah whisper inject"},
                        {"command": "/bin/other"},
                    ]
                },
            ]
        }

        cleaned, changed = _strip_ormah_hooks(existing)

        assert changed is True
        assert cleaned["UserPromptSubmit"] == [
            untouched,
            {"hooks": [{"command": "/bin/other"}]},
        ]


class TestIsOrmahHook:
    def test_non_string_command_returns_false(self):
        assert _is_ormah_hook({"command": 123}) is False
        assert _is_ormah_hook({"command": ["a", "b"]}) is False
        assert _is_ormah_hook({"command": {"x": 1}}) is False

    def test_matches_real_ormah_hook(self):
        assert _is_ormah_hook({"command": "/usr/bin/ormah whisper inject"})
        assert _is_ormah_hook({"command": "/abs/path/ormah whisper store"})

    def test_matches_plugin_wrapper_form(self):
        assert _is_ormah_hook({"command": "/x/plugin/bin/ormah-whisper-inject"})
        assert _is_ormah_hook({"command": "/x/plugin/bin/ormah-whisper-store"})

    def test_rejects_substring_collision(self):
        assert not _is_ormah_hook({"command": "/opt/whisper inject-backup run"})
        assert not _is_ormah_hook({"command": "tools/whisper store-archive"})

    def test_rejects_malformed_command(self):
        assert not _is_ormah_hook({"command": ""})
        assert not _is_ormah_hook({})
        assert not _is_ormah_hook({"command": "unterminated 'quote"})

    def test_non_dict_entry_returns_false(self):
        assert _is_ormah_hook("a string") is False
        assert _is_ormah_hook(123) is False
        assert _is_ormah_hook(None) is False
        assert _is_ormah_hook(["list"]) is False


class TestRemoveClaudeHooksPluginWrapper:
    def test_removes_plugin_wrapper_hook(self, tmp_path):
        data = {"hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "/x/plugin/bin/ormah-whisper-inject"}]}
        ]}}
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps(data, indent=2) + "\n")

        with patch("ormah.setup.Path.home", return_value=tmp_path):
            _remove_claude_hooks()

        result = json.loads((claude_dir / "settings.json").read_text())
        cmds = [
            h["command"]
            for m in result.get("hooks", {}).get("UserPromptSubmit", [])
            for h in m["hooks"]
        ]
        assert "/x/plugin/bin/ormah-whisper-inject" not in cmds


class TestConfigureClaudeHooksMerge:
    def test_preserves_existing_userpromptsubmit_hook(self, tmp_path):
        from ormah.setup import configure_claude_hooks
        import json

        sp = tmp_path / "settings.json"
        sp.write_text(
            json.dumps(
                {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "other-tool"}]}]}}
            )
        )
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
        data = json.loads(sp.read_text())
        cmds = [h["command"] for m in data["hooks"]["UserPromptSubmit"] for h in m["hooks"]]
        assert "other-tool" in cmds
        assert "/abs/ormah whisper inject" in cmds

    def test_rerun_does_not_duplicate(self, tmp_path):
        from ormah.setup import configure_claude_hooks
        import json

        sp = tmp_path / "settings.json"
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
            configure_claude_hooks("/abs/ormah")
        data = json.loads(sp.read_text())
        cmds = [h["command"] for m in data["hooks"]["UserPromptSubmit"] for h in m["hooks"]]
        assert cmds.count("/abs/ormah whisper inject") == 1

    def test_preserves_existing_precompact_and_sessionend(self, tmp_path):
        from ormah.setup import configure_claude_hooks
        import json

        sp = tmp_path / "settings.json"
        sp.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreCompact": [
                            {"hooks": [{"type": "command", "command": "other-precompact"}]}
                        ],
                        "SessionEnd": [
                            {"hooks": [{"type": "command", "command": "other-sessionend"}]}
                        ],
                    }
                }
            )
        )
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
        data = json.loads(sp.read_text())
        pre = [h["command"] for m in data["hooks"]["PreCompact"] for h in m["hooks"]]
        end = [h["command"] for m in data["hooks"]["SessionEnd"] for h in m["hooks"]]
        assert "other-precompact" in pre and "/abs/ormah whisper store" in pre
        assert "other-sessionend" in end and "/abs/ormah whisper store" in end

    def test_corrupt_json_left_unchanged_and_no_false_success(self, tmp_path, capsys):
        from ormah.setup import configure_claude_hooks

        sp = tmp_path / "settings.json"
        sp.write_text('{ "theme": "dark", BROKEN')
        before = sp.read_text()
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
        assert sp.read_text() == before
        assert "Whisper hooks installed" not in capsys.readouterr().out

    def test_non_object_json_left_unchanged(self, tmp_path):
        from ormah.setup import configure_claude_hooks

        sp = tmp_path / "settings.json"
        sp.write_text('["not", "an", "object"]')
        before = sp.read_text()
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
        assert sp.read_text() == before

    def test_non_list_event_left_unchanged(self, tmp_path, capsys):
        """Non-list value on a claimed event (nested schema drift) must leave file unchanged."""
        from ormah.setup import configure_claude_hooks

        sp = tmp_path / "settings.json"
        sp.write_text(json.dumps({"theme": "dark", "hooks": {"UserPromptSubmit": {"oops": 1}}}) + "\n")
        before = sp.read_text()
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
        assert sp.read_text() == before
        assert "Whisper hooks installed" not in capsys.readouterr().out

    def test_uniterable_matcher_hooks_fail_closed(self, tmp_path, capsys):
        """A non-iterable 'hooks' value inside a matcher triggers the backstop:
        file is left unchanged, no success message printed."""
        sp = tmp_path / "settings.json"
        sp.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": 5}]}}) + "\n")
        before = sp.read_text()
        with patch("ormah.setup.os.path.expanduser", return_value=str(sp)):
            configure_claude_hooks("/abs/ormah")
        assert sp.read_text() == before
        assert "Whisper hooks installed" not in capsys.readouterr().out


class TestConfigureCodexHooksMerge:
    def test_preserves_existing_stop_hook(self, tmp_path):
        import json

        from ormah.setup import configure_codex_hooks

        codex = tmp_path / ".codex"
        codex.mkdir()
        hp = codex / "hooks.json"
        hp.write_text(
            json.dumps(
                {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other-stop"}]}]}}
            )
        )
        with patch("ormah.setup.Path.home", return_value=tmp_path), patch(
            "ormah.setup._enable_codex_feature"
        ):
            configure_codex_hooks("/abs/ormah")
        data = json.loads(hp.read_text())
        cmds = [h["command"] for m in data["hooks"]["Stop"] for h in m["hooks"]]
        assert "other-stop" in cmds
        assert "/abs/ormah whisper store" in cmds

    def test_rerun_does_not_duplicate(self, tmp_path):
        import json

        from ormah.setup import configure_codex_hooks

        with patch("ormah.setup.Path.home", return_value=tmp_path), patch(
            "ormah.setup._enable_codex_feature"
        ):
            configure_codex_hooks("/abs/ormah")
            configure_codex_hooks("/abs/ormah")
        data = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
        cmds = [h["command"] for m in data["hooks"]["UserPromptSubmit"] for h in m["hooks"]]
        assert cmds.count("/abs/ormah whisper inject") == 1

    def test_corrupt_hooks_json_no_false_success(self, tmp_path, capsys):
        from ormah.setup import configure_codex_hooks

        codex = tmp_path / ".codex"
        codex.mkdir()
        hp = codex / "hooks.json"
        hp.write_text("{ BROKEN")
        before = hp.read_text()
        with patch("ormah.setup.Path.home", return_value=tmp_path), patch(
            "ormah.setup._enable_codex_feature"
        ) as enable:
            configure_codex_hooks("/abs/ormah")
        assert hp.read_text() == before  # unchanged
        enable.assert_not_called()  # feature flag NOT enabled on abort
        assert "Codex hooks installed" not in capsys.readouterr().out
