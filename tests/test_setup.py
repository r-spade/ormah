"""Tests for ormah setup and server manager."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ormah.server_manager import (
    LAUNCHD_LABEL,
    PLIST_TEMPLATE,
    SYSTEMD_TEMPLATE,
    get_ormah_bin_path,
    is_server_running,
)
from ormah.setup import (
    CLAUDE_MD_SENTINEL_END,
    CLAUDE_MD_SENTINEL_START,
    ENV_PATH,
    WRAPPER_PATH,
    _merge_json_file,
    _read_env_file,
    _write_env_file,
    configure_claude_hooks,
    configure_claude_code_mcp,
    configure_claude_desktop,
    configure_identity,
    configure_llm,
    generate_server_wrapper,
    install_claude_md,
    seed_identity,
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
            wrapper_path="/home/user/.config/ormah/start-server.sh",
            bin_dir="/usr/local/bin",
            log_dir="/tmp/logs",
        )
        assert "<string>com.ormah.server</string>" in rendered
        assert "<string>/home/user/.config/ormah/start-server.sh</string>" in rendered
        assert "<string>/tmp/logs/ormah.out.log</string>" in rendered
        assert "<key>RunAtLoad</key><true/>" in rendered
        assert "<key>KeepAlive</key><true/>" in rendered

    def test_template_includes_path(self):
        rendered = PLIST_TEMPLATE.format(
            label=LAUNCHD_LABEL,
            wrapper_path="/home/user/.config/ormah/start-server.sh",
            bin_dir="/home/user/.local/bin",
            log_dir="/tmp/logs",
        )
        assert "<key>PATH</key><string>/home/user/.local/bin:" in rendered


class TestSystemdTemplate:
    def test_template_renders(self):
        rendered = SYSTEMD_TEMPLATE.format(
            wrapper_path="/home/user/.config/ormah/start-server.sh",
            bin_dir="/usr/local/bin",
            log_dir="/tmp/logs",
        )
        assert "ExecStart=/home/user/.config/ormah/start-server.sh" in rendered
        assert "Restart=on-failure" in rendered
        assert "WantedBy=default.target" in rendered
        assert "After=network.target" in rendered
        assert 'Environment="PATH=/usr/local/bin:' in rendered
        assert "EnvironmentFile" not in rendered
        assert "StandardOutput=append:/tmp/logs/ormah.out.log" in rendered
        assert "StandardError=append:/tmp/logs/ormah.err.log" in rendered

    def test_template_renders_with_spaces_in_path(self):
        rendered = SYSTEMD_TEMPLATE.format(
            wrapper_path="/home/user/.config/ormah/start-server.sh",
            bin_dir="/home/user/my apps",
            log_dir="/tmp/logs",
        )
        assert "ExecStart=/home/user/.config/ormah/start-server.sh" in rendered


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


class TestConfigureClaudeCodeMcp:
    def test_writes_mcp_config_to_claude_json(self, tmp_path):
        config_path = str(tmp_path / ".claude.json")

        with patch("ormah.setup.shutil.which", return_value=None), \
             patch("ormah.setup.os.path.expanduser", return_value=config_path):
            configure_claude_code_mcp("/abs/path/ormah")

        with open(config_path) as f:
            data = json.load(f)

        assert data["mcpServers"]["ormah"]["command"] == "/abs/path/ormah"
        assert data["mcpServers"]["ormah"]["args"] == ["mcp"]

    def test_merges_with_existing_mcp_servers(self, tmp_path):
        config_path = str(tmp_path / ".claude.json")
        with open(config_path, "w") as f:
            json.dump({"mcpServers": {"fetch": {"command": "uvx"}}}, f)

        with patch("ormah.setup.shutil.which", return_value=None), \
             patch("ormah.setup.os.path.expanduser", return_value=config_path):
            configure_claude_code_mcp("/abs/path/ormah")

        with open(config_path) as f:
            data = json.load(f)

        assert "fetch" in data["mcpServers"]
        assert "ormah" in data["mcpServers"]


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

        with patch("ormah.setup.os.path.expanduser", return_value=str(config_dir)):
            configure_claude_desktop("/abs/path/ormah")

        config_path = config_dir / "claude_desktop_config.json"
        with open(config_path) as f:
            data = json.load(f)

        assert data["mcpServers"]["ormah"]["command"] == "/abs/path/ormah"


# --- CLI tests ---


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
            mock_setup.assert_called_once()

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


# --- Server wrapper tests ---


class TestGenerateServerWrapper:
    def test_creates_wrapper_file(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            result = generate_server_wrapper("/usr/local/bin/ormah")
        assert result == wrapper
        assert wrapper.exists()

    def test_wrapper_has_700_permissions(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        file_mode = stat.S_IMODE(wrapper.stat().st_mode)
        assert file_mode == 0o700

    def test_wrapper_contains_ormah_bin(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert "exec /usr/local/bin/ormah server start" in content

    def test_wrapper_contains_api_key_grep(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert "ANTHROPIC_API_KEY" in content
        assert "OPENAI_API_KEY" in content
        assert "GEMINI_API_KEY" in content

    def test_wrapper_no_hardcoded_secrets(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert "sk-ant-" not in content
        assert "sk-" not in content.replace("#!/", "")  # ignore shebang

    def test_idempotent(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
            first_content = wrapper.read_text()
            generate_server_wrapper("/usr/local/bin/ormah")
            second_content = wrapper.read_text()
        assert first_content == second_content

    def test_sources_env_file(self, tmp_path):
        wrapper = tmp_path / "start-server.sh"
        with patch("ormah.setup.WRAPPER_PATH", wrapper), patch("ormah.setup.ENV_DIR", tmp_path):
            generate_server_wrapper("/usr/local/bin/ormah")
        content = wrapper.read_text()
        assert '.config/ormah/.env' in content
        assert "set -a" in content


# --- Identity tests ---


class TestConfigureIdentity:
    def test_returns_name(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "Rishi")
        name = configure_identity()
        assert name == "Rishi"

    def test_empty_returns_none(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "")
        name = configure_identity()
        assert name is None
        captured = capsys.readouterr()
        assert "learn your name naturally" in captured.out
        assert "ormah will learn your name" in captured.out

    def test_eof_returns_none(self, monkeypatch, capsys):
        def raise_eof(_):
            raise EOFError
        monkeypatch.setattr("builtins.input", raise_eof)
        name = configure_identity()
        assert name is None

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "  Rishi  ")
        name = configure_identity()
        assert name == "Rishi"


class TestSeedIdentity:
    def test_posts_to_server(self, capsys):
        mock_resp = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("ormah.setup.httpx.Client", return_value=mock_client):
            seed_identity("Rishi")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "remember" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["content"] == "User's name is Rishi"
        assert body["type"] == "person"
        assert body["tier"] == "core"
        assert body["about_self"] is True

        captured = capsys.readouterr()
        assert "Ormah now knows your name" in captured.out

    def test_handles_server_error_gracefully(self, capsys):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")

        with patch("ormah.setup.httpx.Client", return_value=mock_client):
            seed_identity("Rishi")

        captured = capsys.readouterr()
        assert "Could not seed identity" in captured.out
        assert "learn your name naturally" in captured.out
        assert "ormah will learn your name" in captured.out


# --- LLM configuration tests ---


class TestConfigureLlm:
    def _clear_all_api_keys(self, monkeypatch):
        """Remove all known API keys from env so auto-detect doesn't fire."""
        for key in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
            "AZURE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_anthropic_no_key_in_env(self, tmp_path, monkeypatch):
        """Manual selection of Anthropic without key in env stores only provider/model."""
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        inputs = iter(["1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "litellm"
        assert result["ORMAH_LLM_MODEL"] == "claude-haiku-4-5-20251001"
        assert "ANTHROPIC_API_KEY" not in result

    def test_openai_no_key_in_env(self, tmp_path, monkeypatch):
        """Manual selection of OpenAI without key in env stores only provider/model."""
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        inputs = iter(["2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "litellm"
        assert result["ORMAH_LLM_MODEL"] == "gpt-4.1-mini"
        assert "OPENAI_API_KEY" not in result

    def test_ollama_no_key_needed(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        self._clear_all_api_keys(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _: "4")

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
        monkeypatch.setattr("builtins.input", lambda _: "5")

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
        assert "store, recall, and whisper all work" in captured.out

    def test_auto_detect_does_not_store_key(self, tmp_path, monkeypatch):
        """Auto-detect path stores provider/model but NOT the API key."""
        env_path = tmp_path / ".env"
        monkeypatch.setattr("builtins.input", lambda _: "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_LLM_PROVIDER"] == "litellm"
        assert result["ORMAH_LLM_MODEL"] == "claude-haiku-4-5-20251001"
        assert "ANTHROPIC_API_KEY" not in result

    def test_preserves_existing_env_values(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text("ORMAH_PORT=9999\n")
        self._clear_all_api_keys(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _: "5")

        with (
            patch("ormah.setup.ENV_PATH", env_path),
            patch("ormah.setup.ENV_DIR", tmp_path),
        ):
            configure_llm()
            result = _read_env_file()

        assert result["ORMAH_PORT"] == "9999"
        assert result["ORMAH_LLM_PROVIDER"] == "none"


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
        assert "Added ormah instructions to ~/.claude/CLAUDE.md" in captured.out

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
