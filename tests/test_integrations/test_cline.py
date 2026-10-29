"""Fixtures follow Cline 4.1.20 HookFactory and SDK bridge source, not CLI hooks."""
import json
import os
import subprocess

import pytest

from ormah.integrations import json_config as jc
from ormah.integrations.cline_hook import handle
from ormah.integrations.hosts import cline as host


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home/.config"))
    monkeypatch.setenv("CLINE_MCP_SETTINGS_PATH", str(tmp_path / "profile/cline_mcp_settings.json"))


def payload(project, **changes):
    return dict({"clineVersion": "4.1.20", "hookName": "UserPromptSubmit",
                 "timestamp": "2026-09-23T00:00:00Z", "taskId": "task-42",
                 "workspaceRoots": [str(project)],
                 "userPromptSubmit": {"prompt": "How does this work?", "attachments": []}}, **changes)


def test_setup_cleanup_idempotent_and_preserves(tmp_path, monkeypatch):
    monkeypatch.setenv("ORMAH_AUTH_TOKEN", "test-secret-do-not-save")
    project = tmp_path / "project"
    project.mkdir()
    path = host.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('{/* keep */ "mcpServers":{"other":{"command":"other"}}}')
    host.connect(project)
    before = path.read_bytes()
    host.connect(project)
    assert path.read_bytes() == before
    config = jc.get(path.read_text(), ["mcpServers", "ormah"])[1]
    assert config["args"][-1] == str(project)
    assert config["autoApprove"] == []
    assert config["env"]["ORMAH_AUTH_TOKEN"] == "${env:ORMAH_AUTH_TOKEN}"
    assert "test-secret" not in path.read_text()
    assert host.status()["whisper"] == "native_hook"
    script = project / ".clinerules/hooks/UserPromptSubmit"
    assert os.access(script, os.X_OK)
    # Execute the installed script; invalid input must succeed and never cancel.
    run = subprocess.run([str(script)], input="invalid", text=True, capture_output=True, timeout=10)
    assert run.returncode == 0
    assert json.loads(run.stdout) == {"cancel": False, "contextModification": "", "errorMessage": ""}
    script.chmod(0o600)
    assert host.status()["whisper"] == "unconfigured"
    host.disconnect()
    assert not script.exists()
    assert "/* keep */" in path.read_text() and '"other"' in path.read_text()


def test_hook_conflict_and_requires_workspace(tmp_path):
    with pytest.raises(ValueError, match="--project"):
        host.connect()
    script = tmp_path / ".clinerules/hooks/UserPromptSubmit"
    script.parent.mkdir(parents=True)
    script.write_text("user script")
    with pytest.raises(ValueError, match="Preserving existing"):
        host.connect(tmp_path)
    assert script.read_text() == "user script" and not host.config_path().exists()


async def test_nested_payload_context_injection(tmp_path, httpx_mock):
    httpx_mock.add_response(json={"text": "retrieved memory"})
    result = await handle(payload(tmp_path), str(tmp_path))
    assert result == {"cancel": False, "contextModification": "retrieved memory", "errorMessage": ""}
    body = json.loads(httpx_mock.get_request().content)
    assert body["session_id"].startswith("cline:") and body["space"] == tmp_path.name
    assert body["prompt"] == "How does this work?"


@pytest.mark.parametrize("changes", [
    {"hookName": "prompt_submit"}, {"workspaceRoots": []},
    {"workspaceRoots": ["/other"]}, {"workspaceRoots": ["/one", "/two"]},
    {"taskId": None}, {"userPromptSubmit": None},
])
async def test_wrong_scope_and_cli_contract_fail_open(tmp_path, changes):
    assert (await handle(payload(tmp_path, **changes), str(tmp_path)))["contextModification"] == ""


async def test_offline_never_cancels(tmp_path, httpx_mock):
    import httpx
    httpx_mock.add_exception(httpx.ConnectError("offline"))
    output = await handle(payload(tmp_path), str(tmp_path))
    assert output["cancel"] is False and output["contextModification"] == ""
