"""Native 1.2.9 fixtures plus optional real CLI → mock model/daemon smoke test."""
import json
import os
from pathlib import Path

import pytest

from ormah.integrations import json_config as jc
from ormah.integrations.antigravity_hook import handle
from ormah.integrations.hosts import antigravity as host


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home/.config"))
    monkeypatch.delenv("ORMAH_WORKSPACE", raising=False)
    monkeypatch.delenv("ORMAH_SPACE", raising=False)


def fixture(tmp_path):
    # Captured from native agy 1.2.9 on 2026-09-23 with a mock Gemini endpoint.
    # Docs say transcript.jsonl; this release actually supplies transcript_full.jsonl.
    session = "6e2d05fd-086e-4f33-a7d2-6bd44994d1e6"
    artifact = tmp_path / "brain" / session
    log = artifact / ".system_generated/logs/transcript_full.jsonl"
    log.parent.mkdir(parents=True)
    row = {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
           "created_at": "2026-09-23T17:03:06Z", "content":
           "<USER_REQUEST>\nOrmah scratch prompt\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\ntime\n</ADDITIONAL_METADATA>"}
    log.write_text(json.dumps(row) + "\n")
    return {"conversationId": session, "artifactDirectoryPath": str(artifact),
            "transcriptPath": str(log), "workspacePaths": [str(tmp_path)],
            "invocationNum": 0, "initialNumSteps": 1, "modelName": "gemini-3.6-flash-medium"}, log


@pytest.mark.parametrize("project_scope", [True, False])
def test_config_preserves_hooks_and_permissions(tmp_path, project_scope):
    project = tmp_path if project_scope else None
    root = project / ".agents" if project else Path.home() / ".gemini/config"
    root.mkdir(parents=True)
    config = root / "hooks.json"
    config.write_text('{/* keep */ "custom":{"enabled":false,"Stop":[]}}')
    host.connect(project)
    before = config.read_bytes()
    host.connect(project)
    assert config.read_bytes() == before
    hook = jc.get(config.read_text(), ["ormah", "PreInvocation"])[1][0]
    assert hook["timeout"] == 12 and "antigravity_hook" in hook["command"]
    mcp = jc.get((root / "mcp_config.json").read_text(), ["mcpServers", "ormah"])[1]
    assert "--host" in mcp["args"] and "disabledTools" not in mcp
    assert host.status(project)["surfaces"]["desktop"] == "unverified"
    host.disconnect(project)
    assert "/* keep */" in config.read_text() and '"enabled":false' in config.read_text()


async def test_native_transcript_to_ephemeral_context(tmp_path, httpx_mock):
    payload, log = fixture(tmp_path)
    with log.open("a") as f:
        f.write(json.dumps({"type": "USER_INPUT", "source": "HOOK", "content": "old memory"}) + "\n")
    httpx_mock.add_response(json={"text": "retrieved memory"})
    assert await handle(payload, str(tmp_path)) == {"injectSteps": [{"ephemeralMessage": "retrieved memory"}]}
    body = json.loads(httpx_mock.get_request().content)
    assert body["prompt"] == "Ormah scratch prompt" and body["space"] == tmp_path.name
    assert body["session_id"].startswith("antigravity:")


@pytest.mark.parametrize("failure", ["partial", "malformed", "missing", "session", "workspace", "shape"])
async def test_unrecognized_or_ambiguous_data_fails_open(tmp_path, failure):
    payload, log = fixture(tmp_path)
    if failure == "partial":
        log.write_text(log.read_text() + '{"type":')
    elif failure == "malformed":
        log.write_text(log.read_text() + 'invalid\n')
    elif failure == "missing":
        log.unlink()
    elif failure == "session":
        payload["conversationId"] = "other"
    elif failure == "workspace":
        payload["workspacePaths"].append("/other")
    else:
        payload.pop("invocationNum")
    assert await handle(payload, str(tmp_path)) == {}


async def test_global_scope_matches_unbound_mcp(tmp_path, httpx_mock):
    payload, _ = fixture(tmp_path)
    httpx_mock.add_response(json={"text": "global"})
    await handle(payload)
    assert json.loads(httpx_mock.get_request().content)["space"] is None


async def test_offline_returns_empty(tmp_path, httpx_mock):
    import httpx
    payload, _ = fixture(tmp_path)
    httpx_mock.add_exception(httpx.ConnectError("offline"))
    assert await handle(payload, str(tmp_path)) == {}


@pytest.mark.skipif(not os.environ.get("ORMAH_TEST_AGY_BINARY"), reason="opt-in native agy binary required")
def test_native_cli_model_request_contains_whisper(tmp_path, monkeypatch):
    import http.server
    import subprocess
    import threading

    native = Path(os.environ["ORMAH_TEST_AGY_BINARY"]).resolve()
    requests, whispers = [], []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            if self.path == "/agent/whisper":
                whispers.append((body, self.headers.get("Authorization")))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"text":"ORM_NATIVE_CONTEXT_42"}')
            else:
                requests.append(body)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                response = {"candidates": [{"content": {"role": "model", "parts": [{"text": "complete"}]},
                                             "finishReason": "STOP"}]}
                self.wfile.write(("data: " + json.dumps(response) + "\n\n").encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("ORMAH_URL", endpoint)
    monkeypatch.setenv("ORMAH_AUTH_TOKEN", "scratch-token")
    # No real account, personal settings, daemon, memories or permission bypass.
    settings = Path.home() / ".gemini/antigravity-cli/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"modelProvider": "gemini"}))
    host.connect(tmp_path)
    source = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "GEMINI_API_KEY": "scratch-fake-key", "GOOGLE_GEMINI_BASE_URL": endpoint,
           "PYTHONPATH": str(source), "DBUS_SESSION_BUS_ADDRESS": ""}
    try:
        result = subprocess.run([str(native), "--add-dir", str(tmp_path), "--model", "gemini-3.6-flash-medium",
                                 "--print-timeout", "15s", "-p", "Ormah scratch prompt - no tools needed"],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=25)
        assert result.returncode == 0, result.stderr
        assert whispers and whispers[0][0]["space"] == tmp_path.name
        assert whispers[0][1] == "Bearer scratch-token"
        assert any("ORM_NATIVE_CONTEXT_42" in json.dumps(r.get("contents")) for r in requests)
        # agy lazy-loads MCP: a generic call_mcp_tool plus on-disk tool schemas.
        catalog = Path.home() / ".gemini/antigravity-cli/mcp/ormah"
        assert (catalog / "recall.json").exists() and (catalog / "remember.json").exists()
        assert any("call_mcp_tool" in json.dumps(r.get("tools")) and "ormah" in json.dumps(r)
                   for r in requests)
        session = next((Path.home() / ".gemini/antigravity-cli/brain").iterdir()).name
        followup = subprocess.run([str(native), "--conversation", session, "--add-dir", str(tmp_path),
                                   "--model", "gemini-3.6-flash-medium", "--print-timeout", "15s",
                                   "-p", "Another Ormah scratch prompt"], cwd=tmp_path, env=env,
                                  capture_output=True, text=True, timeout=25)
        assert followup.returncode == 0, followup.stderr
        assert whispers[-1][0]["prompt"] == "Another Ormah scratch prompt"
        assert whispers[-1][0]["session_id"] == whispers[0][0]["session_id"]
    finally:
        server.shutdown()
        server.server_close()
