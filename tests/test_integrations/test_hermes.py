"""Native pre_llm_call fixture and round-trip Hermes YAML configuration."""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from ormah.integrations import yaml_config as yc
from ormah.integrations.hosts import hermes as host


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-profile"))
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)


def test_setup_preserves_comments_models_plugins_and_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("ORMAH_AUTH_TOKEN", "scratch-not-to-store")
    path = host.home() / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text('# model comment\nmodel: "chosen"\nplugins:\n  enabled: [mine] # keep\n'
                    'memory:\n  provider: custom\nmcp_servers:\n  other:\n    command: other\n')
    host.connect(tmp_path)
    first = path.read_bytes()
    host.connect(tmp_path)
    assert path.read_bytes() == first
    assert "scratch-not-to-store" not in path.read_text()
    assert yc.get(path.read_text(), ["plugins", "enabled"])[1] == ["mine", "ormah"]
    assert yc.get(path.read_text(), ["mcp_servers", "ormah"])[1]["env"]["ORMAH_AUTH_TOKEN"] == "${ORMAH_AUTH_TOKEN}"
    assert host.status()["whisper"] == "native_extension"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
    assert host.status()["whisper"] == "unconfigured"
    monkeypatch.setenv("HERMES_HOME", str(path.parent))
    host.disconnect()
    assert '# model comment' in path.read_text() and '# keep' in path.read_text()
    assert '"chosen"' in path.read_text()
    assert yc.get(path.read_text(), ["memory", "provider"])[1] == "custom"
    assert yc.get(path.read_text(), ["plugins", "enabled"])[1] == ["mine"]


@pytest.mark.parametrize("text", ["not: [valid", "model: a\nmodel: b\n",
                                "plugins:\n  disabled: [ormah]\n",
                                "plugins: &prefs\n  enabled: []\ncopy: *prefs\n"])
def test_reject_corrupt_denied_or_alias_edits(tmp_path, text):
    path = host.home() / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    with pytest.raises(ValueError):
        host.connect(tmp_path)
    assert path.read_text() == text
    assert not (host.home() / "plugins/ormah").exists()


def test_preserve_user_modified_plugin(tmp_path):
    host.connect(tmp_path)
    plugin = host.home() / "plugins/ormah/__init__.py"
    plugin.write_text(plugin.read_text() + "\n# custom\n")
    host.disconnect()
    assert plugin.read_text().endswith("# custom\n")


def load_plugin():
    path = host.home() / "plugins/ormah/__init__.py"
    spec = importlib.util.spec_from_file_location("ormah_hermes_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_context_with_bridge_and_scoped_environment(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append((json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                             self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"text":"fixture memory"}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ORMAH_URL", "http://wrong-profile.invalid")
    monkeypatch.setenv("ORMAH_AUTH_TOKEN", "wrong-profile-token")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "src"))
    active = {"ORMAH_URL": f"http://127.0.0.1:{server.server_port}", "ORMAH_AUTH_TOKEN": "active-profile-token"}
    secret_scope = types.ModuleType("agent.secret_scope")
    secret_scope.get_secret = active.get
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.secret_scope", secret_scope)
    host.connect(tmp_path)
    plugin = load_plugin()
    registered = {}
    plugin.register(types.SimpleNamespace(register_hook=lambda event, cb: registered.update({event: cb})))
    try:
        # Exact documented/source call shape; multimodal bytes and history are not sent.
        result = registered["pre_llm_call"](
            session_id="session-42", user_message=[{"type": "text", "text": "actual question"},
                                                   {"type": "image_url", "image_url": "ignored"}],
            conversation_history=[{"role": "user", "content": "old"}],
            is_first_turn=True, model="unchanged", platform="cli", task_id="task", turn_id="turn")
        assert result == {"context": "fixture memory"}
        assert requests[0][0]["prompt"] == "actual question"
        assert requests[0][0]["space"] == tmp_path.name
        assert requests[0][0]["session_id"].startswith("hermes:")
        assert requests[0][1] == "Bearer active-profile-token"
    finally:
        server.shutdown()
        server.server_close()
    assert plugin.recall(session_id=None, user_message="x") is None
    assert plugin.recall(session_id="s", user_message=[]) is None


def test_subprocess_timeout_fails_open(tmp_path, monkeypatch):
    import subprocess
    host.connect(tmp_path)
    plugin = load_plugin()
    scope = types.ModuleType("agent.secret_scope")
    scope.get_secret = lambda key: None
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.secret_scope", scope)
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 12
        raise subprocess.TimeoutExpired("test", 12)
    monkeypatch.setattr(plugin.subprocess, "run", timeout)
    assert plugin.recall(session_id="s", user_message="x") is None


@pytest.mark.skipif(not os.environ.get("ORMAH_TEST_HERMES_SOURCE"), reason="opt-in pinned native source required")
def test_native_hermes_prompt_consumer(tmp_path):
    """Execute the host's real wire-content functions without importing its full runtime."""
    import ast
    source = Path(os.environ["ORMAH_TEST_HERMES_SOURCE"]) / "agent/turn_context.py"
    tree = ast.parse(source.read_text())
    names = {"compose_multimodal_context_part", "compose_user_api_content", "substitute_api_content"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    # Future annotations avoid pulling optional host dependencies into Ormah's env.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *functions], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(source), "exec")
    scope = {}
    exec(code, scope)
    content = scope["compose_user_api_content"]("clean user text", "", "fixture memory")
    message = {"role": "user", "content": "clean user text", "api_content": content}
    scope["substitute_api_content"](message)
    assert message == {"role": "user", "content": "clean user text\n\nfixture memory"}
    assert scope["compose_multimodal_context_part"]("", "fixture memory") == "fixture memory"
