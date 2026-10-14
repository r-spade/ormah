import json
import os
import subprocess
from pathlib import Path

import pytest

from ormah.integrations import json_config as jc
from ormah.integrations.hosts import openclaw as host


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'home/.config'))
    monkeypatch.setenv('OPENCLAW_STATE_DIR', str(tmp_path / 'claw'))
    monkeypatch.delenv('OPENCLAW_CONFIG_PATH', raising=False)


def test_json5_setup_scope_permissions_and_cleanup(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    host.directory().mkdir()
    config = host.config_path()
    config.write_text("{ // settings\n agents: {defaults: {model: 'mine'}}, "
                      "plugins: {allow: ['other']}, mcp: {servers: {other: {command: 'other'}}}, }")
    monkeypatch.setenv('ORMAH_AUTH_TOKEN', 'test-secret-not-written')
    host.connect(workspace)
    original = config.read_bytes()
    host.connect(workspace)
    assert config.read_bytes() == original
    assert b'test-secret-not-written' not in original
    data = jc.parse(config.read_text(), json5=True).value
    assert data['mcp']['servers']['ormah']['args'][-1] == str(workspace)
    assert data['mcp']['servers']['ormah']['env']['ORMAH_AUTH_TOKEN'] == '${ORMAH_AUTH_TOKEN}'
    assert data['plugins']['entries']['ormah']['hooks'] == {'allowConversationAccess': True}
    assert data['plugins']['allow'] == ['other', 'ormah']
    assert host.status()['whisper'] == 'native_extension'
    host.disconnect()
    text = config.read_text()
    data = jc.parse(text, json5=True).value
    assert "model: 'mine'" in text and '// settings' in text
    assert data['plugins']['allow'] == ['other']
    assert data['mcp']['servers'] == {'other': {'command': 'other'}}


def test_explicit_plugin_denial_is_preserved(tmp_path):
    host.directory().mkdir()
    host.config_path().write_text('{plugins:{deny:["ormah"]}}')
    with pytest.raises(ValueError, match='policy'):
        host.connect(tmp_path)
    assert host.config_path().read_text() == '{plugins:{deny:["ormah"]}}'


def test_native_extension_injects_only_matching_workspace(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = b'{"text":"fixture memory"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('ORMAH_URL', f'http://127.0.0.1:{server.server_port}')
    host.connect(tmp_path)
    script = tmp_path / 'test.mjs'
    script.write_text('''import assert from 'node:assert/strict';
const {default: plugin} = await import(process.argv[2]);
let handler;
plugin.register({on(name, fn) { assert.equal(name,'before_prompt_build'); handler=fn; }});
const ctx = {workspaceDir:process.argv[3], sessionId:'session-a', agentId:'main', hookInvocation:{assertActive(){}}};
const result = await handler({prompt:'old history', currentUserMessage:'current prompt',messages:[]},ctx);
assert.deepEqual(result,{prependContext:'fixture memory'});
assert.equal(await handler({prompt:'history',currentUserMessage:'',messages:[]},ctx),undefined);
assert.equal(await handler({prompt:'x',messages:[]},{...ctx,workspaceDir:'/'}),undefined);
let count=0;
const expired = {...ctx,hookInvocation:{assertActive(){if (++count>1) throw Error('expired');}}};
assert.equal(await handler({prompt:'stale',messages:[]},expired),undefined);
''')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')}
    try:
        subprocess.run(['node', str(script), (host.directory()/'ormah-plugin/index.mjs').as_uri(),
                        str(tmp_path)], env=env, check=True, capture_output=True, timeout=20)
    finally:
        server.shutdown()
        server.server_close()
    assert requests[0]['prompt'] == 'current prompt'
    assert requests[0]['session_id'].startswith('openclaw:')
    assert len(requests) == 2
