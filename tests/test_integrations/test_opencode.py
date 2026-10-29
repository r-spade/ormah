import json
import os
import subprocess
from pathlib import Path

import pytest

from ormah.integrations.hosts import opencode
from ormah.integrations import json_config as jc


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'home/.config'))


@pytest.mark.parametrize('project_scope', [True, False])
def test_setup_preserves_config_and_scopes_mcp(tmp_path, project_scope):
    project = tmp_path / 'project' if project_scope else None
    root = project or opencode.directory()
    root.mkdir(parents=True)
    config = root / 'opencode.jsonc'
    config.write_text('{// model\n"model":"chosen","permission":{"*":"ask"},'
                      '"plugin":["my-plugin"],"instructions":["AGENTS.md"],}')
    opencode.connect(project)
    first = config.read_bytes()
    opencode.connect(project)
    assert config.read_bytes() == first
    data = jc.parse(config.read_text()).value
    assert data['mcp']['ormah']['command'][-1] == (str(project) if project else '.')
    assert data['permission'] == {'*': 'ask'}
    assert opencode.status(project)['whisper'] == 'native_extension'
    opencode.disconnect(project)
    data = jc.parse(config.read_text()).value
    assert data['plugin'] == ['my-plugin']
    assert data['instructions'] == ['AGENTS.md']
    assert data['model'] == 'chosen' and '// model' in config.read_text()


def test_native_plugin_contract_with_real_bridge_and_mock_daemon(tmp_path):
    """Execute JS → Python → HTTP → injected TextPart, without an LLM or live store."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'text': 'fixture memory'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    project = tmp_path / 'project'
    project.mkdir()
    opencode.connect(project)
    plugin = project / '.opencode/ormah/plugin.mjs'
    script = tmp_path / 'contract.mjs'
    script.write_text('''import assert from 'node:assert/strict';
const { default: Ormah } = await import(process.argv[2]);
const hooks = await Ormah({directory: process.argv[3]});
const output = {message:{id:'msg_fixture'}, parts:[{type:'text',text:'actual prompt'}]};
await hooks['chat.message']({sessionID:'session-a'}, output);
assert.equal(output.parts.length, 2);
assert.equal(output.parts[1].synthetic, true);
assert.equal(output.parts[1].ignored, undefined);
assert.equal(output.parts[1].messageID, 'msg_fixture');
assert.equal(output.parts[1].sessionID, 'session-a');
assert.match(output.parts[1].text, /fixture memory/);
// The pinned host's toModelMessages retains non-ignored text parts including synthetic text.
const modelText = output.parts.filter(p => p.type === 'text' && !p.ignored && p.text !== '').map(p=>p.text);
assert.equal(modelText.length, 2);
await hooks['chat.message']({sessionID:'session-a'}, output);
assert.equal(output.parts.length, 2);
const synthetic = {message:{id:'msg_continuation'},parts:[{type:'text',text:'continue',synthetic:true}]};
await hooks['chat.message']({sessionID:'session-a'},synthetic);
assert.equal(synthetic.parts.length,1);
''')
    env = {**os.environ, 'ORMAH_URL': f'http://127.0.0.1:{server.server_port}',
           'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')}
    try:
        subprocess.run(['node', str(script), plugin.as_uri(), str(project)],
                       env=env, check=True, capture_output=True, text=True, timeout=20)
    finally:
        server.shutdown()
        server.server_close()
    assert len(requests) == 1
    assert requests[0]['prompt'] == 'actual prompt'
    assert requests[0]['space'] == project.name
    assert requests[0]['session_id'].startswith('opencode:')


def test_directory_package_routes_both_host_generations(tmp_path):
    project = tmp_path / 'project'
    opencode.connect(project)
    data = jc.parse((project / 'opencode.jsonc').read_text()).value
    assets = project / '.opencode/ormah'
    assert data['plugin'] == [assets.as_uri()]
    manifest = json.loads((assets / 'package.json').read_text())
    assert manifest['main'] == './plugin.mjs'
    assert manifest['exports']['.'] == './v2.mjs'
    assert (assets / 'v2.mjs').is_file()


def test_native_v2_config_keeps_other_plugins_and_servers(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    config = project / 'opencode.jsonc'
    config.write_text('{// native V2\n"plugins":["personal"],"mcp":{"servers":{"personal":{"type":"local","command":["personal"]}}}}')
    opencode.connect(project)
    first = config.read_bytes()
    opencode.connect(project)
    assert config.read_bytes() == first
    data = jc.parse(config.read_text()).value
    assert 'plugin' not in data
    assert data['plugins'][0] == 'personal'
    assert set(data['mcp']) == {'servers'}
    assert data['mcp']['servers']['ormah']['disabled'] is False
    assert data['mcp']['servers']['ormah']['command'][-1] == str(project)
    opencode.disconnect(project)
    data = jc.parse(config.read_text()).value
    assert data['plugins'] == ['personal']
    assert set(data['mcp']['servers']) == {'personal'}
    assert '// native V2' in config.read_text()


def test_native_v2_existing_ormah_is_not_shadowed(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    config = project / 'opencode.jsonc'
    before = '{"mcp":{"servers":{"ormah":{"type":"local","command":["personal"]}}}}'
    config.write_text(before)
    with pytest.raises(ValueError, match='Preserving existing configuration'):
        opencode.connect(project)
    assert config.read_text() == before
    assert not (project / '.opencode/ormah').exists()


def test_legacy_file_plugin_requires_disconnect_before_upgrade(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    config = project / 'opencode.jsonc'
    before = json.dumps({'plugin': [(project / '.opencode/ormah/plugin.mjs').as_uri()]})
    config.write_text(before)
    with pytest.raises(ValueError, match='disconnect'):
        opencode.connect(project)
    assert config.read_text() == before


def test_native_config_does_not_shadow_existing_legacy_ormah(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    config = project / 'opencode.jsonc'
    before = '{"mcp":{"ormah":{"type":"local","command":["personal"]},"servers":{}}}'
    config.write_text(before)
    with pytest.raises(ValueError, match='Preserving existing configuration'):
        opencode.connect(project)
    assert config.read_text() == before
