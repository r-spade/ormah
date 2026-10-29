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


def migrate_config(config, native):
    data = jc.parse(config.read_text()).value
    if native:
        data['plugins'] = data.pop('plugin')
        server = data['mcp'].pop('ormah')
        server['disabled'] = not server.pop('enabled')
        data['mcp']['servers'] = {'ormah': server}
    else:
        data['plugin'] = data.pop('plugins')
        server = data['mcp']['servers'].pop('ormah')
        del data['mcp']['servers']
        server['enabled'] = not server.pop('disabled')
        data['mcp']['ormah'] = server
    config.write_text(json.dumps(data))


@pytest.mark.parametrize('project_scope', [True, False])
@pytest.mark.parametrize('native', [True, False])
@pytest.mark.parametrize('reconnect_first', [True, False])
def test_migrated_config_status_reconnect_disconnect(tmp_path, project_scope, native, reconnect_first):
    project = tmp_path / 'project' if project_scope else None
    root = project or opencode.directory()
    root.mkdir(parents=True)
    config = root / 'opencode.jsonc'
    if not native:
        config.write_text('{"plugins":[],"mcp":{"servers":{}}}')
    opencode.connect(project)
    migrate_config(config, native)
    before = config.read_bytes()
    assert opencode.status(project)['tools'] == 'mcp'
    assert opencode.status(project)['whisper'] == 'native_extension'
    assert config.read_bytes() == before  # status never rewrites host configuration
    if reconnect_first:
        opencode.connect(project)
        assert config.read_bytes() == before
    opencode.disconnect(project)
    data = jc.parse(config.read_text()).value
    assert data['plugins' if native else 'plugin'] == []
    assert 'ormah' not in (data['mcp']['servers'] if native else data['mcp'])
    assets = (project / '.opencode' if project else root) / 'ormah'
    assert not (assets / 'package.json').exists()
    opencode.connect(project)
    assert opencode.status(project)['whisper'] == 'native_extension'


def test_migrated_edited_mcp_preserved_with_warning(tmp_path, capsys):
    project = tmp_path / 'project'
    opencode.connect(project)
    config = project / 'opencode.jsonc'
    migrate_config(config, True)
    data = jc.parse(config.read_text()).value
    data['mcp']['servers']['ormah']['command'] = ['personal-command']
    config.write_text(json.dumps(data))
    assert opencode.status(project)['tools'] == 'unconfigured'
    with pytest.raises(ValueError):
        opencode.connect(project)
    opencode.disconnect(project)
    assert 'Preserved user-edited' in capsys.readouterr().err
    data = jc.parse(config.read_text()).value
    assert data['mcp']['servers']['ormah']['command'] == ['personal-command']
    assert data['plugins'] == []


@pytest.mark.parametrize('migrate_plugins', [True, False])
def test_partial_migration_preserves_personal_entries(tmp_path, migrate_plugins):
    project = tmp_path / 'project'
    project.mkdir()
    config = project / 'opencode.jsonc'
    config.write_text('{"plugin":["personal"],"mcp":{"personal":{"command":["keep"]}}}')
    opencode.connect(project)
    data = jc.parse(config.read_text()).value
    if migrate_plugins:
        owned = data['plugin'].pop()
        data['plugins'] = [owned]
    else:
        server = data['mcp'].pop('ormah')
        server['disabled'] = not server.pop('enabled')
        data['mcp']['servers'] = {'ormah': server}
    config.write_text(json.dumps(data))
    assert opencode.status(project)['whisper'] == 'native_extension'
    opencode.connect(project)
    opencode.disconnect(project)
    data = jc.parse(config.read_text()).value
    assert data['plugin'] == ['personal']
    assert data['mcp']['personal'] == {'command': ['keep']}
    assert data.get('plugins', []) == []
    assert 'ormah' not in data['mcp']
    assert 'ormah' not in data['mcp'].get('servers', {})


def test_migrated_mcp_boolean_edit_is_not_adopted(tmp_path, capsys):
    project = tmp_path / 'project'
    opencode.connect(project)
    config = project / 'opencode.jsonc'
    migrate_config(config, True)
    data = jc.parse(config.read_text()).value
    data['mcp']['servers']['ormah']['disabled'] = 0
    config.write_text(json.dumps(data))
    assert opencode.status(project)['tools'] == 'unconfigured'
    opencode.disconnect(project)
    assert 'Preserved user-edited' in capsys.readouterr().err
    assert jc.parse(config.read_text()).value['mcp']['servers']['ormah']['disabled'] == 0


@pytest.mark.parametrize('project_scope', [True, False])
def test_interrupted_connect_then_edit_can_disconnect(tmp_path, monkeypatch, project_scope):
    from ormah.integrations import common, ownership
    project = tmp_path / 'project' if project_scope else None
    root = project or opencode.directory()
    root.mkdir(parents=True)
    config = root / 'opencode.jsonc'
    config.write_text('{"model":"original"}')
    atomic_write = ownership.atomic_write

    def fail_config(path, text):
        if path == config:
            raise PermissionError('simulated interrupted setup')
        atomic_write(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(ownership, 'atomic_write', fail_config)
        with pytest.raises(PermissionError):
            opencode.connect(project)
    receipt = common.receipt('opencode', project)
    assert json.loads(receipt.read_text())['pending']
    edited = '{"model":"personal-edit"}'
    config.write_text(edited)
    with pytest.raises(ValueError, match='disconnect first'):
        opencode.connect(project)
    opencode.disconnect(project)
    assert config.read_text() == edited
    assert not receipt.exists()
    opencode.connect(project)
    assert opencode.status(project)['whisper'] == 'native_extension'


@pytest.mark.parametrize('project_scope', [True, False])
@pytest.mark.parametrize('asset', ['plugin.mjs', 'v2.mjs', 'package.json', 'instructions.md'])
def test_partial_install_status_reports_missing_assets(tmp_path, monkeypatch, project_scope, asset):
    from ormah.integrations import common, ownership
    project = tmp_path / 'project' if project_scope else None
    root = project or opencode.directory()
    root.mkdir(parents=True)
    config = root / 'opencode.jsonc'
    atomic_write = ownership.atomic_write

    def fail_asset(path, text):
        if path.name == asset:
            raise PermissionError('simulated partial installation')
        atomic_write(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(ownership, 'atomic_write', fail_asset)
        with pytest.raises(PermissionError):
            opencode.connect(project)
    receipt = common.receipt('opencode', project)
    before_config, before_receipt = config.read_bytes(), receipt.read_bytes()
    assert opencode.status(project)['tools'] == 'mcp'
    assert opencode.status(project)['whisper'] == 'unconfigured'
    assert config.read_bytes() == before_config
    assert receipt.read_bytes() == before_receipt
    opencode.connect(project)
    assert opencode.status(project)['whisper'] == 'native_extension'
    opencode.disconnect(project)
    assert not receipt.exists()
