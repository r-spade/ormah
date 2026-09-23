"""Cursor config fixtures follow cursor.com/docs/mcp (2026-09-23)."""
import json

import pytest

from ormah.integrations.hosts import cursor
from ormah.integrations import json_config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'home' / '.config'))


@pytest.mark.parametrize('project_scope', [True, False])
def test_cursor_setup_and_disconnect(tmp_path, project_scope):
    project = tmp_path / 'workspace' if project_scope else None
    root = (project or tmp_path / 'home') / '.cursor'
    root.mkdir(parents=True)
    config = root / 'mcp.json'
    config.write_text('{/* retain */ "mcpServers": {"other": {"command": "other"}}}')
    hooks = root / 'hooks.json'
    hooks.write_text('{"version":1,"hooks":{"beforeSubmitPrompt":[{"command":"mine"}]}}')
    before_hooks = hooks.read_bytes()
    cursor.connect(project)
    original = config.read_bytes()
    cursor.connect(project)
    assert config.read_bytes() == original
    server = json_config.get(config.read_text(), ['mcpServers', 'ormah'])[1]
    assert server['args'][:4] == ['-m', 'ormah.integrations.mcp', '--host', 'cursor']
    assert ('--workspace' in server['args']) == project_scope
    if project_scope:
        assert server['args'][-1] == str(project)
        assert 'alwaysApply: true' in (root / 'rules/ormah.mdc').read_text()
    assert cursor.status(project)['tools'] == 'mcp'
    assert cursor.status(project)['whisper'] == 'blocked'
    cursor.disconnect(project)
    assert json_config.get(config.read_text(), ['mcpServers'])[1] == {'other': {'command': 'other'}}
    assert '/* retain */' in config.read_text()
    assert hooks.read_bytes() == before_hooks
    assert cursor.status(project)['tools'] == 'unconfigured'


def test_user_modified_mcp_is_preserved(tmp_path):
    cursor.connect()
    path = tmp_path / 'home/.cursor/mcp.json'
    data = json.loads(path.read_text())
    data['mcpServers']['ormah']['env'] = {'ORMAH_SPACE': 'edited'}
    path.write_text(json.dumps(data))
    cursor.disconnect()
    assert json.loads(path.read_text()) == data


def test_corrupt_cursor_config_is_not_overwritten(tmp_path):
    root = tmp_path / 'home/.cursor'
    root.mkdir(parents=True)
    (root / 'mcp.json').write_text('{broken')
    with pytest.raises(ValueError):
        cursor.connect()
    assert (root / 'mcp.json').read_text() == '{broken'


def test_explicit_transport_and_runtime_variable_references(tmp_path, monkeypatch):
    monkeypatch.setenv('ORMAH_URL', 'http://scratch-daemon:9876')
    monkeypatch.setenv('ORMAH_AUTH_TOKEN', 'scratch-secret-never-serialized')
    cursor.connect(tmp_path)
    config = (tmp_path / '.cursor/mcp.json').read_text()
    server = json_config.get(config, ['mcpServers', 'ormah'])[1]
    assert server['type'] == 'stdio'
    assert server['env']['ORMAH_URL'] == '${env:ORMAH_URL}'
    assert server['env']['ORMAH_AUTH_TOKEN'] == '${env:ORMAH_AUTH_TOKEN}'
    assert 'scratch-secret-never-serialized' not in config
