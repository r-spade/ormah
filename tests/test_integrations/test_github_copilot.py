"""Payloads grounded in VS Code Local reference and Copilot Chat 0.44.0 types."""
import json
from pathlib import Path

import pytest

from ormah.integrations.hosts import github_copilot as host
from ormah.integrations.github_copilot_hook import handle
from ormah.integrations import json_config as jc


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'home/.config'))
    monkeypatch.setenv('ORMAH_VSCODE_USER_DIR', str(tmp_path / 'profile'))


@pytest.mark.parametrize('project_scope', [True, False])
def test_configuration_contract(tmp_path, project_scope):
    project = tmp_path / 'project' if project_scope else None
    root = project / '.vscode' if project else host.user_directory()
    root.mkdir(parents=True)
    config = root / 'mcp.json'
    config.write_text('{/* keep */ "servers":{"other":{"type":"stdio","command":"other"}}}')
    hookdir = project / '.github/hooks' if project else Path.home() / '.copilot/hooks'
    hookdir.mkdir(parents=True)
    (hookdir / 'mine.json').write_text('{"hooks": {}}')
    host.connect(project)
    before = config.read_bytes()
    host.connect(project)
    assert config.read_bytes() == before
    mcp = jc.get(config.read_text(), ['servers', 'ormah'])[1]
    assert mcp['type'] == 'stdio'
    assert mcp['args'][-1] == (str(project) if project else '${workspaceFolder}')
    hook = json.loads((hookdir / 'ormah-vscode.json').read_text())['hooks']['UserPromptSubmit'][0]
    assert hook['cwd'] == '.' and hook['timeout'] == 12
    assert 'github_copilot_hook' in hook['command']
    assert host.status(project)['whisper'] == 'native_hook'
    host.disconnect(project)
    assert (hookdir / 'mine.json').read_text() == '{"hooks": {}}'
    assert '/* keep */' in config.read_text()
    assert not (hookdir / 'ormah-vscode.json').exists()


async def test_prompt_hook_returns_context_consumed_by_local_harness(tmp_path, httpx_mock):
    # Common input assembled by chatHookService.executeHook plus prompt field.
    payload = {'timestamp': '2026-09-23T00:00:00Z', 'hook_event_name': 'UserPromptSubmit',
               'session_id': 'chat-42', 'cwd': str(tmp_path), 'prompt': 'design the API'}
    httpx_mock.add_response(json={'text': 'fixture memory'})
    output = await handle(payload)
    assert output == {'hookSpecificOutput': {
        'hookEventName': 'UserPromptSubmit', 'additionalContext': 'fixture memory'}}
    body = json.loads(httpx_mock.get_request().content)
    assert body['session_id'].startswith('github_copilot:')
    assert body['space'] == tmp_path.name
    assert 'continue' not in output and 'decision' not in output


@pytest.mark.parametrize('payload', [None, {}, {'hook_event_name': 'userPromptSubmitted'},
                                   {'hook_event_name': 'UserPromptSubmit','prompt':'x'}])
async def test_missing_identity_and_other_harnesses_fail_open(payload):
    assert await handle(payload) == {}


async def test_offline_and_explicit_workspace(tmp_path, httpx_mock):
    import httpx
    httpx_mock.add_exception(httpx.ConnectError('offline'))
    assert await handle({'hook_event_name':'UserPromptSubmit', 'prompt':'x',
                         'session_id':'s', 'cwd':'/wrong'}, str(tmp_path)) == {}


def test_mcp_custom_runtime_env_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('ORMAH_URL', 'http://scratch-daemon:9876')
    monkeypatch.setenv('ORMAH_AUTH_TOKEN', 'scratch-secret-never-serialized')
    host.connect(tmp_path)
    config = (tmp_path / '.vscode/mcp.json').read_text()
    server = jc.get(config, ['servers','ormah'])[1]
    assert server['env']['ORMAH_URL'] == '${env:ORMAH_URL}'
    assert server['env']['ORMAH_AUTH_TOKEN'] == '${env:ORMAH_AUTH_TOKEN}'
    assert 'scratch-secret-never-serialized' not in config
