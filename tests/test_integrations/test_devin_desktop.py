"""Contracts from Devin Local 3000.11.1 documentation, distinct from Cascade."""
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from ormah.integrations import json_config as jc
from ormah.integrations.devin_desktop_hook import handle
from ormah.integrations.hosts import devin_desktop as host


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'home/.config'))
    monkeypatch.delenv('ORMAH_WORKSPACE', raising=False)
    monkeypatch.delenv('ORMAH_SPACE', raising=False)


@pytest.mark.parametrize('project_scope', [True, False])
def test_config_contract_and_preservation(tmp_path, monkeypatch, project_scope):
    project = tmp_path / 'project' if project_scope else None
    root = project / '.devin' if project else host.user_directory()
    root.mkdir(parents=True)
    path = root / 'mcp_config.json'
    path.write_text('{/* retain */ "mcpServers":{"other":{"command":"other"}}}')
    hooks = root / 'hooks.v1.json' if project else root / 'config.json'
    keys = ['UserPromptSubmit'] if project else ['hooks', 'UserPromptSubmit']
    hooks.write_text(jc.put('{"model":"mine"}', keys, [{'hooks':[{'command':'echo mine'}]}]))
    cascade = root / 'hooks.json'
    cascade.write_text('{"hooks":{"pre_user_prompt":[]}}')
    monkeypatch.setenv('ORMAH_AUTH_TOKEN', 'never-persist-this')
    host.connect(project)
    first = [p.read_bytes() for p in (path, hooks)]
    host.connect(project)
    assert first == [p.read_bytes() for p in (path, hooks)]
    mcp = jc.get(path.read_text(), ['mcpServers','ormah'])[1]
    assert mcp['env']['ORMAH_AUTH_TOKEN'] == '${env:ORMAH_AUTH_TOKEN}'
    assert 'never-persist-this' not in path.read_text()
    assert ('--workspace' in mcp['args']) == project_scope
    if project:
        assert mcp['args'][-1] == str(project)
    hook = jc.get(hooks.read_text(), keys)[1][-1]['hooks'][0]
    assert hook['type'] == 'command' and hook['timeout'] == 12
    assert 'devin_desktop_hook' in hook['command']
    state = host.status(project)
    assert state['whisper'] == 'native_hook'
    assert state['surfaces']['cascade'] == {'tools':'unconfigured' if project else 'mcp','whisper':'blocked'}
    host.disconnect(project)
    assert '/* retain */' in path.read_text()
    assert jc.get(hooks.read_text(), keys)[1] == [{'hooks':[{'command':'echo mine'}]}]
    assert json.loads(hooks.read_text())['model'] == 'mine'
    assert cascade.read_text() == '{"hooks":{"pre_user_prompt":[]}}'


async def test_native_prompt_to_context_and_separation(tmp_path, monkeypatch, httpx_mock):
    payload = {'hook_event_name':'UserPromptSubmit','prompt':'question','session_id':'shared','prompt_id':'turn1'}
    monkeypatch.setenv('DEVIN_PROJECT_DIR', str(tmp_path))
    httpx_mock.add_response(json={'text':'memory'})
    assert await handle(payload, str(tmp_path)) == {'hookSpecificOutput':{
        'hookEventName':'UserPromptSubmit','additionalContext':'memory'}}
    body = json.loads(httpx_mock.get_request().content)
    assert body['space'] == tmp_path.name and body['session_id'].startswith('devin_desktop:')
    # Global MCP and hook must use the same explicit default, regardless of host cwd.
    httpx_mock.add_response(json={'text':'global memory'})
    await handle(payload)
    global_body = json.loads(httpx_mock.get_requests()[-1].content)
    assert global_body['space'] is None
    assert global_body['session_id'] != body['session_id']
    assert await handle(payload, str(tmp_path / 'other')) == {}


@pytest.mark.parametrize('payload', [None, {}, {'agent_action_name':'pre_user_prompt','tool_info':{'user_prompt':'x'}},
                                    {'hook_event_name':'UserPromptSubmit','prompt':'','session_id':'s'}])
async def test_other_surfaces_and_invalid_fail_open(payload):
    assert await handle(payload) == {}


async def test_offline(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError('offline'))
    assert await handle({'hook_event_name':'UserPromptSubmit','prompt':'x','session_id':'s'}) == {}


def test_generated_command_exits_zero_for_malformed_input(tmp_path):
    host.connect(tmp_path)
    hook = jc.get((tmp_path / '.devin/hooks.v1.json').read_text(), ['UserPromptSubmit'])[1][0]['hooks'][0]
    result = subprocess.run(hook['command'], shell=True, input='bad JSON', text=True,
                            capture_output=True, timeout=15, env={**os.environ, 'PYTHONPATH': str(Path('src').resolve())})
    assert result.returncode == 0 and json.loads(result.stdout) == {} and not result.stderr


def test_corruption_and_user_edits(tmp_path):
    root = tmp_path / '.devin'
    root.mkdir()
    config = root / 'mcp_config.json'
    config.write_text('{bad')
    with pytest.raises(ValueError):
        host.connect(tmp_path)
    assert config.read_text() == '{bad'
    config.write_text('{}')
    host.connect(tmp_path)
    hooks = root / 'hooks.v1.json'
    data = json.loads(hooks.read_text())
    data['UserPromptSubmit'][0]['hooks'][0]['timeout'] = 99
    hooks.write_text(json.dumps(data))
    host.disconnect(tmp_path)
    assert json.loads(hooks.read_text()) == data


async def test_explicit_workspace_env_matches_global_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv('ORMAH_WORKSPACE', str(tmp_path))
    call = AsyncMock(return_value='')
    monkeypatch.setattr('ormah.integrations.devin_desktop_hook.whisper', call)
    await handle({'hook_event_name':'UserPromptSubmit','prompt':'x','session_id':'s'})
    call.assert_awaited_once_with('devin_desktop','x','s',str(tmp_path))
