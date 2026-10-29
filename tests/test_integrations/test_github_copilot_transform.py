"""New Copilot runtime must preserve its model-facing prompt while adding memory."""
import json

import httpx
import pytest

from ormah.integrations import github_copilot_hook as hook
from ormah.integrations.hosts import github_copilot as host


def payload(tmp_path):
    return {'sessionId': 'native-session', 'timestamp': 1793275200000,
            'cwd': str(tmp_path), 'prompt': 'What did we decide?',
            'transformedPrompt': '<attachments>keep exactly</attachments>\nWhat did we decide?'}


async def test_transform_preserves_prompt_and_adds_scoped_memory(tmp_path, httpx_mock):
    original = payload(tmp_path)
    httpx_mock.add_response(json={'text': 'fixture memory'})
    result = await hook.handle_transformed(original)
    assert result == {'modifiedTransformedPrompt': original['transformedPrompt'] + '\n\nfixture memory'}
    sent = json.loads(httpx_mock.get_request().content)
    assert sent['prompt'] == 'What did we decide?'
    assert sent['space'] == tmp_path.name
    assert sent['session_id'].startswith('github_copilot_runtime:')
    assert sent['session_id'].endswith(':native-session')


@pytest.mark.parametrize('key', ['sessionId', 'cwd', 'prompt', 'transformedPrompt'])
@pytest.mark.parametrize('bad', [None, '', '  ', 1, [], {}])
async def test_transform_bad_input_leaves_prompt_unchanged(tmp_path, key, bad):
    value = payload(tmp_path)
    value[key] = bad
    assert await hook.handle_transformed(value) == {}


@pytest.mark.parametrize('value', [None, [], 'text', {'hook_event_name': 'UserPromptSubmit'}])
async def test_transform_rejects_other_payloads(value):
    assert await hook.handle_transformed(value) == {}


async def test_transform_offline_leaves_prompt_unchanged(tmp_path, httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError('offline'))
    assert await hook.handle_transformed(payload(tmp_path)) == {}


async def test_transform_no_memory_leaves_prompt_unchanged(tmp_path, httpx_mock):
    httpx_mock.add_response(json={'text': ''})
    assert await hook.handle_transformed(payload(tmp_path)) == {}


async def test_transform_project_scope_overrides_event_cwd(tmp_path, httpx_mock):
    value = payload(tmp_path)
    value['cwd'] = '/wrong'
    httpx_mock.add_response(json={'text': 'memory'})
    await hook.handle_transformed(value, str(tmp_path))
    assert json.loads(httpx_mock.get_request().content)['space'] == tmp_path.name


def test_dual_registration_and_disconnect(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path/'home'))
    host.connect(tmp_path)
    hooks = tmp_path/'.github/hooks'
    legacy = json.loads((hooks/'ormah-vscode.json').read_text())
    native = json.loads((hooks/'ormah-copilot.json').read_text())
    assert list(legacy['hooks']) == ['UserPromptSubmit']
    assert native['version'] == 1
    assert list(native['hooks']) == ['userPromptTransformed']
    command = native['hooks']['userPromptTransformed'][0]
    assert '--event userPromptTransformed' in command['command']
    assert command['cwd'] == '.'
    before = (hooks/'ormah-copilot.json').read_bytes()
    host.connect(tmp_path)
    assert (hooks/'ormah-copilot.json').read_bytes() == before
    host.disconnect(tmp_path)
    assert not (hooks/'ormah-copilot.json').exists()
    assert not (hooks/'ormah-vscode.json').exists()


async def test_new_event_does_not_share_legacy_injection_state(tmp_path, httpx_mock):
    # The server suppresses repeated injections in a session. New Copilot also
    # fires the legacy event, but discards that event's command output.
    served = set()

    def respond(request):
        body = json.loads(request.content)
        key = body['session_id']
        text = '' if key in served else 'fixture memory'
        served.add(key)
        return httpx.Response(200, json={'text': text})

    httpx_mock.add_callback(respond, is_reusable=True)
    native = payload(tmp_path)
    legacy = {'hook_event_name': 'UserPromptSubmit', 'session_id': native['sessionId'],
              'prompt': native['prompt'], 'cwd': native['cwd']}
    await hook.handle(legacy)
    output = await hook.handle_transformed(native)
    assert output == {'modifiedTransformedPrompt': native['transformedPrompt'] + '\n\nfixture memory'}
    # A repeat of the same actual event still uses stable session identity.
    assert await hook.handle_transformed(native) == {}
