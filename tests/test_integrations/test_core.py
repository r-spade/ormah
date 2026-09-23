"""Contract-independent safety tests for the shared integration machinery."""
import asyncio
import json

import httpx
import pytest

from ormah.integrations import json_config as jc
from ormah.integrations.ownership import Installation
from ormah.integrations import runtime


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a": NaN}', '{"a":1}junk',
                                  '{"a": [}', '{"a" 1}', '{"a":1 "b":2}'])
def test_corrupt_configuration_rejected(text):
    with pytest.raises(ValueError):
        jc.parse(text)


@pytest.mark.parametrize("text", ['{}', '{/* keep */}', '{"user": 1}',
                                  '{"user": 1, // keep\n}', '{"user": "https://x/*"}'])
def test_leaf_edit_preserves_comments_and_values(text):
    updated = jc.put(text, ["mcpServers", "ormah"], {"command": "python"})
    assert jc.get(updated, ["mcpServers", "ormah"])[1] == {"command": "python"}
    cleaned = jc.put(updated, ["mcpServers", "ormah"], None, delete=True)
    assert jc.get(cleaned, ["mcpServers"])[1] == {}
    if '"user"' in text:
        assert jc.get(cleaned, ["user"]) == jc.get(text, ["user"])
    if "keep" in text:
        assert "keep" in updated and "keep" in cleaned


@pytest.mark.parametrize("position", ["first", "last", "only"])
def test_remove_array_item_preserves_neighbor_comments(position):
    text = {'first': '{"hooks": [1, /* neighbor */ 2]}',
            'last': '{"hooks": [2, /* neighbor */ 1]}',
            'only': '{"hooks": [/* neighbor */ 1,]}'}[position]
    result = jc.remove_item(text, ["hooks"], 1)
    assert "/* neighbor */" in result
    assert jc.get(result, ["hooks"])[1] == ([] if position == "only" else [2])


def test_setup_idempotent_disconnect_preserves_edits(tmp_path):
    config, receipt, asset = (tmp_path / n for n in ['config.json', 'receipt.json', 'guide.md'])
    config.write_text('{ // model choice\n"model": "chosen", "hooks": [/* personal */ 1,]}')
    for _ in range(2):
        install = Installation(receipt)
        install.value(config, ["mcp", "ormah"], {"command": "test"})
        install.item(config, ["hooks"], {"command": "ormah"})
        install.file(asset, "owned")
        install.commit()
    assert Installation(receipt).intact()
    asset.write_text("user edited")
    assert Installation(receipt).disconnect() == [str(asset)]
    assert asset.read_text() == "user edited"
    assert jc.get(config.read_text(), ["hooks"])[1] == [1]
    assert "// model choice" in config.read_text()
    assert "/* personal */" in config.read_text()
    assert jc.get(config.read_text(), ["model"])[1] == "chosen"


def test_corrupt_config_does_not_partially_install_or_disconnect(tmp_path):
    first, second, receipt = (tmp_path / n for n in ['first.json', 'second.json', 'receipt.json'])
    first.write_text('{}')
    second.write_text('broken')
    install = Installation(receipt)
    install.value(first, ['ormah'], 1)
    with pytest.raises(ValueError):
        install.value(second, ['ormah'], 1)
    assert first.read_text() == '{}'
    assert not receipt.exists()


def test_conflicting_entry_never_overwritten(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"ormah":{"command":"mine"}}')
    with pytest.raises(ValueError, match="Preserving"):
        Installation(tmp_path / 'receipt').value(config, ['ormah'], {'command': 'ours'})
    assert json.loads(config.read_text())['ormah']['command'] == 'mine'


async def test_whisper_context_session_auth_and_space(tmp_path, monkeypatch, httpx_mock):
    monkeypatch.setenv('ORMAH_URL', 'http://test.local:9876')
    monkeypatch.setenv('ORMAH_AUTH_TOKEN', 'scratch-token')
    monkeypatch.setenv('ORMAH_SPACE', 'override')
    httpx_mock.add_response(url='http://test.local:9876/agent/whisper', json={'text': 'memory'})
    assert await runtime.whisper('cursor', 'hello', 'same', str(tmp_path)) == 'memory'
    request = httpx_mock.get_request()
    assert request.headers['authorization'] == 'Bearer scratch-token'
    payload = json.loads(request.content)
    assert payload['space'] == 'override'
    assert payload['session_id'].startswith('cursor:')
    assert payload['session_id'] != runtime.session_key('cline', 'same', str(tmp_path))
    assert payload['session_id'] != runtime.session_key('cursor', 'same', str(tmp_path / 'other'))


@pytest.mark.parametrize('response', [{'text': 1}, [], {'missing': 'text'}])
async def test_invalid_whisper_response_fails_open(response, httpx_mock):
    httpx_mock.add_response(json=response)
    assert await runtime.whisper('test', 'hello', 'session', None) == ''


async def test_offline_fails_open(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError('offline'))
    assert await runtime.whisper('test', 'hello', 'session', None) == ''


async def test_timeout_and_cancellation(monkeypatch):
    async def slow(_):
        await asyncio.sleep(30)
    monkeypatch.setattr(runtime, 'space_for', slow)
    monkeypatch.setenv('ORMAH_WHISPER_TIMEOUT', '0.1')
    assert await asyncio.wait_for(runtime.whisper('test', 'hello', 's', None), 0.5) == ''
    task = asyncio.create_task(runtime.whisper('test', 'hello', 's', None))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_space_detection_and_global(tmp_path, monkeypatch):
    monkeypatch.delenv('ORMAH_SPACE', raising=False)
    assert await runtime.space_for(str(tmp_path)) == tmp_path.name
    assert await runtime.space_for(None) is None
    assert await runtime.space_for('/') is None


@pytest.mark.parametrize('args,expected_space,has_default', [
    ({'content': 'global', 'space': None}, None, False),
    ({'content': 'scoped', 'space': 'other'}, 'other', False),
    ({'content': 'project'}, None, True),
])
async def test_mcp_explicit_null_bypasses_project_default(args, expected_space, has_default, httpx_mock):
    from ormah.adapters.mcp_adapter import _dispatch
    httpx_mock.add_response(json={'text': 'saved'})
    assert await _dispatch('http://test.local', 'remember', args, default_space='project',
                           headers={'Authorization': 'Bearer scratch'}) == 'saved'
    request = httpx_mock.get_request()
    assert ('default_space' in request.url.params) == has_default
    assert json.loads(request.content).get('space') == expected_space
    assert request.headers['authorization'] == 'Bearer scratch'


def test_json5_ownership_preserves_native_host_syntax(tmp_path):
    config, receipt = tmp_path / 'host.json', tmp_path / 'receipt.json'
    text = "{ // host comment\n model: 'chosen', nested: { value: 0x20, }, list: ['mine'],}"
    config.write_text(text)
    install = Installation(receipt, json5=True)
    install.value(config, ['mcp', 'ormah'], {'command': 'python'})
    install.item(config, ['list'], 'ormah')
    install.commit()
    assert Installation(receipt, json5=True).intact()
    Installation(receipt, json5=True).disconnect()
    result = config.read_text()
    assert "model: 'chosen'" in result and 'value: 0x20' in result
    assert '// host comment' in result
    assert jc.get(result, ['list'], json5=True)[1] == ['mine']


def test_json5_adjacent_comments_remain_outside_owned_value():
    text = '{max:1/* keep number */, enabled:true// keep boolean\n}'
    updated = jc.put(text, ['ormah'], {'command': 'python'}, json5=True)
    assert jc.get(updated, ['max'], json5=True)[1] == 1
    assert jc.get(updated, ['enabled'], json5=True)[1] is True
    edited = jc.put(updated, ['max'], 2, json5=True)
    assert '2/* keep number */' in edited
    assert '// keep boolean' in edited
    assert jc.get(edited, ['ormah'], json5=True)[1] == {'command': 'python'}
