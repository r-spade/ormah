"""Kiro shared V1 file format and prompt payload, documented 2026-09-23."""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from ormah.integrations import json_config as jc
from ormah.integrations.hosts import kiro as host
from ormah.integrations.kiro_hook import handle


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'home/.config'))
    monkeypatch.delenv('ORMAH_SPACE', raising=False)
    monkeypatch.delenv('ORMAH_WORKSPACE', raising=False)


@pytest.mark.parametrize('project_scope', [True, False])
def test_current_config_schema(tmp_path, monkeypatch, project_scope):
    project = tmp_path / 'project' if project_scope else None
    root = (project or Path.home()) / '.kiro'
    (root / 'settings').mkdir(parents=True)
    config = root / 'settings/mcp.json'
    config.write_text('{/* keep */ "mcpServers":{"other":{"command":"other","autoApprove":["mine"]}}}')
    (root / 'hooks').mkdir()
    (root / 'hooks/mine.json').write_text('{"version":"v1","hooks":[]}')
    monkeypatch.setenv('ORMAH_AUTH_TOKEN', 'secret-not-to-write')
    host.connect(project)
    before = config.read_bytes()
    host.connect(project)
    assert config.read_bytes() == before
    server = jc.get(config.read_text(), ['mcpServers','ormah'])[1]
    assert server['env']['ORMAH_AUTH_TOKEN'] == '${ORMAH_AUTH_TOKEN}'
    assert 'secret-not-to-write' not in config.read_text()
    assert 'autoApprove' not in server
    assert ('--workspace' in server['args']) == project_scope
    hook = json.loads((root / 'hooks/ormah.json').read_text())
    assert hook['version'] == 'v1'
    entry = hook['hooks'][0]
    assert entry['trigger'] == 'UserPromptSubmit'
    assert entry['action']['type'] == 'command' and entry['timeout'] == 12
    assert 'kiro_hook' in entry['action']['command']
    assert host.status(project)['whisper'] == 'native_hook'
    host.disconnect(project)
    assert '/* keep */' in config.read_text()
    assert jc.get(config.read_text(), ['mcpServers','other'])[1]['autoApprove'] == ['mine']
    assert (root / 'hooks/mine.json').exists()
    assert not (root / 'hooks/ormah.json').exists()


@pytest.mark.parametrize('event', ['userPromptSubmit','UserPromptSubmit'])
async def test_stdout_contract_and_scopes(tmp_path, httpx_mock, event):
    # CLI docs use camelCase; current V3 native payload uses PascalCase.
    payload = {'hook_event_name':event,'prompt':'my question','cwd':str(tmp_path),'session_id':'same-id'}
    httpx_mock.add_response(json={'text':'memory context'})
    assert await handle(payload, str(tmp_path)) == 'memory context'
    project = json.loads(httpx_mock.get_requests()[0].content)
    assert project['space'] == tmp_path.name
    assert project['session_id'].startswith('kiro:')
    httpx_mock.add_response(json={'text':'global context'})
    assert await handle(payload) == 'global context'
    global_body = json.loads(httpx_mock.get_requests()[-1].content)
    assert global_body['space'] is None and global_body['session_id'] != project['session_id']
    assert await handle(payload, str(tmp_path / 'other')) == ''


@pytest.mark.parametrize('payload', [None, {}, {'hook_event_name':'Stop'},
                                    {'hook_event_name':'UserPromptSubmit','prompt':'','cwd':'/w','session_id':'s'},
                                    {'hook_event_name':'UserPromptSubmit','prompt':'x','cwd':'/w'}])
async def test_invalid_and_old_empty_prompt_bug_fail_open(payload, monkeypatch):
    # Don't guess a session, scrape old logs or reuse a cached USER_PROMPT.
    monkeypatch.setenv('USER_PROMPT', 'stale environment text')
    assert await handle(payload) == ''


async def test_offline_and_bad_daemon(tmp_path, httpx_mock):
    payload = {'hook_event_name':'UserPromptSubmit','prompt':'x','cwd':str(tmp_path),'session_id':'s'}
    httpx_mock.add_exception(httpx.ConnectError('offline'))
    assert await handle(payload) == ''
    httpx_mock.add_response(json={'text':42})
    assert await handle(payload) == ''


def test_generated_command_plain_stdout(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"text":"fixture memory"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('ORMAH_URL', f'http://127.0.0.1:{server.server_port}')
    try:
        host.connect(tmp_path)
        command = json.loads((tmp_path / '.kiro/hooks/ormah.json').read_text())['hooks'][0]['action']['command']
        result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=15,
                                env={**os.environ, 'PYTHONPATH': str(Path('src').resolve())}, input=json.dumps({
                                    'hook_event_name':'UserPromptSubmit','prompt':'question',
                                    'cwd':str(tmp_path),'session_id':'native-session'}))
        assert result.returncode == 0 and result.stdout == 'fixture memory\n' and not result.stderr
        assert requests[0]['space'] == tmp_path.name
        result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=15,
                                env={**os.environ, 'PYTHONPATH': str(Path('src').resolve())}, input='{bad')
        assert result.returncode == 0 and result.stdout == '' and not result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_corrupt_config_and_edited_hook(tmp_path):
    path = tmp_path / '.kiro/settings/mcp.json'
    path.parent.mkdir(parents=True)
    path.write_text('{bad')
    with pytest.raises(ValueError):
        host.connect(tmp_path)
    assert path.read_text() == '{bad'
    path.write_text('{}')
    host.connect(tmp_path)
    hook = tmp_path / '.kiro/hooks/ormah.json'
    hook.write_text('{"version":"v1","hooks":[]}')
    host.disconnect(tmp_path)
    assert hook.read_text() == '{"version":"v1","hooks":[]}'


def test_pinned_native_adapter_consumes_stdout(tmp_path):
    """Execute shipped KAS functions, not a reimplementation of its consumer."""
    bundle = os.environ.get('ORMAH_TEST_KIRO_BUNDLE')
    if not bundle:
        pytest.skip('Set ORMAH_TEST_KIRO_BUNDLE to the documented @kiro/agent 0.66.8 acp-server.js')
    # Names/boundaries are pinned to the official CLI 2.23.1 archive. A host
    # change must be inspected explicitly instead of silently adapting the test.
    script = tmp_path / 'consumer.mjs'
    script.write_text(r'''
import fs from 'node:fs';
import assert from 'node:assert/strict';
const source = fs.readFileSync(process.argv[2], 'utf8');
function between(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  assert(a >= 0 && b > a);
  return source.slice(a, b);
}
const fragment = between('function F_t(', 'function Heo(')
  + between('function d2c(', 'async function L_t(')
  + between('function pvr(', 'var fvr=')
  + between('function hvr(', 'var YZa,')
  + between('function ZZa(', 'function JZa(');
const create = new Function('w', 'Bt', 'lde', 'lRe', 'YZa', 'TDi', 'XZa', fragment + '; return {zeo,pvr,hvr,ZZa};');
const native = create({debug(){}}, {fromHuman:()=>({withEntry:entry=>entry})},
  '<HOOK_INSTRUCTION>', '</HOOK_INSTRUCTION>',
  {sendStdout:false,sendStderr:false,block:false},
  {sendStdout:true,sendStderr:false,block:false},
  {sendStdout:false,sendStderr:true,block:true});
let received, executions=0;
const loaded = {id:'ormah',trigger:'UserPromptSubmit',action:{kind:'command'}};
const module = {workspaceTrusted:true,registry:{list:()=>[loaded]},executor:{execute:async (hooks, request)=>{
  executions++; received=native.pvr(request);
  const commandResult={exitCode:0,stdout:'actual fixture memory',stderr:''};
  const behavior=native.hvr(request.trigger,commandResult.exitCode);
  return {results:[{commandResult,appendix:native.ZZa(commandResult.stdout,'',behavior),blocked:behavior.block}]};
}}};
const state={chatSessionId:'s',execution:{rootConversationId:'r'},
  context:{withNewMessage:message=>({messages:[message]})}};
const hook={id:'ormah',action:{type:'runCommand'}};
const adapter=native.zeo(module,'/project');
const result=await adapter.executeHookAction(hook,state,'latest user question');
assert.deepEqual(received,{session_id:'s',hook_event_name:'UserPromptSubmit',cwd:'/project',prompt:'latest user question'});
assert.equal(result.context.messages[0].text,'<HOOK_INSTRUCTION>\nactual fixture memory\n</HOOK_INSTRUCTION>');
module.workspaceTrusted=false;
assert.equal(await adapter.executeHookAction(hook,state,'other'),state);
assert.equal(executions,1);
''')
    subprocess.run(['node', str(script), bundle], check=True, timeout=15, capture_output=True, text=True)
