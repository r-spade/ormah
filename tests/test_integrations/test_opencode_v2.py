"""V2 prompt admission contract through the actual JS/Python/HTTP bridge."""
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def daemon():
    requests = []
    response = {'body': {'text': 'fixture memory'}, 'status': 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = response['body']
            body = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(response['status'])
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests, response
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_plugin(tmp_path, url, body):
    source = Path(__file__).resolve().parents[2] / 'src'
    asset = source / 'ormah/integrations/assets/opencode/v2.mjs'
    assert asset.exists(), 'V2 prompt adapter is not packaged'
    plugin = tmp_path / 'v2.mjs'
    plugin.write_text(asset.read_text().replace('__ORMAH_PYTHON__', json.dumps(sys.executable)))
    script = tmp_path / 'contract.mjs'
    project = tmp_path / 'actual-project'
    project.mkdir(exist_ok=True)
    script.write_text('''import assert from 'node:assert/strict';
const { default: plugin } = await import(process.argv[2]);
const directory = process.argv[3];
let hook;
const ctx = {
  location: {directory:'/wrong/plugin/location', project:{id:'wrong',directory:'/wrong',canonical:'/wrong'}},
  session: {
    hook: async (name, callback) => { assert.equal(name, 'prompt'); hook = callback; },
    get: async ({sessionID}) => ({id:sessionID,projectID:'project',location:{directory},time:{created:1,updated:1}}),
  },
};
await plugin.setup(ctx);
assert.equal(typeof hook, 'function');
const event = {
  sessionID:'ses_v2', messageID:'msg_v2', delivery:'queue', metadata:{source:'fixture'},
  prompt:{text:'Review @file',files:[{uri:'file:///project/file',mention:{start:7,end:12,text:'@file'}}],
    agents:[{name:'reviewer'}],skills:[{id:'review'}]},
};
''' + body)
    result = subprocess.run(
        ['node', str(script), plugin.as_uri(), str(project)],
        env={**os.environ, 'ORMAH_URL': url, 'PYTHONPATH': str(source)},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_v2_appends_memory_once_preserving_admission_and_session_scope(tmp_path, daemon):
    url, requests, _ = daemon
    run_plugin(tmp_path, url, '''
const before = structuredClone(event);
await hook(event);
assert.equal(event.prompt.text, 'Review @file\\n\\n<ormah-memory>\\nfixture memory\\n</ormah-memory>');
await hook(event);
assert.equal(event.prompt.text.match(/fixture memory/g).length, 1);
const restored = structuredClone(event);
restored.prompt.text = before.prompt.text;
assert.deepEqual(restored, before);
''')
    assert len(requests) == 1
    assert requests[0]['prompt'] == 'Review @file'
    assert requests[0]['space'] == 'actual-project'
    assert requests[0]['session_id'].startswith('opencode:')


@pytest.mark.parametrize('body,status', [({'text': ''}, 200), ({'text': 42}, 200),
                                       ('not-json', 200), ({'error': 'offline'}, 503)])
def test_v2_leaves_prompt_unchanged_without_usable_context(tmp_path, daemon, body, status):
    url, requests, response = daemon
    response.update(body=body, status=status)
    run_plugin(tmp_path, url, '''
const before = structuredClone(event);
await hook(event);
assert.deepEqual(event, before);
''')
    assert len(requests) == 1


@pytest.mark.parametrize('change', [
    "event.sessionID = '   ';", "event.sessionID = 17;",
    "event.prompt.text = '  ';", "event.prompt.text = null;",
    "ctx.session.get = async () => ({location:{directory:'  '}});",
    "ctx.session.get = async () => ({location:{directory:17}});",
    "ctx.session.get = async () => { throw new Error('session unavailable'); };",
    "ctx.session.get = async () => ({});",
])
def test_v2_fails_open_without_valid_prompt_or_session_location(tmp_path, daemon, change):
    url, requests, _ = daemon
    run_plugin(tmp_path, url, change + '''
const before = structuredClone(event);
await hook(event);
assert.deepEqual(event, before);
''')
    assert requests == []


def test_v2_bounds_stalled_session_lookup(tmp_path, daemon):
    url, requests, _ = daemon
    run_plugin(tmp_path, url, '''
ctx.session.get = async () => new Promise(() => {});
const before = structuredClone(event);
const deadline = setTimeout(() => { console.error('session lookup hung'); process.exit(1); }, 4000);
await hook(event);
clearTimeout(deadline);
assert.deepEqual(event, before);
''')
    assert requests == []


def test_v2_fails_open_when_daemon_is_offline(tmp_path):
    # Reserve an unused local port without listening, so connection is refused.
    with socket.socket() as unavailable:
        unavailable.bind(('127.0.0.1', 0))
        url = f'http://127.0.0.1:{unavailable.getsockname()[1]}'
        run_plugin(tmp_path, url, '''
const before = structuredClone(event);
await hook(event);
assert.deepEqual(event, before);
''')
