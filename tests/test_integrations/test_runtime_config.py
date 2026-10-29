"""New hosts must find the same configured daemon as the Ormah server."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(('local_port', 'env_port', 'url', 'expected'), [
    (None, None, None, 'http://127.0.0.1:9123'),
    ('9234', None, None, 'http://127.0.0.1:9234'),
    ('9234', '9345', None, 'http://127.0.0.1:9345'),
    ('9234', '9345', 'http://daemon:9456/', 'http://daemon:9456'),
])
def test_runtime_follows_configured_port_precedence(tmp_path, local_port, env_port, url, expected):
    home = tmp_path / 'home'
    config = home / '.config/ormah/.env'
    config.parent.mkdir(parents=True)
    config.write_text('ORMAH_PORT=9123\n')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    if local_port:
        (workspace / '.env').write_text(f'ORMAH_PORT={local_port}\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORMAH_')}
    env.update(HOME=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
    if env_port:
        env['ORMAH_PORT'] = env_port
    if url:
        env['ORMAH_URL'] = url
    result = subprocess.run(
        [sys.executable, '-c', 'from ormah.integrations.runtime import base_url; print(base_url())'],
        cwd=workspace, env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected
