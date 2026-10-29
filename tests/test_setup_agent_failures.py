"""A refused client configuration must not prevent other clients being wired."""
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from ormah.integrations.ownership import Installation
from ormah.setup import AgentDescriptor, run_setup


@pytest.mark.parametrize('ci', [False, True])
@pytest.mark.parametrize('update', [False, True])
@pytest.mark.parametrize('failure', ['collision', 'write'])
def test_setup_reports_failed_agent_and_connects_next(tmp_path, capsys, update, failure, ci):
    refused = tmp_path / 'refused.json'
    refused.write_text('{"ormah": "user-owned"}')
    connected = tmp_path / 'connected.json'

    def wire_refused():
        if failure == 'write':
            raise PermissionError('config is read-only')
        install = Installation(tmp_path / 'refused-receipt.json')
        install.value(refused, ['ormah'], 'ours')
        install.commit()

    def wire_next():
        install = Installation(tmp_path / 'connected-receipt.json')
        install.value(connected, ['ormah'], 'ours')
        install.commit()

    agents = [
        AgentDescriptor('refused', 'Refused Host', lambda: True, lambda: False,
                        wire_refused, lambda: None),
        AgentDescriptor('next', 'Next Host', lambda: True, lambda: False,
                        wire_next, lambda: None),
    ]
    # Isolate server/process/model/UI side effects; exercise real client writes.
    with ExitStack() as stack:
        for name in ('_persist_env_delta', 'configure_llm', '_preload_local_models', 'backfill_transcripts',
                     'play_finale', '_print_setup_summary', 'webbrowser.open'):
            stack.enter_context(patch('ormah.setup.' + name))
        stack.enter_context(patch('ormah.setup._detected_agents', return_value=agents))
        stack.enter_context(patch('ormah.setup._read_env_file', return_value={}))
        stack.enter_context(patch('ormah.setup.get_ormah_bin_path', return_value='/bin/ormah'))
        stack.enter_context(patch('ormah.setup.generate_server_wrapper',
                                  return_value=tmp_path / 'server'))
        stack.enter_context(patch('ormah.setup.is_server_running', return_value=True))
        stack.enter_context(patch('ormah.setup.restart_with_autostart', return_value=True))
        if ci:
            with pytest.raises(SystemExit) as exc:
                run_setup(update=update, ci=True)
            assert exc.value.code == 1
        else:
            run_setup(update=update)

    assert refused.read_text() == '{"ormah": "user-owned"}'
    assert Installation(tmp_path / 'connected-receipt.json').intact()
    output = capsys.readouterr().out
    assert 'Refused Host' in output
    assert 'Setup complete' in output
    assert 'could not' in output.lower()
    assert ('PermissionError' if failure == 'write' else 'ValueError') in output
    assert 'ormah setup --update' in output
