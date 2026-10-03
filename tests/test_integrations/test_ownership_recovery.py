"""Interrupted setup must recover without adopting unowned or edited entries."""
import json

import pytest

from ormah.integrations import json_config as jc
from ormah.integrations import ownership


def installation(tmp_path):
    install = ownership.Installation(tmp_path / 'receipt.json')
    install.value(tmp_path / 'host.json', ['mcp', 'ormah'], {'command': 'ours'})
    install.item(tmp_path / 'host.json', ['plugins'], 'ormah')
    install.file(tmp_path / 'guide.md', 'owned guide')
    install.item(tmp_path / 'other.json', ['hooks'], {'command': 'ours'})
    return install


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
@pytest.mark.parametrize('after_replace', [False, True])
@pytest.mark.parametrize('write_number', range(1, 8))
def test_every_write_boundary_recovers(tmp_path, monkeypatch, failure, after_replace, write_number):
    # Three files, including two edits to one JSONC file: initial journal plus
    # three replacements and three checkpoints. Also exercise lost acknowledgments.
    config = tmp_path / 'host.json'
    config.write_text('{ // keep\n"plugins": ["mine"], "token": "user-secret"}')
    original = ownership.atomic_write
    calls = 0

    def interrupted(path, text):
        nonlocal calls
        calls += 1
        if calls == write_number and not after_replace:
            raise failure('interrupted before replacement')
        original(path, text)
        if calls == write_number:
            raise failure('interrupted after replacement')

    with monkeypatch.context() as patch:
        patch.setattr(ownership, 'atomic_write', interrupted)
        with pytest.raises(failure):
            installation(tmp_path).commit()
    receipt = tmp_path / 'receipt.json'
    if receipt.exists():
        assert 'user-secret' not in receipt.read_text()
    # Retry through a fresh client; a successful repeated setup is idempotent.
    installation(tmp_path).commit()
    installation(tmp_path).commit()
    assert isinstance(json.loads(receipt.read_text()), list)
    assert ownership.Installation(receipt).intact()
    assert jc.get(config.read_text(), ['plugins'])[1] == ['mine', 'ormah']
    assert ownership.Installation(receipt).disconnect() == []
    assert jc.get(config.read_text(), ['plugins'])[1] == ['mine']
    assert jc.get(config.read_text(), ['token'])[1] == 'user-secret'
    assert '// keep' in config.read_text()
    assert not (tmp_path / 'guide.md').exists()


def fail_on_config(monkeypatch, config):
    original = ownership.atomic_write

    def fail(path, text):
        if path == config:
            raise OSError('write failed')
        original(path, text)

    monkeypatch.setattr(ownership, 'atomic_write', fail)


def test_legacy_committed_deletion_is_not_an_interrupted_install(tmp_path):
    installation(tmp_path).commit()
    config = tmp_path / 'host.json'
    config.write_text(jc.remove_item(config.read_text(), ['plugins'], 'ormah'))
    before = config.read_text()
    with pytest.raises(ValueError, match='registration was edited'):
        installation(tmp_path)
    assert config.read_text() == before


@pytest.mark.parametrize('kind', ['item', 'value', 'file'])
def test_pending_user_edit_is_preserved_on_retry_and_disconnect(tmp_path, monkeypatch, kind):
    config, receipt = tmp_path / 'config', tmp_path / 'receipt'
    config.write_text('{}' if kind != 'file' else '')
    if kind == 'file':
        config.unlink()

    def plan():
        install = ownership.Installation(receipt)
        if kind == 'file':
            install.file(config, 'ours')
        else:
            getattr(install, kind)(config, ['ormah'], 'ours')
        return install

    with monkeypatch.context() as patch:
        fail_on_config(patch, config)
        with pytest.raises(OSError):
            plan().commit()
    edited = {'item': '{"ormah":["mine"]}', 'value': '{"ormah":"mine"}',
              'file': 'my guide'}[kind]
    config.write_text(edited)
    with pytest.raises(ValueError, match='changed after interrupted setup'):
        plan()
    assert config.read_text() == edited
    assert ownership.Installation(receipt).disconnect() == [str(config)]
    assert config.read_text() == edited


def test_partial_checkpoint_preserves_unrelated_edits_and_existing_ownership(tmp_path, monkeypatch):
    config = tmp_path / 'host.json'
    config.write_text('{"plugins":["mine"]}')
    # Begin with a committed registration in the same file as a new item.
    receipt = tmp_path / 'receipt.json'
    first = ownership.Installation(receipt)
    first.value(config, ['mcp', 'ormah'], {'command': 'ours'})
    first.commit()
    with monkeypatch.context() as patch:
        fail_on_config(patch, tmp_path / 'guide.md')
        with pytest.raises(OSError):
            installation(tmp_path).commit()
    # The first file was checkpointed; unrelated subsequent edits must survive.
    config.write_text(jc.put(config.read_text(), ['model'], 'chosen'))
    installation(tmp_path).commit()
    assert ownership.Installation(receipt).intact()
    ownership.Installation(receipt).disconnect()
    assert jc.get(config.read_text(), ['model'])[1] == 'chosen'
    assert jc.get(config.read_text(), ['plugins'])[1] == ['mine']


def test_pending_addition_keeps_prior_committed_ownership(tmp_path, monkeypatch):
    config, receipt = tmp_path / 'host.json', tmp_path / 'receipt.json'
    first = ownership.Installation(receipt)
    first.item(config, ['plugins'], 'ormah')
    first.commit()

    def plan():
        install = ownership.Installation(receipt)
        install.item(config, ['plugins'], 'ormah')
        install.item(config, ['plugins'], 'new')
        return install

    with monkeypatch.context() as patch:
        fail_on_config(patch, config)
        with pytest.raises(OSError):
            plan().commit()
    plan().commit()
    assert jc.get(config.read_text(), ['plugins'])[1] == ['ormah', 'new']
    ownership.Installation(receipt).disconnect()
    assert jc.get(config.read_text(), ['plugins'])[1] == []


def test_disconnect_after_partial_setup(tmp_path, monkeypatch):
    config = tmp_path / 'host.json'
    config.write_text('{"plugins":["mine"]}')
    with monkeypatch.context() as patch:
        fail_on_config(patch, tmp_path / 'guide.md')
        with pytest.raises(OSError):
            installation(tmp_path).commit()
    assert not ownership.Installation(tmp_path / 'receipt.json').intact()
    assert ownership.Installation(tmp_path / 'receipt.json').disconnect() == []
    assert jc.get(config.read_text(), ['plugins'])[1] == ['mine']
    installation(tmp_path).commit()
    assert ownership.Installation(tmp_path / 'receipt.json').intact()
