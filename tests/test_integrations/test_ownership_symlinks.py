"""Connecting and disconnecting must preserve dotfile symlinks."""
import json

import pytest

from ormah.integrations.ownership import Installation


@pytest.mark.parametrize('relative', [True, False])
@pytest.mark.parametrize('existing', [True, False])
def test_config_symlink_survives_connect_and_disconnect(tmp_path, relative, existing):
    target = tmp_path / 'dotfiles' / 'host.json'
    if existing:
        target.parent.mkdir()
        target.write_text('{"mine": true}')
        target.chmod(0o640)
    link = tmp_path / 'host.json'
    link.symlink_to('dotfiles/host.json' if relative else target)
    receipt = tmp_path / 'receipt.json'
    install = Installation(receipt)
    install.value(link, ['mcpServers', 'ormah'], {'command': 'ormah'})
    install.commit()

    assert link.is_symlink()
    assert json.loads(target.read_text())['mcpServers']['ormah'] == {'command': 'ormah'}
    assert Installation(receipt).intact()
    if existing:
        assert target.stat().st_mode & 0o777 == 0o640
    assert Installation(receipt).disconnect() == []
    assert link.is_symlink()
    data = json.loads(target.read_text())
    assert 'ormah' not in data.get('mcpServers', {})
    if existing:
        assert data['mine'] is True
