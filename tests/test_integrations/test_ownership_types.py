"""User edits must not become owned through Python's bool/number equality."""
import json

import pytest

from ormah.integrations import json_config
from ormah.integrations.ownership import Installation


class StyledMap(dict):
    """Like a comment-preserving YAML mapping."""


class StyledList(list):
    """Like a comment-preserving YAML sequence."""


class StyledString(str):
    """Like a quoted YAML string."""


class StyledInt(int):
    """Like a YAML integer retaining its formatting."""


def test_parser_style_wrappers_preserve_semantic_types():
    current = StyledMap(options=StyledList([StyledString("ours"), StyledInt(1), True]))
    assert json_config.values_equal(current, {"options": ["ours", 1, True]})
    assert json_config.values_equal({"options": ["ours", 1, True]}, current)
    assert not json_config.values_equal(current, {"options": ["ours", True, 1]})


@pytest.mark.parametrize("owned,edited", [
    (True, 1), (False, 0), (1, True), (0, False),
    ({"enabled": True}, {"enabled": 1}),
    ({"options": [False]}, {"options": [0]}),
])
@pytest.mark.parametrize("kind", ["value", "item"])
@pytest.mark.parametrize("operation", ["intact", "reconnect", "disconnect"])
def test_type_changed_registration_is_preserved(tmp_path, owned, edited, kind, operation):
    config, receipt = tmp_path / "host.json", tmp_path / "receipt.json"
    install = Installation(receipt)
    getattr(install, kind)(config, ["ormah"], owned)
    install.commit()
    text = json.dumps({"ormah": [edited] if kind == "item" else edited})
    config.write_text(text)

    install = Installation(receipt)
    if operation == "intact":
        assert not install.intact()
    elif operation == "reconnect":
        with pytest.raises(ValueError):
            getattr(install, kind)(config, ["ormah"], owned)
    else:
        assert install.disconnect() == [str(config)]
    assert config.read_text() == text


@pytest.mark.parametrize("kind", ["value", "item"])
def test_new_registration_cannot_adopt_type_changed_receipt(tmp_path, kind):
    config, receipt = tmp_path / "host.json", tmp_path / "receipt.json"
    install = Installation(receipt)
    getattr(install, kind)(config, ["ormah"], {"enabled": True})
    install.commit()
    edited = {"enabled": 1}
    text = json.dumps({"ormah": [edited] if kind == "item" else edited})
    config.write_text(text)

    with pytest.raises(ValueError):
        getattr(Installation(receipt), kind)(config, ["ormah"], edited)
    assert config.read_text() == text


@pytest.mark.parametrize("json5,text", [
    (False, '{"hooks": [1, /* keep */ true, 0, false]}'),
    (True, '{hooks: [1, /* keep */ true, 0, false]}'),
])
def test_remove_item_selects_exact_type_and_preserves_comments(json5, text):
    result = json_config.remove_item(text, ["hooks"], True, json5=json5)
    result = json_config.remove_item(result, ["hooks"], False, json5=json5)
    found, remaining = json_config.get(result, ["hooks"], json5=json5)
    assert found
    assert json.dumps(remaining) == "[1, 0]"
    assert "/* keep */" in result


def test_remove_item_leaves_numeric_lookalike_untouched():
    text = '{"hooks": [{"enabled": 1}]}'
    assert json_config.remove_item(text, ["hooks"], {"enabled": True}) == text


@pytest.mark.parametrize("kind", ["value", "item"])
def test_unchanged_nested_registration_remains_idempotent(tmp_path, kind):
    config, receipt = tmp_path / "host.json", tmp_path / "receipt.json"
    value = {"enabled": True, "options": [False, 1, 0, 1.5, None, "text"]}
    for _ in range(2):
        install = Installation(receipt)
        getattr(install, kind)(config, ["ormah"], value)
        install.commit()
        assert Installation(receipt).intact()
    assert Installation(receipt).disconnect() == []
    assert json.loads(config.read_text()) == ({"ormah": []} if kind == "item" else {})
