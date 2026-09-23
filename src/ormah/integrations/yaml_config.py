"""Round-trip YAML editor for host configuration; never construct tagged objects."""
from io import StringIO

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError


def _load(text):
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.allow_duplicate_keys = False
    try:
        data = yaml.load(text)
    except YAMLError as exc:
        raise ValueError("Invalid YAML configuration; preserving original file") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Expected a YAML configuration mapping")
    return yaml, data


def _dump(yaml, data):
    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def _unaliased(value):
    if (getattr(value, "anchor", None) and value.anchor.value) or getattr(value, "merge", None):
        raise ValueError("Preserving aliased/merged YAML value; use an explicit host section")


def get(text, keys):
    _, data = _load(text)
    for key in keys:
        if not isinstance(data, dict):
            raise ValueError("Expected a YAML mapping at " + ".".join(keys))
        if key not in data:
            return False, None
        data = data[key]
    return True, data


def put(text, keys, value, *, delete=False):
    yaml, data = _load(text)
    target = data
    _unaliased(target)
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
        if not isinstance(target, dict):
            raise ValueError("Expected a YAML mapping at " + ".".join(keys))
        # Mutating an aliased/merged mapping could change an unrelated setting.
        _unaliased(target)
    if delete:
        target.pop(keys[-1], None)
    else:
        target[keys[-1]] = value
    return _dump(yaml, data)


def append(text, keys, value):
    found, current = get(text, keys)
    if found and not isinstance(current, list):
        raise ValueError("Expected a YAML list")
    # Keep the existing sequence's comments/style while replacing its value.
    values = current if found else []
    _unaliased(values)
    values.append(value)
    return put(text, keys, values)


def remove_item(text, keys, value):
    found, current = get(text, keys)
    if not found or not isinstance(current, list):
        return text
    _unaliased(current)
    current.remove(value)
    return put(text, keys, current)
