"""Small JSONC editor: replace only the requested value, retaining other text.

Supports the JSON-with-comments and trailing-commas dialect used by editors.
Rejects duplicate keys and malformed documents rather than guessing ownership.
No reserialization of a user's whole configuration, comments, or key order.
"""
from __future__ import annotations

import json
import json5 as json5_parser
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(
    r'\s+|//[^\n]*|/\*[\s\S]*?\*/|"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
    r'|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\[\]:,]'
)

_TOKEN5 = re.compile(
    r"\s+|//[^\n]*|/\*[\s\S]*?\*/|'(?:\\[\s\S]|[^'\\])*'"
    r'|"(?:\\[\s\S]|[^"\\])*"|[{}\[\]:,]|[^\s{}\[\]:,]+ '
    .rstrip()
)


@dataclass
class Node:
    start: int
    end: int
    value: object
    children: dict | list = field(default_factory=dict)
    key_start: int | None = None
    comma: int | None = None


def parse(text: str, *, json5: bool = False) -> Node:
    if json5:
        json5_parser.loads(text, allow_duplicate_keys=False)
    tokens = []
    pos = 0
    for match in (_TOKEN5 if json5 else _TOKEN).finditer(text):
        if match.start() != pos:
            raise ValueError(f"Invalid JSONC at character {pos}")
        pos = match.end()
        token = match.group()
        if token.isspace() or token.startswith(("//", "/*")):
            continue
        tokens.append((token, match.start(), match.end()))
    if pos != len(text):
        raise ValueError(f"Invalid JSONC at character {pos}")
    index = 0

    def read() -> Node:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("Incomplete JSONC")
        token, start, end = tokens[index]
        index += 1
        if token not in ("{", "["):
            try:
                return Node(start, end, json5_parser.loads(token) if json5 else json.loads(token))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSONC value at {start}") from exc
        mapping = token == "{"
        closing = "}" if mapping else "]"
        children = {} if mapping else []
        while index < len(tokens) and tokens[index][0] != closing:
            key_start = None
            key = None
            if mapping:
                raw, key_start, _ = tokens[index]
                if not json5 and not raw.startswith('"'):
                    raise ValueError("JSONC object keys must be quoted")
                key = (next(iter(json5_parser.loads("{" + raw + ":0}")))
                       if json5 else json.loads(raw))
                if key in children:
                    raise ValueError(f"Duplicate JSONC key: {key}")
                index += 1
                if index >= len(tokens) or tokens[index][0] != ":":
                    raise ValueError("Missing JSONC colon")
                index += 1
            child = read()
            child.key_start = key_start
            if mapping:
                children[key] = child
            else:
                children.append(child)
            if index < len(tokens) and tokens[index][0] == ",":
                child.comma = tokens[index][1]
                index += 1
            elif index >= len(tokens) or tokens[index][0] != closing:
                raise ValueError("Missing JSONC comma")
        if index >= len(tokens):
            raise ValueError("Unclosed JSONC container")
        end = tokens[index][2]
        index += 1
        value = ({k: v.value for k, v in children.items()} if mapping
                 else [v.value for v in children])
        return Node(start, end, value, children)

    root = read()
    if index != len(tokens):
        raise ValueError("Trailing JSONC content")
    return root


def get(text: str, keys: list[str], *, json5: bool = False) -> tuple[bool, object]:
    node = parse(text, json5=json5)
    for key in keys:
        if not isinstance(node.value, dict):
            raise ValueError(f"Expected object at {key}")
        if key not in node.children:
            return False, None
        node = node.children[key]
    return True, node.value


def put(text: str, keys: list[str], value: object, *, delete: bool = False,
        json5: bool = False) -> str:
    """Set/delete a leaf. Newly needed parent objects are created, never clobbered."""
    if not keys:
        raise ValueError("Cannot replace configuration root")
    node = parse(text, json5=json5)
    for i, key in enumerate(keys):
        if not isinstance(node.value, dict):
            raise ValueError(f"Expected object at {key}")
        child = node.children.get(key)
        if i < len(keys) - 1:
            if child is None:
                if delete:
                    return text
                nested = value
                for part in reversed(keys[i + 1:]):
                    nested = {part: nested}
                return put(text, keys[:i + 1], nested, json5=json5)
            node = child
            continue
        if child is not None:
            if not delete:
                return text[:child.start] + json.dumps(value, ensure_ascii=False) + text[child.end:]
            start = child.key_start
            end = child.end
            if child.comma is not None:
                end = child.comma + 1
            else:
                siblings = list(node.children.values())
                previous = siblings.index(child) - 1
                if previous >= 0:
                    comma = siblings[previous].comma
                    # Remove the previous comma separately, retaining intervening comments.
                    text = text[:comma] + text[comma + 1:]
                    start -= 1
                    end -= 1
            return text[:start] + text[end:]
        if delete:
            return text
        insert = node.end - 1
        prefix = ""
        if node.children:
            last = list(node.children.values())[-1]
            if last.comma is None:
                text = text[:last.end] + "," + text[last.end:]
                insert += 1
        prefix += "\n  " + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False) + "\n"
        return text[:insert] + prefix + text[insert:]
    raise AssertionError("unreachable")


def append(text: str, keys: list[str], value: object, *, json5: bool = False) -> str:
    found, current = get(text, keys, json5=json5)
    if not found:
        return put(text, keys, [value], json5=json5)
    if not isinstance(current, list):
        raise ValueError("Expected hook/plugin array")
    node = parse(text, json5=json5)
    for key in keys:
        node = node.children[key]
    insert = node.end - 1
    if node.children and node.children[-1].comma is None:
        pos = node.children[-1].end
        text = text[:pos] + "," + text[pos:]
        insert += 1
    return text[:insert] + "\n  " + json.dumps(value, ensure_ascii=False) + "\n" + text[insert:]


def remove_item(text: str, keys: list[str], value: object, *, json5: bool = False) -> str:
    found, current = get(text, keys, json5=json5)
    if not found or not isinstance(current, list) or value not in current:
        return text
    node = parse(text, json5=json5)
    for key in keys:
        node = node.children[key]
    index = current.index(value)
    child = node.children[index]
    start, end = child.start, child.end
    if child.comma is not None:
        end = child.comma + 1
    elif index > 0:
        comma = node.children[index - 1].comma
        text = text[:comma] + text[comma + 1:]
        start -= 1
        end -= 1
    return text[:start] + text[end:]
