#!/usr/bin/env python3
"""Verify release versions before publishing."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


RELEASE_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[A-Za-z0-9.+_-]+)?$")
PLUGIN_MANIFEST = Path("integrations/claude-plugin/.claude-plugin/plugin.json")


def _read_project_version(root: Path) -> str:
    pyproject_path = root / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)
    return str(pyproject["project"]["version"])


def _read_plugin_version(root: Path) -> str:
    manifest_path = root / PLUGIN_MANIFEST
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    return str(manifest["version"])


def verify_release_versions(requested_version: str, root: Path) -> str:
    root = root.resolve()
    requested_version = requested_version.strip()

    if requested_version.startswith("v"):
        raise ValueError("Release version should not include the leading 'v'. Use 0.12.0.")

    if not RELEASE_VERSION_RE.fullmatch(requested_version):
        raise ValueError(
            "Release version must look like 0.12.0 and must not contain whitespace."
        )

    project_version = _read_project_version(root)
    plugin_version = _read_plugin_version(root)

    if requested_version != project_version:
        raise ValueError(
            f"Requested release version {requested_version} does not match "
            f"pyproject.toml version {project_version}."
        )

    if plugin_version != project_version:
        raise ValueError(
            f"Claude plugin manifest version {plugin_version} does not match "
            f"pyproject.toml version {project_version}."
        )

    return project_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release version without a leading v, e.g. 0.12.0")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the parent of .github/.",
    )
    args = parser.parse_args(argv)

    try:
        version = verify_release_versions(args.version, args.root)
    except (KeyError, OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"Release version verification failed: {exc}", file=sys.stderr)
        return 1

    print(f"Release version verified: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
