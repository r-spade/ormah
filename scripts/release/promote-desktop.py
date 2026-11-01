#!/usr/bin/env python3
"""Publish a draft desktop release once its exact DMG passed clean-machine tests."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

REPO = "r-spade/ormah"


def gh(*args):
    return subprocess.check_output(["gh", *args, "-R", REPO], text=True)


def check_results(dmg_sha256, app, terminal):
    """Refuse unless both routes passed and the app result is for this exact DMG."""
    if app.get("mode") != "app" or app.get("status") != "passed":
        raise SystemExit("App acceptance did not pass")
    if app.get("artifact_sha256") != dmg_sha256:
        raise SystemExit("App acceptance was run against a different DMG")
    if terminal.get("mode") != "terminal" or terminal.get("status") != "passed":
        raise SystemExit("Terminal acceptance did not pass")
    if not app.get("ormah_version") or terminal.get("ormah_version") != app["ormah_version"]:
        raise SystemExit("App and terminal acceptance tested different Ormah versions")


def check_not_older(version, tags):
    """Refuse to point the feed back at an older release when retrying an old promotion."""
    def numbers(text):
        return tuple(map(int, text.split(".")))

    for tag in tags:
        if tag["tagName"].startswith("desktop-v") and tag["tagName"] != "desktop-latest":
            published = tag["tagName"].removeprefix("desktop-v")
            if not tag["isDraft"] and numbers(published) > numbers(version):
                raise SystemExit(f"Desktop {published} is already published")


def feed(version, notes, directory):
    base = f"https://github.com/{REPO}/releases/download/desktop-v{version}"

    def entry(suffix):
        path = next(Path(directory).glob(f"*{suffix}"))
        signature = Path(f"{path}.sig").read_text().strip()
        return {"signature": signature, "url": f"{base}/{path.name}"}

    return {
        "version": version,
        "notes": notes,
        "pub_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platforms": {
            "linux-x86_64-deb": entry(".deb"),
            "linux-x86_64": entry(".AppImage"),
            "darwin-aarch64": entry(".app.tar.gz"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Desktop version, for example 0.3.29")
    parser.add_argument("--app", required=True, help="result.json from macos-vm.sh app")
    parser.add_argument("--terminal", required=True, help="result.json from macos-vm.sh terminal")
    args = parser.parse_args()
    tag = f"desktop-v{args.version}"

    release = json.loads(gh("release", "view", tag, "--json", "isDraft,body"))
    gh("release", "view", "desktop-latest", "--json", "tagName")  # Fail before publishing.
    check_not_older(
        args.version,
        json.loads(gh("release", "list", "--limit", "1000", "--json", "tagName,isDraft")),
    )
    with tempfile.TemporaryDirectory() as tmp:
        gh("release", "download", tag, "--dir", tmp)
        dmg = next(Path(tmp).glob("*.dmg"))
        check_results(
            hashlib.sha256(dmg.read_bytes()).hexdigest(),
            json.loads(Path(args.app).read_text()),
            json.loads(Path(args.terminal).read_text()),
        )
        latest = Path(tmp) / "latest.json"
        latest.write_text(json.dumps(feed(args.version, release["body"], tmp), indent=2) + "\n")

        # Safe to rerun: a failed feed upload leaves a published release to retry from.
        if release["isDraft"]:
            gh("release", "edit", tag, "--draft=false", "--latest=false")
        gh("release", "upload", "desktop-latest", str(latest), "--clobber")
    print(f"Published {tag} and updated the desktop-latest feed.")


if __name__ == "__main__":
    main()
