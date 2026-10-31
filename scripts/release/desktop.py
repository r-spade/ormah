#!/usr/bin/env python3
"""Owner-operated desktop candidate and promotion commands. No protection bypasses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REPO = os.environ.get("REPO", "r-spade/ormah")
OWNER = "r-spade"
APP_CHECKS = {
    "fetch",
    "clone",
    "display",
    "ssh_ready",
    "transfer",
    "artifact_checksum",
    "clean_baseline",
    "gui_session",
    "install",
    "server_ready",
    "server_owner",
    "ormah_version",
    "python_version",
    "models",
    "remember",
    "recall",
    "reboot",
    "boot_before",
    "reboot_completed",
    "ssh_after_reboot",
    "server_returns",
    "server_owner_after_reboot",
    "recall_after_reboot",
    "app_process",
    "app_process_after_reboot",
    "desktop_version",
    "uv_version",
}
MANUAL_CHECKS = {
    "gatekeeper_browser_first_open",
    "signed_in_ai_tool",
    "small_display",
    "network_fault",
    "linux_appimage",
    "linux_deb",
}


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def gh(*args):
    return run("gh", *args)


def release(tag):
    return json.loads(
        gh(
            "release",
            "view",
            tag,
            "-R",
            REPO,
            "--json",
            "isDraft,isPrerelease,tagName,body,assets,targetCommitish",
        )
    )


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def versions():
    python = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    conf = json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text())["version"]
    cargo = tomllib.loads((ROOT / "desktop/src-tauri/Cargo.toml").read_text())
    lock = tomllib.loads((ROOT / "desktop/src-tauri/Cargo.lock").read_text())
    locked = next(p["version"] for p in lock["package"] if p["name"] == "ormah-desktop")
    if conf != cargo["package"]["version"] or conf != locked:
        raise ValueError("Desktop version must match tauri.conf.json, Cargo.toml and Cargo.lock")
    for version in (python, conf):
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError(f"Expected a stable release version: {version}")
    return python, conf


def prepare_check():
    python, desktop = versions()
    # Listing errors fail closed; a failed `view` must not be mistaken for absence.
    releases = json.loads(gh("release", "list", "-R", REPO, "--limit", "1000", "--json", "tagName"))
    tag = f"desktop-v{desktop}"
    if any(r["tagName"] == tag for r in releases):
        raise ValueError(f"{tag} already exists; use a new reviewed desktop version")
    return python, desktop


def preflight():
    head = run("git", "rev-parse", "HEAD")
    expected = os.environ.get("EXPECTED_SOURCE_SHA", "")
    if expected and expected != head:
        raise ValueError("Dispatched ref moved; checkout does not match requested source SHA")
    if run("git", "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Use a clean checkout of the reviewed release commit")
    main_sha = gh("api", f"repos/{REPO}/commits/main", "--jq", ".sha")
    if subprocess.run(["git", "merge-base", "--is-ancestor", head, main_sha]).returncode:
        raise ValueError("Candidate source must already be merged into protected main")
    python, desktop = prepare_check()
    published = release(f"v{python}")
    if published["isDraft"]:
        raise ValueError("Python release must be published first")
    source = gh("api", f"repos/{REPO}/commits/v{python}", "--jq", ".sha")
    if subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"]).returncode:
        raise ValueError("Build must contain the published Python release commit")
    return python, desktop


def artifact_paths(directory):
    paths = {}
    for path in Path(directory).rglob("*"):
        if path.is_file():
            if path.name in paths:
                raise ValueError(f"Duplicate artifact name: {path.name}")
            paths[path.name] = path
    for suffix in (".dmg", ".app.tar.gz", ".AppImage", ".deb"):
        matches = [p for name, p in paths.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {suffix} artifact")
        if suffix != ".dmg":
            sig = paths.get(matches[0].name + ".sig")
            if sig is None or not sig.read_text().strip():
                raise ValueError(f"Missing updater signature for {matches[0].name}")
    if any(p.stat().st_size == 0 for p in paths.values()):
        raise ValueError("Empty artifact")
    return paths


def create_draft(directory, source_sha):
    python, desktop = preflight()
    if source_sha != run("git", "rev-parse", "HEAD"):
        raise ValueError("Source SHA does not match checkout")
    paths = artifact_paths(directory)
    manifest = {
        "schema": 1,
        "source_sha": source_sha,
        "ormah_version": python,
        "desktop_version": desktop,
        "uv_version": (ROOT / "desktop/uv-version").read_text().strip(),
        "assets": {name: digest(path) for name, path in paths.items()},
    }
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "candidate.json"
        file.write_text(json.dumps(manifest, indent=2) + "\n")
        gh(
            "release",
            "create",
            f"desktop-v{desktop}",
            "-R",
            REPO,
            "--draft",
            "--target",
            source_sha,
            "--title",
            f"Ormah Desktop {desktop} (Ormah {python})",
            "--notes",
            "Candidate: awaiting clean-machine acceptance and owner promotion. "
            "Do not replace candidate assets; use a new desktop version after a rebuild.",
            str(file),
            *map(str, paths.values()),
        )


def verify_evidence(directory, acceptance, manual, tag):
    directory = Path(directory)
    manifest = json.loads((directory / "candidate.json").read_text())
    if manifest.get("schema") != 1 or tag != f"desktop-v{manifest['desktop_version']}":
        raise ValueError("Candidate schema/tag mismatch")
    paths = artifact_paths(directory)
    # Evidence files may also be attached by an earlier failed promotion attempt.
    assets = manifest["assets"]
    for name, expected in assets.items():
        if Path(name).name != name or name not in paths or digest(paths[name]) != expected:
            raise ValueError(f"Candidate checksum mismatch: {name}")
    for name in paths:
        if (
            name.endswith((".dmg", ".app.tar.gz", ".AppImage", ".deb", ".sig"))
            and name not in assets
        ):
            raise ValueError(f"Unrecorded release artifact: {name}")
    app = json.loads(Path(acceptance).read_text())
    dmg = next(name for name in assets if name.endswith(".dmg"))
    if (
        app.get("schema") != 1
        or app.get("mode") != "app"
        or app.get("status") != "passed"
        or app.get("artifact_sha256") != assets[dmg]
        or app.get("expected_version") != manifest["ormah_version"]
        or app.get("expected_desktop_version") != manifest["desktop_version"]
        or app.get("expected_uv_version") != manifest["uv_version"]
    ):
        raise ValueError("Missing passing app acceptance for this exact candidate")
    if not APP_CHECKS <= app.get("checks", {}).keys() or any(
        app["checks"][check].get("passed") is not True for check in APP_CHECKS
    ):
        raise ValueError("Incomplete or failing app checks")
    attestation = json.loads(Path(manual).read_text())
    if (
        attestation.get("schema") != 1
        or attestation.get("tag") != tag
        or attestation.get("assets") != assets
        or not MANUAL_CHECKS <= attestation.get("checks", {}).keys()
        or any(attestation["checks"][c] is not True for c in MANUAL_CHECKS)
    ):
        raise ValueError("Manual macOS/Linux checks must pass for these exact asset hashes")
    return manifest, paths


def feed(manifest, paths, notes):
    base = f"https://github.com/{REPO}/releases/download/desktop-v{manifest['desktop_version']}"

    def entry(suffix):
        path = next(p for n, p in paths.items() if n.endswith(suffix))
        return {
            "signature": paths[path.name + ".sig"].read_text().strip(),
            "url": f"{base}/{path.name}",
        }

    return {
        "version": manifest["desktop_version"],
        "notes": notes,
        "pub_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platforms": {
            "linux-x86_64-deb": entry(".deb"),
            "linux-x86_64": entry(".AppImage"),
            "darwin-aarch64": entry(".app.tar.gz"),
        },
    }


def promote(tag, acceptance, manual):
    actor = gh("api", "user", "--jq", ".login")
    if actor != OWNER:
        raise ValueError(f"Only {OWNER} can promote (authenticated as {actor})")
    with tempfile.TemporaryDirectory() as tmp:
        before = release(tag)
        gh("release", "download", tag, "-R", REPO, "--dir", tmp)
        manifest, paths = verify_evidence(tmp, acceptance, manual, tag)
        latest = Path(tmp) / "latest.json"
        latest.write_text(json.dumps(feed(manifest, paths, before["body"]), indent=2) + "\n")
        # Refuse stale or out-of-order promotions, including on retry.
        releases = json.loads(
            gh("release", "list", "-R", REPO, "--limit", "1000", "--json", "tagName,isDraft")
        )
        version = tuple(map(int, manifest["desktop_version"].split(".")))
        for item in releases:
            match = re.fullmatch(r"desktop-v(\d+)\.(\d+)\.(\d+)", item["tagName"])
            if match and not item["isDraft"] and tuple(map(int, match.groups())) > version:
                raise ValueError("A newer desktop is already published")
        if release(tag)["assets"] != before["assets"]:
            raise ValueError("Release assets changed during verification")
        has_feed = any(r["tagName"] == "desktop-latest" for r in releases)
        if has_feed:
            rolling = release("desktop-latest")
            if rolling["isDraft"] or not rolling["isPrerelease"]:
                raise ValueError("Rolling feed must be a published prerelease; repair manually")
        record = Path(tmp) / "promotion-record.json"
        record.write_text(
            json.dumps(
                {
                    "actor": actor,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "tag": tag,
                    "candidate": manifest,
                    "acceptance": json.loads(Path(acceptance).read_text()),
                    "manual": json.loads(Path(manual).read_text()),
                },
                indent=2,
            )
            + "\n"
        )
        gh("release", "upload", tag, "-R", REPO, str(record), "--clobber")
        gh("release", "edit", tag, "-R", REPO, "--draft=false", "--latest=false")
        # Only promotion can touch the rolling feed. Publishing happens first so URLs resolve.
        if has_feed:
            gh("release", "upload", "desktop-latest", str(latest), "--clobber", "-R", REPO)
        else:
            gh(
                "release",
                "create",
                "desktop-latest",
                str(latest),
                "-R",
                REPO,
                "--prerelease",
                "--title",
                "Desktop updater feed",
                "--notes",
                "Rolling desktop updater feed.",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidate", help="Dispatch draft build from a reviewed Python tag")
    candidate.add_argument("--ormah", required=True)
    candidate.add_argument("--ref", help="Reviewed ref; defaults to the Python release tag")
    sub.add_parser("preflight")
    sub.add_parser("prepare-check")
    draft = sub.add_parser("draft")
    draft.add_argument("--artifacts", required=True)
    draft.add_argument("--source-sha", required=True)
    promotion = sub.add_parser("promote", help="Owner attests checks, publishes and updates feed")
    promotion.add_argument("--tag", required=True)
    promotion.add_argument("--acceptance", required=True)
    promotion.add_argument("--manual", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "candidate":
        python, _ = preflight()
        if args.ormah != python:
            raise ValueError("Checkout must match --ormah")
        subprocess.run(["python3", str(ROOT / "scripts/release/wait-pypi.py"), python], check=True)
        ref = args.ref or f"v{python}"
        if gh("api", f"repos/{REPO}/commits/{ref}", "--jq", ".sha") != run(
            "git", "rev-parse", "HEAD"
        ):
            raise ValueError("Checkout must match the dispatched ref")
        gh(
            "workflow",
            "run",
            "desktop-release.yml",
            "-R",
            REPO,
            "--ref",
            ref,
            "-f",
            f"source_sha={run('git', 'rev-parse', 'HEAD')}",
        )
    elif args.command == "preflight":
        preflight()
    elif args.command == "prepare-check":
        prepare_check()
    elif args.command == "draft":
        create_draft(args.artifacts, args.source_sha)
    else:
        promote(args.tag, args.acceptance, args.manual)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Desktop release refused: {exc}") from exc
