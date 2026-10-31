"""Release safety tests run without publishing or touching host Ormah services."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


train = module("desktop_release", "scripts/release/desktop.py")
pypi = module("wait_pypi", "scripts/release/wait-pypi.py")


def test_workflows_only_draft_and_keep_propagation_guard():
    workflow = (ROOT / ".github/workflows/desktop-release.yml").read_text()
    assert "desktop-v*" in workflow
    assert "refs/tags/desktop-v" in workflow
    assert "desktop.py draft" in workflow
    assert "desktop.py preflight" in workflow
    assert "desktop-latest" not in workflow
    assert "desktop.py promote" not in workflow
    assert "notarytool submit" in workflow and "stapler staple" in workflow
    assert "*.deb*" in workflow and "*.AppImage*" in workflow
    for path in ("desktop-release", "release"):
        text = (ROOT / f".github/workflows/{path}.yml").read_text()
        assert "scripts/release/wait-pypi.py" in text
    assert "Desktop candidate required" in (ROOT / ".github/workflows/release.yml").read_text()


@pytest.fixture
def evidence(tmp_path):
    artifacts = tmp_path / "assets"
    artifacts.mkdir()
    names = [
        "Ormah.dmg",
        "Ormah.app.tar.gz",
        "Ormah.app.tar.gz.sig",
        "Ormah.AppImage",
        "Ormah.AppImage.sig",
        "Ormah.deb",
        "Ormah.deb.sig",
    ]
    for name in names:
        (artifacts / name).write_text(name)
    hashes = {name: train.digest(artifacts / name) for name in names}
    manifest = {
        "schema": 1,
        "desktop_version": "0.4.0",
        "ormah_version": "0.17.0",
        "source_sha": "a" * 40,
        "uv_version": "0.13.6",
        "assets": hashes,
    }
    (artifacts / "candidate.json").write_text(json.dumps(manifest))
    app = {
        "schema": 1,
        "mode": "app",
        "status": "passed",
        "expected_version": "0.17.0",
        "expected_desktop_version": "0.4.0",
        "expected_uv_version": "0.13.6",
        "artifact_sha256": hashes["Ormah.dmg"],
        "checks": {name: {"passed": True} for name in train.APP_CHECKS},
    }
    manual = {
        "schema": 1,
        "tag": "desktop-v0.4.0",
        "assets": hashes,
        "checks": dict.fromkeys(train.MANUAL_CHECKS, True),
    }
    app_file, manual_file = tmp_path / "app.json", tmp_path / "manual.json"
    app_file.write_text(json.dumps(app))
    manual_file.write_text(json.dumps(manual))
    return artifacts, app_file, manual_file


def test_gate_accepts_exact_evidence_and_generates_all_platforms(evidence):
    manifest, paths = train.verify_evidence(*evidence, "desktop-v0.4.0")
    feed = train.feed(manifest, paths, "Notes")
    assert set(feed["platforms"]) == {"darwin-aarch64", "linux-x86_64", "linux-x86_64-deb"}
    assert feed["platforms"]["linux-x86_64-deb"]["signature"] == "Ormah.deb.sig"


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_sha256", "bad"),
        ("status", "failed"),
        ("mode", "terminal"),
        ("expected_version", "0.16.0"),
        ("expected_desktop_version", "0.3.28"),
        ("expected_uv_version", "0.12.24"),
        ("checks", {}),
        ("schema", 0),
    ],
)
def test_gate_refuses_wrong_app_evidence(evidence, field, value):
    data = json.loads(evidence[1].read_text())
    data[field] = value
    evidence[1].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        train.verify_evidence(*evidence, "desktop-v0.4.0")


@pytest.mark.parametrize("check", sorted(train.APP_CHECKS))
def test_every_automatic_check_is_required(evidence, check):
    data = json.loads(evidence[1].read_text())
    data["checks"][check]["passed"] = False
    evidence[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checks"):
        train.verify_evidence(*evidence, "desktop-v0.4.0")


@pytest.mark.parametrize("check", sorted(train.MANUAL_CHECKS))
def test_every_manual_check_is_required(evidence, check):
    data = json.loads(evidence[2].read_text())
    data["checks"][check] = False
    evidence[2].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Manual"):
        train.verify_evidence(*evidence, "desktop-v0.4.0")


@pytest.mark.parametrize(
    "name", ["Ormah.dmg", "Ormah.app.tar.gz", "Ormah.AppImage", "Ormah.deb", "Ormah.deb.sig"]
)
def test_changed_artifacts_cannot_be_promoted(evidence, name):
    (evidence[0] / name).write_text("rebuilt bytes")
    with pytest.raises(ValueError, match="checksum"):
        train.verify_evidence(*evidence, "desktop-v0.4.0")


def test_unrecorded_artifact_is_refused(evidence):
    (evidence[0] / "unexpected.sig").write_text("new")
    with pytest.raises(ValueError, match="Unrecorded"):
        train.verify_evidence(*evidence, "desktop-v0.4.0")


def test_missing_signature_is_refused(evidence):
    (evidence[0] / "Ormah.deb.sig").unlink()
    with pytest.raises(ValueError, match="signature"):
        train.verify_evidence(*evidence, "desktop-v0.4.0")


def test_nonowner_cannot_promote(monkeypatch):
    monkeypatch.setattr(train, "gh", lambda *args: "someone-else")
    with pytest.raises(ValueError, match="Only r-spade"):
        train.promote("desktop-v0.4.0", "missing", "missing")


def test_promotion_verifies_before_mutating(evidence, monkeypatch):
    calls = []

    def gh(*args):
        calls.append(args)
        if args[:2] == ("api", "user"):
            return "r-spade"
        if args[:2] == ("release", "view"):
            return json.dumps(
                {"isDraft": True, "isPrerelease": True, "assets": [], "body": "notes"}
            )
        if args[:2] == ("release", "download"):
            import shutil

            shutil.copytree(evidence[0], args[-1], dirs_exist_ok=True)
            return ""
        raise AssertionError(f"Unexpected mutation: {args}")

    data = json.loads(evidence[1].read_text())
    data["status"] = "failed"
    evidence[1].write_text(json.dumps(data))
    monkeypatch.setattr(train, "gh", gh)
    with pytest.raises(ValueError):
        train.promote("desktop-v0.4.0", evidence[1], evidence[2])
    assert all(call[1] not in {"upload", "edit", "create"} for call in calls)


def test_pypi_retries_stale_and_network_failures(monkeypatch):
    from io import BytesIO

    responses = [
        OSError("network"),
        {"info": {"version": "0.16.0"}},
        {"info": {"version": "0.17.0"}, "releases": {"0.17.0": [{"packagetype": "bdist_wheel"}]}},
    ]
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return BytesIO(json.dumps(item).encode())

    monkeypatch.setattr(pypi.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pypi.time, "sleep", lambda seconds: None)
    pypi.wait_for_version("0.17.0")
    assert len(requests) == 3
    assert all(r.headers["Cache-control"] == "no-cache" for r in requests)
    assert len({r.full_url for r in requests}) == 3


def test_pypi_deadline_fails_closed(monkeypatch):
    monkeypatch.setattr(
        pypi.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    with pytest.raises(SystemExit, match="did not confirm"):
        pypi.wait_for_version("0.17.0", timeout=0)


STUB = r"""#!/usr/bin/env python3
import json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(os.environ['STUB_ROOT'])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
body = sys.stdin.read() if name == 'ssh' and args[-1] == 'bash -se' else ''
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps({'tool': name, 'args': args, 'body': body,
                       'askpass': os.environ.get('SSH_ASKPASS'),
                       'password_file': os.environ.get('ORMAH_ACCEPTANCE_PASSWORD_FILE')}) + '\n')
if name == 'tart':
    if args[0] == 'ip': print('192.168.64.2')
    elif args[0] == 'clone' and os.environ.get('FAIL_CLONE'): sys.exit(1)
    elif args[0] == 'run':
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        count = root / 'runs'
        n = int(count.read_text()) + 1 if count.exists() else 1
        count.write_text(str(n))
        while not (root / ('reboot' if n == 1 else 'stop')).exists(): time.sleep(.02)
    elif args[0] == 'stop': (root / 'stop').touch()
elif name == 'ssh':
    if args[-1] == 'sysctl -n kern.boottime':
        print('boot-2' if (root / 'reboot').exists() else 'boot-1')
    if body:
        syntax = subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True)
        if syntax.returncode: print(syntax.stderr); sys.exit(2)
        if os.environ.get('FAIL_BODY') and os.environ['FAIL_BODY'] in body: sys.exit(1)
        if 'sudo -n /sbin/reboot' in body: (root / 'reboot').touch()
    elif args[-1].startswith('cat >'): sys.stdin.read()
"""


@pytest.mark.parametrize(
    "mode,fail,keep",
    [
        ("app", "", False),
        ("terminal", "", False),
        ("app", "", True),
        ("app", "open -a", False),
        ("terminal", "setup --ci", False),
        ("app", "TextEmbedding", False),
        ("app", "pgrep", False),
    ],
)
def test_vm_lifecycle_with_stubbed_tart_and_ssh(tmp_path, mode, fail, keep, request):
    def stop_stubs():
        (tmp_path / "reboot").touch()
        (tmp_path / "stop").touch()

    request.addfinalizer(stop_stubs)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("tart", "ssh"):
        file = bin_dir / name
        file.write_text(STUB)
        file.chmod(0o755)
    artifact = tmp_path / "input"
    artifact.write_text("exact test artifact")
    output = tmp_path / "result"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "STUB_ROOT": str(tmp_path),
        "FAIL_BODY": fail,
    }
    args = [
        "bash",
        str(ROOT / "scripts/acceptance/macos-vm.sh"),
        mode,
        "--dmg" if mode == "app" else "--installer",
        str(artifact),
        "--output",
        str(output),
    ]
    if keep:
        args.append("--keep")
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == (not fail), result.stdout + result.stderr
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == ("failed" if fail else "passed")
    assert report["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    tart_calls = [c["args"] for c in calls if c["tool"] == "tart"]
    assert tart_calls[0] == ["clone", "ormah-base", report["vm"]]
    assert ["set", report["vm"], "--display", "1920x1200"] in tart_calls
    assert (["delete", report["vm"]] in tart_calls) == (not fail and not keep)
    assert all(c[1] == report["vm"] for c in tart_calls if c[0] in {"stop", "delete"})
    ssh_calls = [c for c in calls if c["tool"] == "ssh"]
    assert all("admin@192.168.64.2" in c["args"] for c in ssh_calls)
    assert all("admin/admin" not in " ".join(c["args"]) for c in ssh_calls)
    assert all(
        "-F" in c["args"] and "StrictHostKeyChecking=accept-new" in c["args"] for c in ssh_calls
    )
    assert not Path(ssh_calls[0]["password_file"]).exists()
    if not fail:
        assert all(v["passed"] for v in report["checks"].values())
        if mode == "app":
            assert train.APP_CHECKS <= report["checks"].keys()
        else:
            assert report["checks"]["login_shell_after_reboot"]["passed"]
    # Terminate only stub Tart processes intentionally retained by --keep/failure.
    (tmp_path / "reboot").touch()
    (tmp_path / "stop").touch()


def test_prepare_requires_an_unused_desktop_version(monkeypatch):
    monkeypatch.setattr(train, "versions", lambda: ("0.17.0", "0.3.28"))
    monkeypatch.setattr(train, "gh", lambda *args: '[{"tagName":"desktop-v0.3.28"}]')
    with pytest.raises(ValueError, match="new reviewed desktop version"):
        train.prepare_check()
    monkeypatch.setattr(train, "gh", lambda *args: "[]")
    assert train.prepare_check() == ("0.17.0", "0.3.28")


def test_prepare_fails_closed_on_github_failure(monkeypatch):
    monkeypatch.setattr(train, "versions", lambda: ("0.17.0", "0.3.29"))

    def fail(*args):
        raise subprocess.CalledProcessError(1, ["gh", *args])

    monkeypatch.setattr(train, "gh", fail)
    with pytest.raises(subprocess.CalledProcessError):
        train.prepare_check()


def test_preflight_rejects_unpublished_python(monkeypatch):
    monkeypatch.setattr(train, "run", lambda *args: "" if args[1] == "status" else "a" * 40)
    monkeypatch.setattr(train, "gh", lambda *args: "a" * 40)
    monkeypatch.setattr(
        train.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0)
    )
    monkeypatch.setattr(train, "prepare_check", lambda: ("0.17.0", "0.3.29"))
    monkeypatch.setattr(train, "release", lambda tag: {"isDraft": True})
    with pytest.raises(ValueError, match="Python release"):
        train.preflight()


def test_versions_follow_existing_sync_convention():
    assert train.versions() == (
        train.tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
        json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text())["version"],
    )


def test_draft_never_publishes_or_updates_feed(evidence, monkeypatch):
    monkeypatch.setattr(train, "preflight", lambda: ("0.17.0", "0.4.0"))
    monkeypatch.setattr(train, "run", lambda *args: "a" * 40)
    (evidence[0] / "candidate.json").unlink()
    calls = []

    def gh(*args):
        calls.append(args)
        manifest_path = next(Path(a) for a in args if str(a).endswith("/candidate.json"))
        manifest = json.loads(manifest_path.read_text())
        assert manifest["assets"]["Ormah.dmg"] == train.digest(evidence[0] / "Ormah.dmg")
        assert "Ormah.deb.sig" in manifest["assets"]
        return ""

    monkeypatch.setattr(train, "gh", gh)
    train.create_draft(evidence[0], "a" * 40)
    assert len(calls) == 1
    assert calls[0][:2] == ("release", "create")
    assert "--draft" in calls[0]
    assert "desktop-latest" not in calls[0]


@pytest.mark.parametrize("published", [False, True])
def test_promotion_order_and_recovery(evidence, monkeypatch, published):
    import shutil

    calls = []

    def gh(*args):
        calls.append(args)
        if args[:2] == ("api", "user"):
            return "r-spade"
        if args[:2] == ("release", "view"):
            return json.dumps(
                {
                    "isDraft": not published and args[2] != "desktop-latest",
                    "isPrerelease": True,
                    "assets": [],
                    "body": "notes",
                }
            )
        if args[:2] == ("release", "download"):
            shutil.copytree(evidence[0], args[-1], dirs_exist_ok=True)
        if args[:2] == ("release", "list"):
            return '[{"tagName":"desktop-latest","isDraft":false}]'
        if args[:2] == ("release", "upload"):
            name = "promotion-record.json" if args[2] != "desktop-latest" else "latest.json"
            file = next(Path(a) for a in args if str(a).endswith("/" + name))
            data = json.loads(file.read_text())
            if name == "promotion-record.json":
                assert data["actor"] == "r-spade"
                assert data["acceptance"]["artifact_sha256"]
        return ""

    monkeypatch.setattr(train, "gh", gh)
    train.promote("desktop-v0.4.0", evidence[1], evidence[2])
    writes = [c for c in calls if c[:2] in [("release", "upload"), ("release", "edit")]]
    assert writes[0][:3] == ("release", "upload", "desktop-v0.4.0")
    assert writes[1][:3] == ("release", "edit", "desktop-v0.4.0")
    assert "--draft=false" in writes[1]
    assert writes[2][:3] == ("release", "upload", "desktop-latest")


def test_preflight_rejects_moved_dispatch_ref(monkeypatch):
    monkeypatch.setenv("EXPECTED_SOURCE_SHA", "b" * 40)
    monkeypatch.setattr(train, "run", lambda *args: "a" * 40)
    with pytest.raises(ValueError, match="ref moved"):
        train.preflight()


def test_preflight_rejects_unreviewed_source(monkeypatch):
    monkeypatch.setattr(train, "run", lambda *args: "" if args[1] == "status" else "a" * 40)
    monkeypatch.setattr(train, "gh", lambda *args: "b" * 40)
    monkeypatch.setattr(
        train.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 1)
    )
    with pytest.raises(ValueError, match="protected main"):
        train.preflight()
