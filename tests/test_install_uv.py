"""The standalone installer and desktop builds share one tested uv pin."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_pin_is_standalone_and_shared():
    pin = (ROOT / "desktop/uv-version").read_text().strip()
    installer = (ROOT / "install.sh").read_text()
    assert f'UV_VERSION="{pin}"' in installer
    assert "https://astral.sh/uv/$UV_VERSION/install.sh" in installer
    assert 'UV_MIN_VERSION="0.10.9"' in installer
    assert "--python 3.11" in installer
    assert '"${UV_FLAGS[@]}" 2>/dev/null' not in installer
    for name in ("ci", "desktop-release"):
        workflow = (ROOT / f".github/workflows/{name}.yml").read_text()
        yaml.safe_load(workflow)
        assert "bash desktop/scripts/build-sidecar.sh" in workflow
        assert "releases/latest/download" not in workflow
    sidecar = (ROOT / "desktop/scripts/build-sidecar.sh").read_text()
    assert 'cat "$SCRIPT_DIR/../uv-version"' in sidecar
    assert "sha256sum -c" in sidecar and "shasum -a 256 -c" in sidecar
    assert "already present" not in sidecar


@pytest.mark.parametrize(
    "existing,warning", [("0.9.0", True), ("0.10.9", False), ("0.13.6", False), ("0.14.0", False)]
)
def test_installer_reuses_uv_without_upgrading(tmp_path, existing, warning):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f"""#!/bin/bash
if [[ "$1" == --version ]]; then echo 'uv {existing}'; exit; fi
if [[ "$1 $2" == 'tool list' ]]; then exit; fi
printf '%s\\n' "$*" >> "$HOME/calls"
""")
    uv.chmod(0o755)
    ormah = bin_dir / "ormah"
    ormah.write_text("#!/bin/sh\necho 'ormah 0.17.0'\n")
    ormah.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "install.sh"), "--no-setup"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "SHELL": "/bin/bash",
        },
    )
    assert result.returncode == 0, result.stderr
    assert f"uv {existing}" in result.stdout
    assert ("below supported minimum" in result.stdout) == warning
    assert "--python 3.11" in (tmp_path / "calls").read_text()
    assert "self update" not in (tmp_path / "calls").read_text()


def test_sidecar_replaces_stale_binary_and_refuses_bad_checksum(tmp_path):
    import io
    import tarfile

    script_dir = tmp_path / "desktop/scripts"
    script_dir.mkdir(parents=True)
    (script_dir.parent / "uv-version").write_text("0.13.6\n")
    script = script_dir / "build-sidecar.sh"
    script.write_text((ROOT / "desktop/scripts/build-sidecar.sh").read_text())
    assets = tmp_path / "assets"
    assets.mkdir()
    archive = assets / "uv-x86_64-unknown-linux-gnu.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        content = b"#!/bin/sh\necho 'uv 0.13.6'\n"
        member = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    checksum = assets / (archive.name + ".sha256")
    checksum.write_text(f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text("""#!/bin/bash
set -eu
[[ "$2" == https://github.com/astral-sh/uv/releases/download/0.13.6/* ]]
cp "$ASSETS/${2##*/}" "$4"
""")
    curl.chmod(0o755)
    dest = script_dir.parent / "src-tauri/binaries/uv-x86_64-unknown-linux-gnu"
    dest.parent.mkdir(parents=True)
    dest.write_text("stale uv")
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "ASSETS": str(assets)}
    cmd = ["bash", str(script), "x86_64-unknown-linux-gnu"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "0.13.6" in subprocess.check_output([str(dest)], text=True)
    checksum.write_text(f"{'0' * 64}  {archive.name}\n")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAILED" in result.stdout
