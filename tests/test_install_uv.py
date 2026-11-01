"""The standalone installer and desktop builds share one uv pin."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_installer_and_builds_use_the_same_uv_pin():
    pin = (ROOT / "desktop/uv-version").read_text().strip()
    installer = (ROOT / "install.sh").read_text()
    assert f'UV_VERSION="{pin}"' in installer
    assert "https://astral.sh/uv/$UV_VERSION/install.sh" in installer
    assert 'cat "$SCRIPT_DIR/../uv-version"' in (ROOT / "desktop/scripts/build-sidecar.sh").read_text()
    for name in ("ci", "desktop-release"):
        workflow = (ROOT / f".github/workflows/{name}.yml").read_text()
        assert "bash desktop/scripts/build-sidecar.sh" in workflow
        assert "releases/latest/download" not in workflow
