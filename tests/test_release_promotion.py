"""Desktop releases reach users only through the acceptance-checked promote script."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("promote", ROOT / "scripts/release/promote-desktop.py")
promote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote)

APP = {"mode": "app", "status": "passed", "artifact_sha256": "abc", "ormah_version": "1.0.0"}
TERMINAL = {"mode": "terminal", "status": "passed", "ormah_version": "1.0.0"}


def test_accepts_passing_results_for_the_exact_dmg():
    promote.check_results("abc", APP, TERMINAL)


@pytest.mark.parametrize(
    "app,terminal",
    [
        ({**APP, "status": "failed"}, TERMINAL),
        ({**APP, "artifact_sha256": "other"}, TERMINAL),
        (TERMINAL, TERMINAL),
        (APP, {**TERMINAL, "status": "failed"}),
        (APP, {**TERMINAL, "ormah_version": "0.1.0"}),
        (APP, APP),
    ],
)
def test_refuses_failed_or_mismatched_results(app, terminal):
    with pytest.raises(SystemExit):
        promote.check_results("abc", app, terminal)


def test_refuses_to_move_the_feed_to_an_older_version():
    tags = [
        {"tagName": "desktop-latest", "isDraft": False},
        {"tagName": "desktop-v0.3.30", "isDraft": False},
        {"tagName": "desktop-v0.3.31", "isDraft": True},
    ]
    promote.check_not_older("0.3.30", tags)
    promote.check_not_older("0.3.31", tags)
    with pytest.raises(SystemExit):
        promote.check_not_older("0.3.29", tags)


def test_release_workflow_cannot_publish_or_touch_the_feed():
    workflow = (ROOT / ".github/workflows/desktop-release.yml").read_text()
    assert "--draft" in workflow
    assert "desktop-latest" not in workflow


def test_release_stops_early_when_the_desktop_version_is_used():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    assert workflow.index("Require an unused desktop version") < workflow.index("Run tests")
