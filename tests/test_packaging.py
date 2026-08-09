"""Tests for release packaging metadata and CLI fallbacks."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

from ormah.cli import _cmd_eval_whisper_run


class TestBuildMetadata:
    def test_wheel_packages_exclude_eval(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())

        wheel_packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert wheel_packages == ["src/ormah"]

    def test_sdist_only_includes_release_paths(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())

        included = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
        assert "eval" not in included
        assert "src/ormah" in included
        assert "ui" in included
        assert "install.sh" in included


class TestReleaseVersionVerification:
    def _write_release_files(self, root: Path, project_version: str, plugin_version: str):
        (root / "integrations/claude-plugin/.claude-plugin").mkdir(parents=True)
        (root / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "ormah"',
                    f'version = "{project_version}"',
                    "",
                ]
            )
        )
        (root / "integrations/claude-plugin/.claude-plugin/plugin.json").write_text(
            json.dumps({"name": "ormah", "version": plugin_version})
        )

    def _run_verifier(self, root: Path, version: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            [
                sys.executable,
                str(repo_root / ".github/release/verify_release_versions.py"),
                version,
                "--root",
                str(root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_accepts_matching_project_and_plugin_versions(self, tmp_path):
        self._write_release_files(tmp_path, project_version="1.2.3", plugin_version="1.2.3")

        result = self._run_verifier(tmp_path, "1.2.3")

        assert result.returncode == 0
        assert "Release version verified: 1.2.3" in result.stdout

    def test_rejects_requested_version_mismatch(self, tmp_path):
        self._write_release_files(tmp_path, project_version="1.2.3", plugin_version="1.2.3")

        result = self._run_verifier(tmp_path, "1.2.4")

        assert result.returncode == 1
        assert "does not match pyproject.toml version 1.2.3" in result.stderr

    def test_rejects_plugin_manifest_mismatch(self, tmp_path):
        self._write_release_files(tmp_path, project_version="1.2.3", plugin_version="1.2.2")

        result = self._run_verifier(tmp_path, "1.2.3")

        assert result.returncode == 1
        assert "Claude plugin manifest version 1.2.2 does not match" in result.stderr

    def test_rejects_leading_v_release_version(self, tmp_path):
        self._write_release_files(tmp_path, project_version="1.2.3", plugin_version="1.2.3")

        result = self._run_verifier(tmp_path, "v1.2.3")

        assert result.returncode == 1
        assert "should not include the leading 'v'" in result.stderr


class TestReleaseWorkflow:
    def test_manual_release_workflow_publishes_wheel_with_trusted_publishing(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/release.yml").read_text()

        assert "workflow_dispatch:" in workflow
        assert "RELEASE_ALLOWED_ACTOR: r-spade" in workflow
        assert "Only $RELEASE_ALLOWED_ACTOR can run the release workflow" in workflow
        assert "python .github/release/verify_release_versions.py" in workflow
        assert "uv run pytest" in workflow
        assert "npm run build" in workflow
        assert "uv build --wheel --out-dir dist" in workflow
        assert "pypa/gh-action-pypi-publish@release/v1" in workflow
        assert "id-token: write" in workflow
        assert "gh release create" in workflow

    def test_desktop_publish_commands_have_explicit_repository_context(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/desktop-release.yml").read_text()

        assert 'REPO="${{ github.repository }}"' in workflow
        assert 'gh release view "$GITHUB_REF_NAME" -R "$REPO"' in workflow


class TestEvalCliFallback:
    def test_eval_command_exits_cleanly_when_harness_is_missing(self, monkeypatch, capsys):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "eval.whisper.cli":
                raise ModuleNotFoundError("No module named 'eval'", name="eval")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(SystemExit) as exc:
            _cmd_eval_whisper_run(object())

        assert exc.value.code == 1
        assert "not installed in the published Ormah runtime" in capsys.readouterr().out
