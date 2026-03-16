"""Tests for shared space detection."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

from ormah.adapters.space_detect import detect_space_from_cwd, resolve_space


# --- detect_space_from_cwd ---


def test_detect_from_git_repo(tmp_path):
    repo = tmp_path / "my-project"
    repo.mkdir()
    git_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=str(repo) + "\n")
    with patch("os.getcwd", return_value=str(repo)), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_cwd() == "my-project"


def test_detect_from_git_subdirectory(tmp_path):
    repo = tmp_path / "my-project"
    sub = repo / "src" / "lib"
    sub.mkdir(parents=True)
    git_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=str(repo) + "\n")
    with patch("os.getcwd", return_value=str(sub)), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_cwd() == "my-project"


def test_detect_fallback_to_cwd_basename(tmp_path):
    d = tmp_path / "some-dir"
    d.mkdir()
    git_result = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="not a repo")
    with patch("os.getcwd", return_value=str(d)), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_cwd() == "some-dir"


def test_detect_returns_none_for_home():
    home = os.path.expanduser("~")
    with patch("os.getcwd", return_value=home):
        assert detect_space_from_cwd() is None


def test_detect_returns_none_for_root():
    with patch("os.getcwd", return_value="/"):
        assert detect_space_from_cwd() is None


def test_detect_handles_git_not_found():
    with patch("os.getcwd", return_value="/tmp/some-dir"), \
         patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_space_from_cwd() == "some-dir"


def test_detect_handles_git_timeout():
    with patch("os.getcwd", return_value="/tmp/some-dir"), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5)):
        assert detect_space_from_cwd() == "some-dir"


# --- resolve_space ---


def test_resolve_explicit_wins():
    with patch.dict(os.environ, {"ORMAH_SPACE": "from-env"}):
        assert resolve_space("explicit") == "explicit"


def test_resolve_env_var():
    with patch.dict(os.environ, {"ORMAH_SPACE": "from-env"}), \
         patch("ormah.adapters.space_detect.detect_space_from_cwd", return_value="from-cwd"):
        assert resolve_space() == "from-env"


def test_resolve_env_var_not_set_falls_to_cwd():
    with patch.dict(os.environ, {}, clear=True), \
         patch("ormah.adapters.space_detect.detect_space_from_cwd", return_value="from-cwd"):
        # Remove ORMAH_SPACE if present
        os.environ.pop("ORMAH_SPACE", None)
        assert resolve_space() == "from-cwd"


def test_resolve_all_none():
    with patch.dict(os.environ, {}, clear=True), \
         patch("ormah.adapters.space_detect.detect_space_from_cwd", return_value=None):
        os.environ.pop("ORMAH_SPACE", None)
        assert resolve_space() is None
