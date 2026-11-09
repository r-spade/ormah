"""Tests for shared space detection."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

from ormah.adapters.space_detect import detect_space_from_cwd, detect_space_from_dir, resolve_space


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


def test_detect_returns_none_for_symlinked_home(tmp_path):
    (tmp_path / "real" / "home").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    home = str(tmp_path / "link" / "home")
    with patch.dict(os.environ, {"HOME": home}):
        assert detect_space_from_dir(home) is None


def test_detect_returns_none_for_parent_of_home(tmp_path):
    (tmp_path / "Users" / "someone").mkdir(parents=True)
    git_result = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="not a repo")
    with patch.dict(os.environ, {"HOME": str(tmp_path / "Users" / "someone")}), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_dir(str(tmp_path / "Users")) is None


def test_detect_keeps_sibling_that_shares_home_prefix(tmp_path):
    (tmp_path / "someone").mkdir()
    (tmp_path / "someone-else").mkdir()
    git_result = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="not a repo")
    with patch.dict(os.environ, {"HOME": str(tmp_path / "someone")}), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_dir(str(tmp_path / "someone-else")) == "someone-else"


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


def test_detect_handles_not_a_directory():
    with patch("subprocess.run", side_effect=NotADirectoryError):
        assert detect_space_from_dir("/tmp/some-file") == "some-file"


def test_detect_handles_permission_error():
    with patch("subprocess.run", side_effect=PermissionError):
        assert detect_space_from_dir("/tmp/locked-dir") == "locked-dir"


def test_detect_falls_back_when_git_toplevel_is_home(tmp_path):
    home = tmp_path / "home"
    downloads = home / "Downloads"
    downloads.mkdir(parents=True)
    git_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=str(home) + "\n")
    with patch.dict(os.environ, {"HOME": str(home)}), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_dir(str(downloads)) == "Downloads"


def test_detect_keeps_nested_repo_under_git_home(tmp_path):
    home = tmp_path / "home"
    sub = home / "code" / "my-project" / "src"
    sub.mkdir(parents=True)
    git_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=str(home / "code" / "my-project") + "\n"
    )
    with patch.dict(os.environ, {"HOME": str(home)}), \
         patch("subprocess.run", return_value=git_result):
        assert detect_space_from_dir(str(sub)) == "my-project"


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
