"""Tests for ormah console output formatting."""

from __future__ import annotations

import sys
import time
from unittest.mock import patch

import pytest

from ormah.console import (
    Spinner,
    _reset_color_cache,
    fail,
    info,
    ok,
    step,
    warn,
)


@pytest.fixture(autouse=True)
def _reset_color():
    """Reset color detection cache before each test."""
    _reset_color_cache()
    yield
    _reset_color_cache()


# ── Output function tests ───────────────────────────────────────────────────


class TestOutputFunctions:
    def test_ok_prefix_no_color(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        ok("done")
        assert capsys.readouterr().out == "[ok] done\n"

    def test_info_prefix_no_color(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        info("working")
        assert capsys.readouterr().out == "[..] working\n"

    def test_warn_prefix_no_color(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        warn("careful")
        assert capsys.readouterr().out == "[!!] careful\n"

    def test_step_prefix_no_color(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        step("Phase 1")
        assert capsys.readouterr().out == "\n==> Phase 1\n"

    def test_fail_goes_to_stderr(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        fail("broken")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "[xx] broken\n"

    def test_ok_with_color(self, capsys, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        ok("done")
        out = capsys.readouterr().out
        assert "[ok]" in out
        assert "done" in out
        assert "\033[32m" in out  # green

    def test_warn_with_color(self, capsys, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        warn("careful")
        out = capsys.readouterr().out
        assert "\033[33m" in out  # yellow

    def test_step_with_color(self, capsys, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        step("Build")
        out = capsys.readouterr().out
        assert "\033[1m" in out  # bold
        assert "==>" in out


# ── Color detection tests ────────────────────────────────────────────────────


class TestColorDetection:
    def test_no_color_env_disables_color(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        ok("test")
        out = capsys.readouterr().out
        assert "\033[" not in out

    def test_no_color_takes_precedence_over_force(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        ok("test")
        out = capsys.readouterr().out
        assert "\033[" not in out

    def test_non_tty_no_color(self, capsys, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        # When stdout is not a TTY (e.g. pytest capture), no color
        ok("test")
        out = capsys.readouterr().out
        assert out == "[ok] test\n"


# ── Spinner tests ────────────────────────────────────────────────────────────


class TestSpinner:
    def test_non_tty_prints_info_lines(self, capsys):
        """In non-TTY mode, Spinner prints [..] lines instead of animating."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            with Spinner("Loading") as sp:
                sp.update("Phase 2")
                sp.succeed("Done")

        out = capsys.readouterr().out
        assert "[..] Loading\n" in out
        assert "[..] Phase 2\n" in out
        assert "[ok] Done" in out

    def test_non_tty_fail(self, capsys):
        with patch.object(sys.stdout, "isatty", return_value=False):
            with Spinner("Loading") as sp:
                sp.fail("Broke")

        out = capsys.readouterr().out
        assert "[..] Loading\n" in out
        assert "[!!] Broke" in out

    def test_non_tty_no_duplicate_on_same_message(self, capsys):
        """Updating with the same message shouldn't print again."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            with Spinner("Loading") as sp:
                sp.update("Loading")  # same message, no new line
                sp.succeed("Done")

        out = capsys.readouterr().out
        assert out.count("[..] Loading") == 1

    def test_succeed_shows_elapsed_for_long_waits(self, capsys):
        with patch.object(sys.stdout, "isatty", return_value=False):
            with Spinner("Loading") as sp:
                # Fake the start time to simulate a long wait
                sp._start_time = time.monotonic() - 10.0
                sp.succeed("Done")

        out = capsys.readouterr().out
        assert "[ok] Done" in out
        assert "s" in out  # elapsed seconds shown

    def test_context_manager_cleanup(self):
        """Spinner cleans up even on exception."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            try:
                with Spinner("Loading"):
                    raise ValueError("boom")
            except ValueError:
                pass
            # No assertion — just verify no crash/hang
