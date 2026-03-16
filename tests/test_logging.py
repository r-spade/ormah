"""Tests for structured logging setup."""

from __future__ import annotations

import json
import logging

from ormah.logging_setup import _JSONFormatter, setup_logging


def test_json_formatter_basic():
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    line = formatter.format(record)
    obj = json.loads(line)

    assert obj["level"] == "INFO"
    assert obj["logger"] == "test.logger"
    assert obj["msg"] == "hello world"
    assert "ts" in obj


def test_json_formatter_extra_fields():
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="bg",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="job failed",
        args=(),
        exc_info=None,
    )
    record.job_id = "auto_linker"
    record.duration_ms = 123.4

    line = formatter.format(record)
    obj = json.loads(line)

    assert obj["job_id"] == "auto_linker"
    assert obj["duration_ms"] == 123.4


def test_json_formatter_exception():
    formatter = _JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="oh no",
        args=(),
        exc_info=exc_info,
    )
    line = formatter.format(record)
    obj = json.loads(line)

    assert "exception" in obj
    assert "boom" in obj["exception"]


def test_setup_logging_text():
    setup_logging(log_format="text")
    root = logging.getLogger()
    assert len(root.handlers) >= 1
    # Should be a StreamHandler with a plain Formatter
    handler = root.handlers[-1]
    assert not isinstance(handler.formatter, _JSONFormatter)


def test_setup_logging_json():
    setup_logging(log_format="json")
    root = logging.getLogger()
    assert len(root.handlers) >= 1
    handler = root.handlers[-1]
    assert isinstance(handler.formatter, _JSONFormatter)

    # Cleanup: switch back to text so other tests aren't affected
    setup_logging(log_format="text")
