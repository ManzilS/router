"""Tests for the structured logging module."""

import json
import logging

import pytest

from src.utils.logging import (
    DevFormatter,
    StructuredFormatter,
    get_request_elapsed_ms,
    get_request_id,
    request_id_var,
    request_start_var,
    setup_logging,
)


def test_request_id_default():
    assert get_request_id() == "-"


def test_request_id_set():
    token = request_id_var.set("abc123")
    try:
        assert get_request_id() == "abc123"
    finally:
        request_id_var.reset(token)


def test_elapsed_ms_zero_when_no_start():
    assert get_request_elapsed_ms() == 0.0


def test_structured_formatter_outputs_json():
    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    output = fmt.format(record)
    data = json.loads(output)
    assert data["msg"] == "hello world"
    assert data["level"] == "INFO"
    assert "request_id" in data


def test_dev_formatter_outputs_human_readable():
    fmt = DevFormatter()
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="", lineno=0,
        msg="something happened", args=(), exc_info=None,
    )
    output = fmt.format(record)
    assert "WARNING" in output
    assert "something happened" in output


def test_setup_logging_dev_mode():
    setup_logging(log_level="DEBUG", dev_mode=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) > 0


def test_setup_logging_prod_mode():
    setup_logging(log_level="INFO", dev_mode=False)
    root = logging.getLogger()
    assert root.level == logging.INFO
