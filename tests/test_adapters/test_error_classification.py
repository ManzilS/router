"""Tests for adapter HTTP error classification."""

import pytest

from src.adapters.openai_ext import _classify_http_error
from src.utils.errors import (
    AdapterAuthError,
    AdapterError,
    AdapterRateLimitError,
)


class FakeResponse:
    def __init__(self, status_code: int, text: str = "error body"):
        self.status_code = status_code
        self.text = text


class FakeHTTPStatusError(Exception):
    def __init__(self, status_code: int, text: str = ""):
        self.response = FakeResponse(status_code, text)


def test_classify_401_as_auth_error():
    exc = FakeHTTPStatusError(401, "Unauthorized")
    err = _classify_http_error(exc, "openai")
    assert isinstance(err, AdapterAuthError)
    assert err.status_code == 401
    assert err.adapter == "openai"


def test_classify_403_as_auth_error():
    exc = FakeHTTPStatusError(403, "Forbidden")
    err = _classify_http_error(exc, "openai")
    assert isinstance(err, AdapterAuthError)


def test_classify_429_as_rate_limit():
    exc = FakeHTTPStatusError(429, "Too Many Requests")
    err = _classify_http_error(exc, "openai")
    assert isinstance(err, AdapterRateLimitError)
    assert err.status_code == 429


def test_classify_500_as_adapter_error():
    exc = FakeHTTPStatusError(500, "Internal Server Error")
    err = _classify_http_error(exc, "test")
    assert isinstance(err, AdapterError)
    assert err.status_code == 502  # Our AdapterError base uses 502


def test_classify_preserves_body_snippet():
    exc = FakeHTTPStatusError(500, "a" * 300)
    err = _classify_http_error(exc, "test")
    # Details should be truncated to 200 chars
    assert len(err.details) <= 200
