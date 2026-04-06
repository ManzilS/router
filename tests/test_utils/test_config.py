"""Tests for the Settings config."""

from src.utils.config import Settings


def test_default_settings():
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8080
    assert s.dev_mode is False
    assert s.default_adapter == "lmstudio"
    assert s.max_body_size == 10_485_760
    assert s.api_key == ""
    assert s.rate_limit_rpm == 0


def test_settings_override():
    s = Settings(dev_mode=True, api_key="test", rate_limit_rpm=60)
    assert s.dev_mode is True
    assert s.api_key == "test"
    assert s.rate_limit_rpm == 60
