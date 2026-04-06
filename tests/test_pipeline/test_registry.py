"""Tests for the plugin registry — auto-discovery and instantiation."""

import pytest

from src.core.registry import PluginRegistry, load_plugins_config


def test_load_default_config():
    """When no file exists, defaults are returned."""
    config = load_plugins_config("/nonexistent/path.yaml")
    assert "adapters" in config
    assert "middleware" in config
    assert "lmstudio" in config["adapters"]


def test_registry_discovers_lmstudio():
    """The default config should discover the LMStudio adapter."""
    config = load_plugins_config(None)
    registry = PluginRegistry(config=config)
    registry.discover()

    assert "lmstudio" in registry.adapters
    assert registry.adapters["lmstudio"].name == "lmstudio"


def test_registry_respects_enabled_flag():
    """Disabled adapters should not be loaded."""
    config = {
        "adapters": {
            "openai": {"enabled": False, "module": "src.adapters.openai_ext"},
        },
        "middleware": {"pre": [], "post": []},
    }
    registry = PluginRegistry(config=config)
    registry.discover()

    assert "openai" not in registry.adapters


def test_registry_loads_middleware():
    """Enabled middleware should be discovered and instantiated."""
    config = {
        "adapters": {},
        "middleware": {
            "pre": [
                {"name": "early_exit", "enabled": True, "module": "src.middleware.early_exit"},
            ],
            "post": [
                {"name": "logger", "enabled": True, "module": "src.middleware.logger"},
            ],
        },
    }
    registry = PluginRegistry(config=config)
    registry.discover()

    assert len(registry.pre_middleware) == 1
    assert registry.pre_middleware[0].name == "early_exit"
    assert len(registry.post_middleware) == 1
    assert registry.post_middleware[0].name == "logger"


def test_registry_passes_settings_to_adapter():
    """Settings from config should be passed as kwargs to the adapter."""
    config = {
        "adapters": {
            "openai": {
                "enabled": True,
                "module": "src.adapters.openai_ext",
                "settings": {
                    "base_url": "http://custom:9999/v1",
                    "api_key": "test-key",
                },
            },
        },
        "middleware": {"pre": [], "post": []},
    }
    registry = PluginRegistry(config=config)
    registry.discover()

    adapter = registry.adapters["openai"]
    assert adapter.base_url == "http://custom:9999/v1"
    assert adapter.api_key == "test-key"


def test_registry_handles_bad_module():
    """A bad module path should not crash discovery."""
    config = {
        "adapters": {
            "bad": {"enabled": True, "module": "src.adapters.nonexistent_module"},
        },
        "middleware": {"pre": [], "post": []},
    }
    registry = PluginRegistry(config=config)
    registry.discover()  # Should not raise

    assert "bad" not in registry.adapters


def test_registry_empty_config():
    """Empty config should produce empty registries."""
    config = {"adapters": {}, "middleware": {"pre": [], "post": []}}
    registry = PluginRegistry(config=config)
    registry.discover()

    assert len(registry.adapters) == 0
    assert len(registry.pre_middleware) == 0
    assert len(registry.post_middleware) == 0
