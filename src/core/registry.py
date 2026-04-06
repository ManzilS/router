"""Plugin Registry — auto-discovers and manages adapters and middleware.

The registry reads ``plugins.yaml`` to decide which plugins are active
and in what order.  Plugins are discovered from the ``src/adapters/`` and
``src/middleware/`` directories automatically — drop a file in the right
folder and add a line to ``plugins.yaml`` and it's live.

This is the single place where plugins are resolved.  Nothing else in
the codebase needs to know the concrete class names.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

import yaml

from src.adapters.base import AdapterBase
from src.middleware.base import MiddlewareBase

logger = logging.getLogger(__name__)

# Default plugins.yaml that ships with the router
_DEFAULT_CONFIG = {
    "adapters": {
        "lmstudio": {"enabled": True, "module": "src.adapters.lmstudio_ext"},
        "openai": {"enabled": False, "module": "src.adapters.openai_ext"},
        "ollama": {"enabled": False, "module": "src.adapters.ollama_ext"},
        "gemini": {"enabled": False, "module": "src.adapters.gemini_ext"},
    },
    "middleware": {
        "pre": [
            {"name": "early_exit", "enabled": False, "module": "src.middleware.early_exit"},
            {"name": "router", "enabled": False, "module": "src.middleware.router"},
        ],
        "post": [
            {"name": "logger", "enabled": False, "module": "src.middleware.logger"},
        ],
    },
}


def load_plugins_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load plugins.yaml, falling back to defaults if not found."""
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    return loaded
            logger.warning("plugins.yaml was empty, using defaults")
        else:
            logger.info("No plugins.yaml found at %s, using defaults", path)
    return _DEFAULT_CONFIG


def _find_plugin_class(module_path: str, base_class: type) -> type | None:
    """Import a module and find the first class that subclasses base_class."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        logger.error("Failed to import plugin module: %s", module_path)
        return None

    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            return obj
    logger.error("No %s subclass found in %s", base_class.__name__, module_path)
    return None


class PluginRegistry:
    """Discovers, instantiates, and manages all adapters and middleware."""

    def __init__(self, config: dict[str, Any], settings_override: dict[str, Any] | None = None):
        self.config = config
        self.settings_override = settings_override or {}
        self._adapters: dict[str, AdapterBase] = {}
        self._pre_middleware: list[MiddlewareBase] = []
        self._post_middleware: list[MiddlewareBase] = []

    def discover(self) -> None:
        """Scan the config and instantiate all enabled plugins."""
        self._discover_adapters()
        self._discover_middleware()

    # --- Public accessors ---

    @property
    def adapters(self) -> dict[str, AdapterBase]:
        return self._adapters

    @property
    def pre_middleware(self) -> list[MiddlewareBase]:
        return self._pre_middleware

    @property
    def post_middleware(self) -> list[MiddlewareBase]:
        return self._post_middleware

    # --- Internal ---

    def _discover_adapters(self) -> None:
        adapters_cfg = self.config.get("adapters", {})
        for adapter_name, adapter_conf in adapters_cfg.items():
            if not adapter_conf.get("enabled", False):
                logger.debug("Adapter '%s' is disabled, skipping", adapter_name)
                continue

            module_path = adapter_conf.get("module", "")
            cls = _find_plugin_class(module_path, AdapterBase)
            if cls is None:
                continue

            # Merge settings: plugins.yaml `settings:` block + env overrides
            plugin_settings = adapter_conf.get("settings", {})
            override_key = f"adapter_{adapter_name}"
            if override_key in self.settings_override:
                plugin_settings.update(self.settings_override[override_key])

            try:
                instance = cls(**plugin_settings)
                self._adapters[adapter_name] = instance
                logger.info("Loaded adapter: %s (%s)", adapter_name, cls.__name__)
            except Exception:
                logger.exception("Failed to instantiate adapter '%s'", adapter_name)

    def _discover_middleware(self) -> None:
        mw_cfg = self.config.get("middleware", {})

        for phase, target_list in [("pre", self._pre_middleware), ("post", self._post_middleware)]:
            entries = mw_cfg.get(phase, [])
            for entry in entries:
                if not entry.get("enabled", False):
                    logger.debug("Middleware '%s' (%s) is disabled", entry.get("name"), phase)
                    continue

                module_path = entry.get("module", "")
                cls = _find_plugin_class(module_path, MiddlewareBase)
                if cls is None:
                    continue

                plugin_settings = entry.get("settings", {})
                try:
                    instance = cls(**plugin_settings)
                    target_list.append(instance)
                    logger.info("Loaded %s-middleware: %s (%s)", phase, entry.get("name"), cls.__name__)
                except Exception:
                    logger.exception("Failed to instantiate middleware '%s'", entry.get("name"))
