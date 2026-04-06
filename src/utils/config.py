"""Application configuration loaded from environment / .env file
and the declarative plugins.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"

    # Orchestrator
    default_adapter: str = "lmstudio"
    max_loops: int = 3

    # Plugin config path (relative to project root)
    plugins_config: str = "plugins.yaml"

    # Storage (used by plugins that need persistence)
    data_dir: str = str(Path.home() / ".ai-router")

    model_config = {"env_prefix": "ROUTER_", "env_file": ".env"}
