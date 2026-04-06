"""Application configuration loaded from environment / .env file
and the declarative plugins.yaml."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"

    # Dev mode — enables verbose logging, error details in responses,
    # hot reload, request/response body dumps
    dev_mode: bool = False

    # Orchestrator
    default_adapter: str = "lmstudio"
    max_loops: int = 3

    # Plugin config path (relative to project root)
    plugins_config: str = "plugins.yaml"

    # Storage (used by plugins that need persistence)
    data_dir: str = str(Path.home() / ".ai-router")

    # --- Production hardening ---

    # CORS — comma-separated origins, or "*" for all
    cors_origins: str = "*"

    # Request limits
    request_timeout: float = 120.0  # seconds — overall gateway timeout
    max_body_size: int = 10_485_760  # 10 MB

    # Authentication — if set, requires Bearer token
    api_key: str = ""  # empty = no auth

    # Rate limiting — requests per minute per IP (0 = disabled)
    rate_limit_rpm: int = 0

    model_config = {"env_prefix": "ROUTER_", "env_file": ".env"}
