"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"

    # Default adapter
    default_adapter: str = "openai"
    max_loops: int = 3

    # OpenAI adapter
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # Ollama adapter
    ollama_base_url: str = "http://localhost:11434"

    # Gemini adapter
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Storage
    db_path: str = str(Path.home() / ".ai-gateway" / "history.db")
    chroma_path: str = str(Path.home() / ".ai-gateway" / "chroma_db")

    # Fallback / circuit breaker
    fallback_chain: list[str] = []  # e.g. ["openai", "ollama", "gemini"]
    fallback_failure_threshold: int = 3
    fallback_recovery_timeout: float = 60.0

    # Feature toggles
    enable_rag: bool = True
    enable_early_exit: bool = True
    enable_logger: bool = True

    model_config = {"env_prefix": "AIGW_", "env_file": ".env"}
