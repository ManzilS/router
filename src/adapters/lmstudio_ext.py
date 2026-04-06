"""Adapter for LMStudio's local OpenAI-compatible server.

LMStudio runs on localhost:1234 by default and speaks the OpenAI API
format.  This adapter inherits from OpenAIAdapter and just changes the
defaults so it works out of the box with zero configuration.
"""

from __future__ import annotations

from src.adapters.openai_ext import OpenAIAdapter


class LMStudioAdapter(OpenAIAdapter):
    """LMStudio adapter — OpenAI-compatible at localhost:1234."""

    name = "lmstudio"

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout)
