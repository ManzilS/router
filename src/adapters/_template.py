"""
=============================================================================
ADAPTER TEMPLATE — Copy this file to create a new adapter.
=============================================================================

Steps:
  1. Copy this file:  cp _template.py myai_ext.py
  2. Rename the class to match your AI (e.g. MyAIAdapter)
  3. Implement the three required methods
  4. Add an entry to plugins.yaml:

     adapters:
       myai:
         enabled: true
         module: src.adapters.myai_ext
         settings:
           api_key: "..."

  That's it — the plugin registry handles the rest.

The __init__ kwargs must match the `settings:` keys in plugins.yaml.
=============================================================================
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.adapters.base import AdapterBase
from src.core.models import (
    Choice,
    Message,
    PipelineRequest,
    PipelineResponse,
    Role,
    Usage,
)

logger = logging.getLogger(__name__)


class TemplateAdapter(AdapterBase):
    """Replace this with your adapter's name and description."""

    name = "template"
    supports_streaming = False  # Set True if you implement stream()

    def __init__(
        self,
        base_url: str = "http://localhost:9999",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        # These kwargs come from plugins.yaml `settings:` block
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    def translate_to_ai(self, request: PipelineRequest) -> dict[str, Any]:
        """Convert our Universal Format into whatever JSON your AI expects.

        This is where you map our Message/Role types into the AI's format.
        """
        # Example: most APIs want a list of {"role": "...", "content": "..."}
        messages = [
            {"role": m.role.value, "content": m.content}
            for m in request.messages
        ]
        return {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }

    # ------------------------------------------------------------------
    def translate_to_universal(self, raw: dict[str, Any]) -> PipelineResponse:
        """Convert the AI's raw JSON response into our Universal Format.

        Map their response structure into Choice + Message + Usage.
        """
        return PipelineResponse(
            model=raw.get("model", "unknown"),
            choices=[
                Choice(
                    message=Message(
                        role=Role.ASSISTANT,
                        content=raw.get("response", ""),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=raw.get("prompt_tokens", 0),
                completion_tokens=raw.get("completion_tokens", 0),
                total_tokens=raw.get("total_tokens", 0),
            ),
        )

    # ------------------------------------------------------------------
    async def send(self, request: PipelineRequest) -> PipelineResponse:
        """Translate → call the API → translate back."""
        payload = self.translate_to_ai(request)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/your/endpoint",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return self.translate_to_universal(resp.json())

    # ------------------------------------------------------------------
    # Optional: implement stream() for SSE streaming support.
    # If you don't implement this, the base class will fall back to
    # calling send() and yielding one big chunk.
    #
    # async def stream(self, request: PipelineRequest) -> AsyncIterator[str]:
    #     """Yield SSE-formatted 'data: {...}' lines."""
    #     ...
