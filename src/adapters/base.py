"""Abstract base class for all AI adapters.

Every adapter must implement two translation methods and a send method.
Adapters that support streaming also implement ``stream()``.
This guarantees that the rest of the system only speaks the Universal Format.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from src.core.models import PipelineRequest, PipelineResponse


class AdapterBase(abc.ABC):
    """Contract that every adapter must fulfil."""

    name: str = "base"
    supports_streaming: bool = False

    @abc.abstractmethod
    def translate_to_ai(self, request: PipelineRequest) -> dict[str, Any]:
        """Convert Universal Format -> AI-specific payload (dict/JSON)."""

    @abc.abstractmethod
    def translate_to_universal(self, raw: dict[str, Any]) -> PipelineResponse:
        """Convert AI-specific response -> Universal Format."""

    @abc.abstractmethod
    async def send(self, request: PipelineRequest) -> PipelineResponse:
        """Translate, call the remote API, and return a PipelineResponse."""

    async def stream(self, request: PipelineRequest) -> AsyncIterator[str]:
        """Yield SSE-formatted ``data: {...}`` lines for streaming responses.

        Override in subclasses that support streaming. The default
        implementation falls back to ``send()`` and yields one big chunk.
        """
        resp = await self.send(request)
        # Fake a single SSE chunk from the non-streaming response
        import json, time
        chunk = {
            "id": resp.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": resp.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": resp.text},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
