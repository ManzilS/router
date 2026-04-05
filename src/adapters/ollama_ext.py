"""Adapter for Ollama's native /api/chat endpoint."""

from __future__ import annotations

import json
import logging
import time
import uuid
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


class OllamaAdapter(AdapterBase):
    name = "ollama"
    supports_streaming = True

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    def translate_to_ai(self, request: PipelineRequest) -> dict[str, Any]:
        messages = [
            {"role": m.role.value, "content": m.content}
            for m in request.messages
        ]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        return payload

    # ------------------------------------------------------------------
    def translate_to_universal(self, raw: dict[str, Any]) -> PipelineResponse:
        msg = raw.get("message", {})
        return PipelineResponse(
            model=raw.get("model", ""),
            choices=[
                Choice(
                    message=Message(
                        role=Role(msg.get("role", "assistant")),
                        content=msg.get("content", ""),
                    ),
                    finish_reason="stop" if raw.get("done") else None,
                )
            ],
            usage=Usage(
                prompt_tokens=raw.get("prompt_eval_count", 0),
                completion_tokens=raw.get("eval_count", 0),
                total_tokens=(
                    raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0)
                ),
            ),
        )

    # ------------------------------------------------------------------
    async def send(self, request: PipelineRequest) -> PipelineResponse:
        payload = self.translate_to_ai(request)
        payload["stream"] = False
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return self.translate_to_universal(resp.json())

    # ------------------------------------------------------------------
    async def stream(self, request: PipelineRequest) -> AsyncIterator[str]:
        """Stream Ollama responses, converting NDJSON -> OpenAI SSE format.

        Ollama streams newline-delimited JSON objects.  We convert each one
        into the ``data: {...}`` SSE format that OpenAI clients expect.
        """
        payload = self.translate_to_ai(request)
        payload["stream"] = True
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    content = obj.get("message", {}).get("content", "")
                    done = obj.get("done", False)

                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": obj.get("model", request.model),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                    if done:
                        yield "data: [DONE]\n\n"
                        return
