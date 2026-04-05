"""Adapter for OpenAI-compatible APIs (OpenAI, LMStudio, vLLM, etc.)."""

from __future__ import annotations

import json
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


class OpenAIAdapter(AdapterBase):
    name = "openai"
    supports_streaming = True

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    def translate_to_ai(self, request: PipelineRequest) -> dict[str, Any]:
        messages = []
        for m in request.messages:
            msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.name:
                msg["name"] = m.name
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in m.tool_calls
                ]
            messages.append(msg)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        return payload

    # ------------------------------------------------------------------
    def translate_to_universal(self, raw: dict[str, Any]) -> PipelineResponse:
        choices = []
        for c in raw.get("choices", []):
            msg_data = c.get("message", {})
            tool_calls = None
            if "tool_calls" in msg_data:
                from src.core.models import FunctionCall, ToolCall

                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in msg_data["tool_calls"]
                ]

            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=Message(
                        role=Role(msg_data.get("role", "assistant")),
                        content=msg_data.get("content", ""),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

        usage_raw = raw.get("usage", {})
        return PipelineResponse(
            id=raw.get("id", ""),
            choices=choices,
            model=raw.get("model", ""),
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
        )

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------
    async def send(self, request: PipelineRequest) -> PipelineResponse:
        payload = self.translate_to_ai(request)
        # Force non-streaming for the synchronous path
        payload["stream"] = False

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return self.translate_to_universal(resp.json())

    # ------------------------------------------------------------------
    async def stream(self, request: PipelineRequest) -> AsyncIterator[str]:
        """Yield raw SSE lines from the upstream OpenAI-compatible API."""
        payload = self.translate_to_ai(request)
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        yield f"{line}\n\n"
                        if line.strip() == "data: [DONE]":
                            return
                    elif line == "":
                        continue
