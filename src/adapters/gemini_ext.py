"""Adapter for Google's Gemini REST API (generateContent)."""

from __future__ import annotations

import logging
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

# Gemini uses different role names
_ROLE_MAP = {
    Role.SYSTEM: "user",  # Gemini handles system via systemInstruction
    Role.USER: "user",
    Role.ASSISTANT: "model",
    Role.TOOL: "user",
}


class GeminiAdapter(AdapterBase):
    name = "gemini"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.default_model = model
        self.timeout = timeout

    # ------------------------------------------------------------------
    def translate_to_ai(self, request: PipelineRequest) -> dict[str, Any]:
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for m in request.messages:
            if m.role == Role.SYSTEM:
                system_parts.append({"text": m.content})
                continue
            contents.append(
                {
                    "role": _ROLE_MAP[m.role],
                    "parts": [{"text": m.content}],
                }
            )

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {
                "role": "user",
                "parts": system_parts,
            }

        payload["generationConfig"] = {
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = request.max_tokens
        if request.stop:
            payload["generationConfig"]["stopSequences"] = request.stop

        return payload

    # ------------------------------------------------------------------
    def translate_to_universal(self, raw: dict[str, Any]) -> PipelineResponse:
        candidates = raw.get("candidates", [])
        choices: list[Choice] = []
        for i, cand in enumerate(candidates):
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            choices.append(
                Choice(
                    index=i,
                    message=Message(role=Role.ASSISTANT, content=text),
                    finish_reason=cand.get("finishReason", "STOP").lower(),
                )
            )

        usage_raw = raw.get("usageMetadata", {})
        return PipelineResponse(
            model=self.default_model,
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_raw.get("promptTokenCount", 0),
                completion_tokens=usage_raw.get("candidatesTokenCount", 0),
                total_tokens=usage_raw.get("totalTokenCount", 0),
            ),
        )

    # ------------------------------------------------------------------
    async def send(self, request: PipelineRequest) -> PipelineResponse:
        model = request.model if request.model != "default" else self.default_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={self.api_key}"
        )
        payload = self.translate_to_ai(request)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return self.translate_to_universal(resp.json())
