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
from src.utils.errors import (
    AdapterAuthError,
    AdapterError,
    AdapterRateLimitError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)

# Gemini uses different role names
_ROLE_MAP = {
    Role.SYSTEM: "user",  # Gemini handles system via systemInstruction
    Role.USER: "user",
    Role.ASSISTANT: "model",
    Role.TOOL: "user",
}

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


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
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30,
                ),
                headers={"X-Goog-Api-Key": self.api_key},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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
        url = f"{_GEMINI_BASE}/models/{model}:generateContent"
        payload = self.translate_to_ai(request)
        client = self._get_client()

        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return self.translate_to_universal(resp.json())
        except httpx.TimeoutException as exc:
            raise AdapterTimeoutError(
                "Gemini timed out", adapter=self.name
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise AdapterAuthError(
                    "Gemini authentication failed", adapter=self.name
                ) from exc
            if status == 429:
                raise AdapterRateLimitError(
                    "Gemini rate limit hit", adapter=self.name
                ) from exc
            raise AdapterError(
                f"Gemini returned {status}", adapter=self.name
            ) from exc
        except httpx.ConnectError as exc:
            raise AdapterError(
                "Cannot connect to Gemini API", adapter=self.name
            ) from exc
