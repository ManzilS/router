"""Tests for SSE streaming through the gateway."""

import json

import pytest
from httpx import AsyncClient, ASGITransport

from src.adapters.base import AdapterBase
from src.core.models import Choice, Message, PipelineResponse, Role
from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.middleware.early_exit import EarlyExitMiddleware
from src.utils.config import Settings


class FakeStreamingAdapter(AdapterBase):
    """Adapter that yields 3 SSE chunks then [DONE]."""
    name = "lmstudio"
    supports_streaming = True

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        return PipelineResponse(
            model="fake",
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content="word1 word2 word3"),
                    finish_reason="stop",
                )
            ],
        )

    async def stream(self, request):
        words = ["word1 ", "word2 ", "word3"]
        for i, w in enumerate(words):
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": w},
                        "finish_reason": "stop" if i == len(words) - 1 else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _make_app(pre_middleware=None):
    """Build a minimal app with the fake streaming adapter."""
    from fastapi import FastAPI
    from src.gateway.server import router

    pipeline = Pipeline(
        pre_middleware=pre_middleware or [],
        adapters={"lmstudio": FakeStreamingAdapter()},
        default_adapter="lmstudio",
    )
    orch = Orchestrator(pipeline=pipeline)

    app = FastAPI()
    app.state.orchestrator = orch
    app.state.settings = Settings(default_adapter="lmstudio")
    app.include_router(router)
    return app


@pytest.fixture
def app():
    return _make_app()


@pytest.mark.asyncio
async def test_streaming_returns_sse_chunks(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
    assert len(lines) == 4  # 3 content chunks + [DONE]

    full_text = ""
    for line in lines:
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        full_text += obj["choices"][0]["delta"].get("content", "")
    assert full_text == "word1 word2 word3"


@pytest.mark.asyncio
async def test_streaming_early_exit():
    """Early exit queries should still work under stream=true."""
    app = _make_app(pre_middleware=[EarlyExitMiddleware()])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
    assert any("[DONE]" in l for l in lines)
    content_line = lines[0]
    obj = json.loads(content_line[len("data:"):].strip())
    assert "pong" in obj["choices"][0]["delta"]["content"].lower()


@pytest.mark.asyncio
async def test_nonstreaming_still_works(app):
    """Verify non-streaming requests are unaffected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "word1 word2 word3"
