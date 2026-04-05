"""Tests for SSE streaming through the gateway."""

import json

import pytest
from httpx import AsyncClient, ASGITransport

from src.adapters.base import AdapterBase
from src.core.models import Choice, Message, PipelineResponse, Role
from src.gateway.main import create_app
from src.utils.config import Settings


class FakeStreamingAdapter(AdapterBase):
    """Adapter that yields 3 SSE chunks then [DONE]."""
    name = "openai"
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


@pytest.fixture
def app():
    settings = Settings(
        enable_rag=False,
        enable_logger=False,
        enable_early_exit=False,
    )
    app = create_app(settings)
    app.state.orchestrator.pipeline.adapters["openai"] = FakeStreamingAdapter()
    return app


@pytest.mark.asyncio
async def test_streaming_returns_sse_chunks(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    # Parse out the SSE lines
    lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
    assert len(lines) == 4  # 3 content chunks + [DONE]

    # Verify content accumulates
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
    settings = Settings(
        enable_rag=False,
        enable_logger=False,
        enable_early_exit=True,
    )
    app = create_app(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
    # Should have content chunk + [DONE]
    assert any("[DONE]" in l for l in lines)
    # The content should be the early exit response
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
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "word1 word2 word3"
