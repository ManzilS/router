"""Integration test — send a fake request to the FastAPI app and verify
the full round-trip through mocked adapters."""

import pytest
from httpx import AsyncClient, ASGITransport

from src.adapters.base import AdapterBase
from src.core.models import Choice, Message, PipelineResponse, Role
from src.gateway.main import create_app
from src.utils.config import Settings


class FakeAdapter(AdapterBase):
    name = "openai"

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        return PipelineResponse(
            model="fake-model",
            choices=[
                Choice(
                    message=Message(
                        role=Role.ASSISTANT,
                        content=f"Echo: {request.messages[-1].content}",
                    ),
                    finish_reason="stop",
                )
            ],
        )


@pytest.fixture
def app():
    settings = Settings(
        enable_rag=False,
        enable_logger=False,
        enable_early_exit=False,
    )
    app = create_app(settings)
    # Swap in our fake adapter
    app.state.orchestrator.pipeline.adapters["openai"] = FakeAdapter()
    return app


@pytest.mark.asyncio
async def test_chat_completions_round_trip(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Echo: Hello world"
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_list_models(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert any(m["id"] == "openai" for m in data["data"])


@pytest.mark.asyncio
async def test_early_exit_time_query():
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
                "messages": [{"role": "user", "content": "What time is it?"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "time" in data["choices"][0]["message"]["content"].lower()


@pytest.mark.asyncio
async def test_adapter_routing_via_model_prefix(app):
    """Test that 'ollama/llama3' routes to the ollama adapter."""
    # Add a fake ollama adapter
    class FakeOllama(FakeAdapter):
        name = "ollama"

        async def send(self, request):
            return PipelineResponse(
                model="ollama-fake",
                choices=[
                    Choice(
                        message=Message(
                            role=Role.ASSISTANT,
                            content="From Ollama!",
                        ),
                        finish_reason="stop",
                    )
                ],
            )

    app.state.orchestrator.pipeline.adapters["ollama"] = FakeOllama()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama/llama3",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "From Ollama!"
