"""Tests for gateway-level middleware: auth, rate limiting, body limits, CORS."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.adapters.base import AdapterBase
from src.core.models import Choice, Message, PipelineResponse, Role
from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.gateway.main import create_app
from src.utils.config import Settings


class FakeAdapter(AdapterBase):
    name = "lmstudio"

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        return PipelineResponse(
            model="fake",
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content="OK"),
                    finish_reason="stop",
                )
            ],
        )


def _make_app(**settings_kwargs):
    settings = Settings(default_adapter="lmstudio", **settings_kwargs)
    app = create_app(settings)
    pipeline = Pipeline(
        adapters={"lmstudio": FakeAdapter()},
        default_adapter="lmstudio",
    )
    app.state.orchestrator = Orchestrator(pipeline=pipeline)
    return app


def _chat_payload():
    return {
        "messages": [{"role": "user", "content": "Hi"}],
    }


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auth_blocks_without_key():
    app = _make_app(api_key="secret-key-123")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_auth_blocks_wrong_key():
    app = _make_app(api_key="secret-key-123")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_allows_correct_key():
    app = _make_app(api_key="secret-key-123")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"Authorization": "Bearer secret-key-123"},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_skips_health_endpoints():
    app = _make_app(api_key="secret-key-123")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_no_auth_when_key_not_set():
    app = _make_app(api_key="")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Request ID context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_id_in_response_header():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=_chat_payload())
    assert "x-request-id" in resp.headers


@pytest.mark.asyncio
async def test_custom_request_id_preserved():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Request-ID": "my-custom-id"},
        )
    assert resp.headers["x-request-id"] == "my-custom-id"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "adapters" in data


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_blocks_excess_requests():
    app = _make_app(rate_limit_rpm=3)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            resp = await client.post("/v1/chat/completions", json=_chat_payload())
            assert resp.status_code == 200

        # 4th request should be rate limited
        resp = await client.post("/v1/chat/completions", json=_chat_payload())
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers


@pytest.mark.asyncio
async def test_rate_limit_disabled_by_default():
    app = _make_app()  # rate_limit_rpm defaults to 0
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(10):
            resp = await client.post("/v1/chat/completions", json=_chat_payload())
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Body limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_body_limit_rejects_oversized():
    app = _make_app(max_body_size=100)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"Content-Length": "999999"},
        )
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Dev mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dev_mode_enables_docs():
    app = _make_app(dev_mode=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/docs")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_prod_mode_disables_docs():
    app = _make_app(dev_mode=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/docs")
    assert resp.status_code in (404, 422)
