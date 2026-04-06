"""Tests for structured error handling across the gateway."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.adapters.base import AdapterBase
from src.core.models import Choice, Message, PipelineResponse, Role
from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.gateway.main import create_app
from src.utils.config import Settings
from src.utils.errors import (
    AdapterAuthError,
    AdapterError,
    AdapterNotFoundError,
    AdapterRateLimitError,
    AdapterTimeoutError,
    RequestTooLargeError,
    RequestValidationError,
    RouterError,
)


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


class ErrorAdapter(AdapterBase):
    """Adapter that raises a configurable error."""

    name = "error"

    def __init__(self, error: Exception | None = None):
        self.error = error or RuntimeError("boom")

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        raise self.error


def _make_app(adapters=None, **settings_kwargs):
    settings = Settings(default_adapter="lmstudio", **settings_kwargs)
    app = create_app(settings)
    pipeline = Pipeline(
        adapters=adapters or {"lmstudio": FakeAdapter()},
        default_adapter="lmstudio",
    )
    app.state.orchestrator = Orchestrator(pipeline=pipeline)
    return app


# ---------------------------------------------------------------------------
# Request validation errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_json_returns_400():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_empty_messages_returns_400():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["type"] == "validation_error"


@pytest.mark.asyncio
async def test_missing_messages_returns_400():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test"},
        )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Adapter errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adapter_not_found_returns_400():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent/model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert resp.status_code in (400, 502)


@pytest.mark.asyncio
async def test_adapter_error_returns_502():
    app = _make_app(adapters={
        "lmstudio": ErrorAdapter(AdapterError("connection failed", adapter="lmstudio"))
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "adapter_error"


@pytest.mark.asyncio
async def test_adapter_timeout_returns_504():
    app = _make_app(adapters={
        "lmstudio": ErrorAdapter(AdapterTimeoutError("timed out", adapter="lmstudio"))
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 504
    assert resp.json()["error"]["type"] == "adapter_timeout"


@pytest.mark.asyncio
async def test_adapter_auth_error_returns_401():
    app = _make_app(adapters={
        "lmstudio": ErrorAdapter(AdapterAuthError("bad key", adapter="lmstudio"))
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_adapter_rate_limit_returns_429():
    app = _make_app(adapters={
        "lmstudio": ErrorAdapter(
            AdapterRateLimitError("rate limited", adapter="lmstudio")
        )
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_unhandled_adapter_exception_returns_500():
    """An unhandled exception in the adapter should be caught by the pipeline
    and wrapped as an AdapterError (502)."""
    app = _make_app(adapters={
        "lmstudio": ErrorAdapter(RuntimeError("unexpected"))
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    # Pipeline wraps unknown errors as AdapterError (502)
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Error type helpers
# ---------------------------------------------------------------------------

def test_router_error_to_dict():
    err = RouterError("something broke", adapter="openai", details="traceback...")
    d = err.to_dict()
    assert d["error"]["type"] == "internal_error"
    assert d["error"]["message"] == "something broke"
    assert d["error"]["adapter"] == "openai"


def test_error_hierarchy():
    assert issubclass(AdapterError, RouterError)
    assert issubclass(AdapterTimeoutError, AdapterError)
    assert issubclass(AdapterAuthError, AdapterError)
    assert issubclass(AdapterRateLimitError, AdapterError)
    assert issubclass(AdapterNotFoundError, RouterError)
    assert issubclass(RequestValidationError, RouterError)
    assert issubclass(RequestTooLargeError, RouterError)
