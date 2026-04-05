"""Tests for the early-exit middleware."""

import pytest

from src.core.models import Message, PipelineRequest, Role
from src.core.state import PipelineState
from src.middleware.early_exit import EarlyExitMiddleware


@pytest.fixture
def middleware():
    return EarlyExitMiddleware()


def _state_with_message(text: str) -> PipelineState:
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content=text)]
        )
    )


@pytest.mark.asyncio
async def test_time_query(middleware):
    state = _state_with_message("What time is it?")
    state = await middleware.process(state)

    assert state.early_exit is True
    assert "current time" in state.response.text.lower()


@pytest.mark.asyncio
async def test_date_query(middleware):
    state = _state_with_message("What date is today?")
    state = await middleware.process(state)

    assert state.early_exit is True
    assert state.response is not None


@pytest.mark.asyncio
async def test_ping(middleware):
    state = _state_with_message("ping")
    state = await middleware.process(state)

    assert state.early_exit is True
    assert "pong" in state.response.text.lower()


@pytest.mark.asyncio
async def test_math(middleware):
    state = _state_with_message("2 + 3 * 4")
    state = await middleware.process(state)

    assert state.early_exit is True
    assert "14" in state.response.text


@pytest.mark.asyncio
async def test_normal_query_passes_through(middleware):
    state = _state_with_message("Explain quantum computing")
    state = await middleware.process(state)

    assert state.early_exit is False
    assert state.response is None


@pytest.mark.asyncio
async def test_disabled(middleware):
    middleware.enabled = False
    state = _state_with_message("What time is it?")
    state = await middleware.process(state)

    assert state.early_exit is False
