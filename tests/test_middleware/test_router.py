"""Tests for the router fan-out middleware."""

import pytest

from src.core.models import Message, PipelineRequest, Role
from src.core.state import PipelineState
from src.middleware.router import RouterMiddleware


@pytest.fixture
def middleware():
    return RouterMiddleware(targets=["openai", "ollama"])


def _state(metadata=None):
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="test")],
            metadata=metadata or {},
        )
    )


@pytest.mark.asyncio
async def test_sets_fan_out_targets(middleware):
    state = _state()
    state = await middleware.process(state)
    assert state.fan_out_targets == ["openai", "ollama"]


@pytest.mark.asyncio
async def test_metadata_override(middleware):
    state = _state(metadata={"fan_out": ["gemini", "lmstudio"]})
    state = await middleware.process(state)
    assert state.fan_out_targets == ["gemini", "lmstudio"]


@pytest.mark.asyncio
async def test_single_target_no_fan_out():
    mw = RouterMiddleware(targets=["openai"])
    state = _state()
    state = await mw.process(state)
    assert state.fan_out_targets == []  # single target = no fan-out


@pytest.mark.asyncio
async def test_skips_when_response_exists(middleware):
    state = _state()
    from src.core.models import Choice, PipelineResponse

    state.response = PipelineResponse(
        choices=[Choice(message=Message(role=Role.ASSISTANT, content="done"))]
    )
    state = await middleware.process(state)
    assert state.fan_out_targets == []
