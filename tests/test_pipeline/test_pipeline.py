"""Tests for the pipeline engine — uses mock adapters and middleware."""

import pytest

from src.core.models import (
    Choice,
    Message,
    PipelineRequest,
    PipelineResponse,
    Role,
    Usage,
)
from src.core.pipeline import Pipeline
from src.core.state import PipelineState
from src.adapters.base import AdapterBase
from src.middleware.base import MiddlewareBase


# --- Mock adapter ---
class MockAdapter(AdapterBase):
    name = "mock"

    def __init__(self, reply: str = "Mock reply"):
        self.reply = reply
        self.last_request = None

    def translate_to_ai(self, request):
        return {"messages": [m.model_dump() for m in request.messages]}

    def translate_to_universal(self, raw):
        return PipelineResponse(
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content=raw["reply"]),
                    finish_reason="stop",
                )
            ],
        )

    async def send(self, request):
        self.last_request = request
        return self.translate_to_universal({"reply": self.reply})


# --- Mock middleware ---
class AppendMiddleware(MiddlewareBase):
    """Appends a marker to the last user message."""

    name = "append"

    def __init__(self, marker: str = " [processed]"):
        self.marker = marker

    async def process(self, state):
        if state.response is None:  # pre-processing
            for m in reversed(state.request.messages):
                if m.role == Role.USER:
                    m.content += self.marker
                    break
        return state


class EarlyExitMW(MiddlewareBase):
    name = "early_exit_test"

    async def process(self, state):
        state.early_exit = True
        state.response = PipelineResponse(
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content="early!"),
                    finish_reason="stop",
                )
            ]
        )
        return state


# --- Tests ---

@pytest.fixture
def mock_adapter():
    return MockAdapter(reply="Hello from mock")


@pytest.fixture
def basic_state():
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="Hi")],
            target_adapter="mock",
        )
    )


@pytest.mark.asyncio
async def test_basic_pipeline_execution(mock_adapter, basic_state):
    pipeline = Pipeline(adapters={"mock": mock_adapter})
    state = await pipeline.execute(basic_state)

    assert state.response is not None
    assert state.response.text == "Hello from mock"
    assert mock_adapter.last_request is not None


@pytest.mark.asyncio
async def test_pre_middleware_modifies_request(mock_adapter, basic_state):
    pipeline = Pipeline(
        pre_middleware=[AppendMiddleware(" [pre]")],
        adapters={"mock": mock_adapter},
    )
    state = await pipeline.execute(basic_state)

    # The middleware should have modified the message before sending
    assert mock_adapter.last_request.messages[0].content == "Hi [pre]"


@pytest.mark.asyncio
async def test_early_exit_skips_adapter(mock_adapter, basic_state):
    pipeline = Pipeline(
        pre_middleware=[EarlyExitMW()],
        adapters={"mock": mock_adapter},
    )
    state = await pipeline.execute(basic_state)

    assert state.early_exit is True
    assert state.response.text == "early!"
    assert mock_adapter.last_request is None  # adapter was never called


@pytest.mark.asyncio
async def test_missing_adapter_raises(basic_state):
    pipeline = Pipeline(adapters={})
    basic_state.request.target_adapter = "nonexistent"

    with pytest.raises(ValueError, match="No adapter registered"):
        await pipeline.execute(basic_state)
