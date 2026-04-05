"""Tests for the orchestrator — loop control and fan-out."""

import pytest

from src.core.models import (
    Choice,
    FunctionCall,
    Message,
    PipelineRequest,
    PipelineResponse,
    Role,
    ToolCall,
    Usage,
)
from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.core.state import PipelineState
from src.adapters.base import AdapterBase


class LoopingAdapter(AdapterBase):
    """Returns tool calls for the first N requests, then a normal reply."""

    name = "looping"

    def __init__(self, tool_call_count: int = 1):
        self.call_count = 0
        self.tool_call_count = tool_call_count

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        self.call_count += 1
        if self.call_count <= self.tool_call_count:
            return PipelineResponse(
                choices=[
                    Choice(
                        message=Message(
                            role=Role.ASSISTANT,
                            content="",
                            tool_calls=[
                                ToolCall(
                                    function=FunctionCall(
                                        name="lookup",
                                        arguments='{"q": "test"}',
                                    )
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            )
        return PipelineResponse(
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content="Final answer"),
                    finish_reason="stop",
                )
            ]
        )


@pytest.fixture
def basic_state():
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="test")],
            target_adapter="looping",
        )
    )


@pytest.mark.asyncio
async def test_orchestrator_loops_on_tool_calls(basic_state):
    adapter = LoopingAdapter(tool_call_count=1)
    pipeline = Pipeline(adapters={"looping": adapter})
    orch = Orchestrator(pipeline=pipeline, max_loops=3)

    # Add a pending tool result so the loop has data
    basic_state.pending_tool_results.append(
        {"tool_call_id": "call_test", "content": "tool result"}
    )

    state = await orch.run(basic_state)

    assert state.response.text == "Final answer"
    assert adapter.call_count == 2  # once with tool call, once final
    assert state.loop_count == 1


@pytest.mark.asyncio
async def test_orchestrator_respects_max_loops(basic_state):
    adapter = LoopingAdapter(tool_call_count=10)  # always returns tool calls
    pipeline = Pipeline(adapters={"looping": adapter})
    orch = Orchestrator(pipeline=pipeline, max_loops=2)

    state = await orch.run(basic_state)

    # Should stop after max_loops even though adapter keeps returning tool calls
    assert state.loop_count == 2
    assert adapter.call_count == 3  # initial + 2 loops
