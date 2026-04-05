"""Integration tests for the fallback orchestrator path."""

import pytest

from src.adapters.base import AdapterBase
from src.core.models import Choice, Message, PipelineRequest, PipelineResponse, Role
from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.core.state import PipelineState
from src.middleware.fallback import FallbackMiddleware


class FailingAdapter(AdapterBase):
    name = "failing"

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        raise ConnectionError("Adapter is down!")


class WorkingAdapter(AdapterBase):
    name = "working"

    def translate_to_ai(self, request):
        return {}

    def translate_to_universal(self, raw):
        return raw

    async def send(self, request):
        return PipelineResponse(
            model="working-model",
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content="I'm alive!"),
                    finish_reason="stop",
                )
            ],
        )


@pytest.fixture
def fallback_mw():
    return FallbackMiddleware(chain=["primary", "backup"])


@pytest.fixture
def pipeline_with_fallback(fallback_mw):
    return Pipeline(
        pre_middleware=[fallback_mw],
        adapters={
            "primary": FailingAdapter(),
            "backup": WorkingAdapter(),
        },
    )


@pytest.fixture
def state():
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="test")],
            target_adapter="primary",
        )
    )


@pytest.mark.asyncio
async def test_fallback_to_backup_on_failure(pipeline_with_fallback, state):
    orch = Orchestrator(pipeline=pipeline_with_fallback)
    state = await orch.run(state)

    assert state.response is not None
    assert state.response.text == "I'm alive!"
    assert state.response.model == "working-model"


@pytest.mark.asyncio
async def test_fallback_all_fail(fallback_mw):
    pipeline = Pipeline(
        pre_middleware=[fallback_mw],
        adapters={
            "primary": FailingAdapter(),
            "backup": FailingAdapter(),
        },
    )
    orch = Orchestrator(pipeline=pipeline)
    state = PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="test")],
            target_adapter="primary",
        )
    )

    with pytest.raises(ConnectionError, match="Adapter is down"):
        await orch.run(state)


@pytest.mark.asyncio
async def test_fallback_records_circuit_state(fallback_mw, pipeline_with_fallback, state):
    orch = Orchestrator(pipeline=pipeline_with_fallback)
    await orch.run(state)

    # Primary should have recorded a failure
    primary_circuit = fallback_mw._circuits.get("primary")
    assert primary_circuit is not None
    assert primary_circuit.failures >= 1

    # Backup should have recorded a success
    backup_circuit = fallback_mw._circuits.get("backup")
    assert backup_circuit is not None
    assert backup_circuit.failures == 0


@pytest.mark.asyncio
async def test_no_fallback_chain_uses_normal_path():
    """Without a fallback chain, the orchestrator uses normal dispatch."""
    adapter = WorkingAdapter()
    pipeline = Pipeline(adapters={"working": adapter})
    orch = Orchestrator(pipeline=pipeline)

    state = PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="test")],
            target_adapter="working",
        )
    )
    state = await orch.run(state)

    assert state.response.text == "I'm alive!"
