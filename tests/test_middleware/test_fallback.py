"""Tests for the fallback / circuit-breaker middleware."""

import time

import pytest

from src.core.models import Message, PipelineRequest, Role
from src.core.state import PipelineState
from src.middleware.fallback import FallbackMiddleware, _CircuitState


# --- Circuit breaker unit tests ---

class TestCircuitState:
    def test_starts_closed(self):
        cs = _CircuitState(failure_threshold=3)
        assert cs.is_available is True
        assert cs.state == "closed"

    def test_opens_after_threshold(self):
        cs = _CircuitState(failure_threshold=2)
        cs.record_failure()
        assert cs.is_available is True  # 1 failure, not yet tripped
        cs.record_failure()
        assert cs.state == "open"
        assert cs.is_available is False

    def test_recovers_after_timeout(self):
        cs = _CircuitState(failure_threshold=1, recovery_timeout=0.1)
        cs.record_failure()
        assert cs.is_available is False

        time.sleep(0.15)
        assert cs.is_available is True  # moved to half_open
        assert cs.state == "half_open"

    def test_success_resets(self):
        cs = _CircuitState(failure_threshold=2)
        cs.record_failure()
        cs.record_failure()
        assert cs.state == "open"

        # Simulate recovery
        cs.state = "half_open"
        cs.record_success()
        assert cs.state == "closed"
        assert cs.failures == 0


# --- Middleware tests ---

def _make_state() -> PipelineState:
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="test")],
        )
    )


@pytest.mark.asyncio
async def test_sets_fallback_chain():
    mw = FallbackMiddleware(chain=["openai", "ollama", "gemini"])
    state = _make_state()
    state = await mw.process(state)

    assert state.extras["fallback_chain"] == ["openai", "ollama", "gemini"]
    assert state.request.target_adapter == "openai"


@pytest.mark.asyncio
async def test_skips_tripped_adapters():
    mw = FallbackMiddleware(chain=["openai", "ollama", "gemini"], failure_threshold=1)

    # Trip the openai circuit
    mw.record_failure("openai")

    state = _make_state()
    state = await mw.process(state)

    # openai should be skipped
    assert state.extras["fallback_chain"] == ["ollama", "gemini"]
    assert state.request.target_adapter == "ollama"


@pytest.mark.asyncio
async def test_disabled():
    mw = FallbackMiddleware(chain=["openai", "ollama"], enabled=False)
    state = _make_state()
    state = await mw.process(state)

    assert "fallback_chain" not in state.extras


@pytest.mark.asyncio
async def test_empty_chain_is_noop():
    mw = FallbackMiddleware(chain=[])
    state = _make_state()
    state = await mw.process(state)

    assert "fallback_chain" not in state.extras
