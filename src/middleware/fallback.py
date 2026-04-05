"""Fallback / Circuit-Breaker middleware — tries adapters in priority order.

Unlike the fan-out router (which sends to ALL targets simultaneously and
costs N× compute), the fallback tries adapters **sequentially**:

    Try A → if it fails → try B → if it fails → try C

This costs 1× compute in the happy path.  It also tracks adapter health
so that a consistently failing adapter gets temporarily skipped (circuit
breaker pattern).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.state import PipelineState
from src.middleware.base import MiddlewareBase

logger = logging.getLogger(__name__)

# Circuit breaker states
_CLOSED = "closed"      # healthy — requests flow through
_OPEN = "open"          # tripped — skip this adapter
_HALF_OPEN = "half_open"  # cooling down — allow one probe request


class _CircuitState:
    """Tracks health for a single adapter."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: int = 0
        self.state: str = _CLOSED
        self.last_failure_time: float = 0.0

    @property
    def is_available(self) -> bool:
        if self.state == _CLOSED:
            return True
        if self.state == _OPEN:
            # Check if recovery timeout has passed → move to half-open
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = _HALF_OPEN
                return True
            return False
        # half_open — allow one probe
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = _CLOSED

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = _OPEN
            logger.warning(
                "Circuit breaker OPEN after %d failures (recovery in %ds)",
                self.failures,
                self.recovery_timeout,
            )


class FallbackMiddleware(MiddlewareBase):
    """Pre-processing middleware that sets up sequential fallback targets.

    When active, this replaces the single ``target_adapter`` with a
    fallback chain stored in ``state.extras["fallback_chain"]``.  The
    pipeline itself stays unchanged — the **orchestrator** or a patched
    pipeline dispatch consumes the chain.

    For simplicity, this middleware patches ``target_adapter`` to the
    first *available* adapter in the chain.  If the pipeline execution
    fails, the orchestrator can call ``advance_fallback()`` to try the
    next one.
    """

    name = "fallback"

    def __init__(
        self,
        chain: list[str] | None = None,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        enabled: bool = True,
    ) -> None:
        self.chain = chain or []
        self.enabled = enabled
        self._circuits: dict[str, _CircuitState] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

    def _get_circuit(self, adapter_name: str) -> _CircuitState:
        if adapter_name not in self._circuits:
            self._circuits[adapter_name] = _CircuitState(
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
        return self._circuits[adapter_name]

    def get_available_chain(self) -> list[str]:
        """Return only the adapters whose circuits are not open."""
        return [a for a in self.chain if self._get_circuit(a).is_available]

    def record_success(self, adapter_name: str) -> None:
        self._get_circuit(adapter_name).record_success()

    def record_failure(self, adapter_name: str) -> None:
        self._get_circuit(adapter_name).record_failure()

    async def process(self, state: PipelineState) -> PipelineState:
        if not self.enabled or state.response is not None:
            return state

        # Allow per-request override via metadata
        chain = state.request.metadata.get("fallback_chain", self.chain)
        if not chain:
            return state

        available = [a for a in chain if self._get_circuit(a).is_available]
        if not available:
            logger.error("All fallback adapters have tripped circuits!")
            # Reset to let at least the first one through
            available = chain[:1]

        # Store the full chain for the orchestrator to consume
        state.extras["fallback_chain"] = list(available)
        state.extras["fallback_index"] = 0

        # Point to the first available adapter
        state.request.target_adapter = available[0]
        logger.info("Fallback chain: %s (starting with %s)", available, available[0])

        return state
