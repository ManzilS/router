"""Router middleware — duplicates a request to multiple adapters for fan-out.

When the ``fan_out_targets`` list on the state is populated (e.g. by config or
a previous middleware), this middleware flags the orchestrator to dispatch
to each target and collect all responses.
"""

from __future__ import annotations

import logging

from src.core.state import PipelineState
from src.middleware.base import PreMiddleware

logger = logging.getLogger(__name__)


class RouterMiddleware(PreMiddleware):
    """Pre-processing middleware that sets up fan-out targets."""

    name = "router"

    def __init__(self, targets: list[str] | None = None) -> None:
        self.default_targets = targets or []

    async def process(self, state: PipelineState) -> PipelineState:
        # Only act during pre-processing (no response yet)
        if state.response is not None:
            return state

        # Use metadata override or fall back to configured defaults
        targets = state.request.metadata.get("fan_out", self.default_targets)
        if targets and len(targets) > 1:
            state.fan_out_targets = list(targets)
            logger.info("Router: fan-out to %s", state.fan_out_targets)

        return state
