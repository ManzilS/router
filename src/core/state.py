"""Pipeline state tracker — monitors loop iterations and carries context
across orchestration cycles."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.core.models import PipelineRequest, PipelineResponse


class PipelineState(BaseModel):
    """Mutable state object that travels with a request through the pipeline."""

    request: PipelineRequest
    response: PipelineResponse | None = None

    # Orchestration
    loop_count: int = 0
    max_loops: int = 3
    should_loop: bool = False
    early_exit: bool = False

    # Accumulator for tool results injected back into the conversation
    pending_tool_results: list[dict[str, Any]] = Field(default_factory=list)

    # Routing — filled by the router middleware when fan-out is needed
    fan_out_targets: list[str] = Field(default_factory=list)
    fan_out_responses: list[PipelineResponse] = Field(default_factory=list)

    # Arbitrary bag for middleware to stash data
    extras: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def can_loop(self) -> bool:
        return self.should_loop and self.loop_count < self.max_loops

    def increment_loop(self) -> None:
        self.loop_count += 1
        self.should_loop = False
