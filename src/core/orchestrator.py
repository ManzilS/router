"""The Orchestrator — the loop controller that manages multi-step
interactions where an AI response triggers further processing.

It wraps the Pipeline and handles:
  1. Fan-out routing (send to multiple adapters simultaneously)
  2. Tool-call loops (AI asks for a tool -> middleware runs it -> loop back)
  3. Max-loop safety limits

The orchestrator is intentionally generic — it doesn't know the names
of any specific middleware.  All coordination happens through the
``PipelineState`` object (extras, fan_out_targets, etc.).
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import TYPE_CHECKING

from src.core.models import Message, PipelineResponse, Role
from src.core.state import PipelineState

if TYPE_CHECKING:
    from src.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, pipeline: Pipeline, max_loops: int = 3) -> None:
        self.pipeline = pipeline
        self.max_loops = max_loops

    async def run(self, state: PipelineState) -> PipelineState:
        """Execute the full orchestration cycle."""
        state.max_loops = self.max_loops

        while True:
            # Run pre-middleware (routing decisions, early exits, etc.)
            state = await self._run_pre_middleware(state)
            if state.early_exit:
                return state

            # Fan-out or single dispatch
            if state.fan_out_targets:
                state = await self._fan_out(state)
            else:
                state = await self.pipeline.dispatch(state)

            if state.early_exit:
                return state

            # Check if the AI wants to loop (tool calls)
            if state.response and state.response.has_tool_calls:
                state.should_loop = True

            if not state.can_loop:
                break

            # Inject tool results back and loop
            state = self._prepare_next_loop(state)
            state.increment_loop()
            logger.info(
                "Orchestrator loop %d/%d", state.loop_count, state.max_loops
            )

        return state

    # ------------------------------------------------------------------
    async def _run_pre_middleware(self, state: PipelineState) -> PipelineState:
        """Run all pre-middleware nodes."""
        for mw in self.pipeline.pre_middleware:
            state = await mw.process(state)
            if state.early_exit:
                return state
        return state

    # ------------------------------------------------------------------
    async def _fan_out(self, state: PipelineState) -> PipelineState:
        """Send the same request to multiple adapters concurrently."""
        targets = state.fan_out_targets
        state.fan_out_targets = []  # consume them

        async def _dispatch(adapter_name: str) -> PipelineResponse:
            branch = copy.deepcopy(state)
            branch.request.target_adapter = adapter_name
            branch = await self.pipeline.dispatch(branch)
            return branch.response  # type: ignore[return-value]

        results = await asyncio.gather(
            *[_dispatch(t) for t in targets],
            return_exceptions=True,
        )

        for target, result in zip(targets, results):
            if isinstance(result, Exception):
                logger.error("Fan-out to %s failed: %s", target, result)
            else:
                state.fan_out_responses.append(result)

        # Use the first successful response as the primary
        for resp in state.fan_out_responses:
            state.response = resp
            break

        return state

    # ------------------------------------------------------------------
    def _prepare_next_loop(self, state: PipelineState) -> PipelineState:
        """Append the assistant's tool-call message and any pending tool
        results to the conversation, then clear the response for the next
        pass through the pipeline."""
        if state.response is None:
            return state

        # Add the assistant message with tool calls
        assistant_msg = state.response.choices[0].message
        state.request.messages.append(assistant_msg)

        # Add any pending tool results as tool-role messages
        for result in state.pending_tool_results:
            state.request.messages.append(
                Message(
                    role=Role.TOOL,
                    content=result.get("content", ""),
                    tool_call_id=result.get("tool_call_id", ""),
                )
            )
        state.pending_tool_results.clear()

        # Clear response for the next pass
        state.response = None
        return state
