"""The Orchestrator — the loop controller that manages multi-step
interactions where an AI response triggers further processing.

It wraps the Pipeline and handles:
  1. Fan-out routing (send to multiple adapters simultaneously)
  2. Fallback routing (try adapters sequentially until one works)
  3. Tool-call loops (AI asks for a tool -> middleware runs it -> loop back)
  4. Max-loop safety limits
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
            # Always run pre-middleware first (sets up fallback_chain, etc.)
            state = await self._run_pre_middleware(state)
            if state.early_exit:
                return state

            # --- Fan-out routing ---
            if state.fan_out_targets:
                state = await self._fan_out(state)
            # --- Fallback routing ---
            elif "fallback_chain" in state.extras:
                state = await self._fallback(state)
            else:
                state = await self.pipeline.dispatch(state)

            if state.early_exit:
                return state

            # --- Check if the AI wants to loop (tool calls) ---
            if state.response and state.response.has_tool_calls:
                state.should_loop = True

            if not state.can_loop:
                break

            # Inject tool results back into the conversation and loop
            state = self._prepare_next_loop(state)
            state.increment_loop()
            logger.info(
                "Orchestrator loop %d/%d", state.loop_count, state.max_loops
            )

        return state

    # ------------------------------------------------------------------
    async def _run_pre_middleware(self, state: PipelineState) -> PipelineState:
        """Run pre-middleware only (early exit, fallback setup, etc.)."""
        for mw in self.pipeline.pre_middleware:
            state = await mw.process(state)
            if state.early_exit:
                return state
        return state

    # ------------------------------------------------------------------
    async def _fallback(self, state: PipelineState) -> PipelineState:
        """Try adapters sequentially until one succeeds.

        Uses ``pipeline.dispatch()`` (not ``execute()``) so pre-middleware
        is NOT re-run on each retry — it already ran once above.
        """
        chain: list[str] = state.extras.pop("fallback_chain")
        idx: int = state.extras.pop("fallback_index", 0)

        # Find the fallback middleware to report success/failure
        fallback_mw = None
        for mw in self.pipeline.pre_middleware:
            if mw.name == "fallback":
                fallback_mw = mw
                break

        last_error: Exception | None = None
        for i in range(idx, len(chain)):
            adapter_name = chain[i]
            state.request.target_adapter = adapter_name
            logger.info("Fallback: trying adapter '%s' (%d/%d)", adapter_name, i + 1, len(chain))
            try:
                state = await self.pipeline.dispatch(state)
                if fallback_mw is not None:
                    fallback_mw.record_success(adapter_name)
                return state
            except Exception as exc:
                logger.warning("Fallback: adapter '%s' failed: %s", adapter_name, exc)
                last_error = exc
                if fallback_mw is not None:
                    fallback_mw.record_failure(adapter_name)
                # Reset response for next attempt
                state.response = None
                continue

        # All adapters in the chain failed
        if last_error is not None:
            raise last_error
        raise RuntimeError("Fallback chain exhausted with no adapters")

    # ------------------------------------------------------------------
    async def _fan_out(self, state: PipelineState) -> PipelineState:
        """Send the same request to multiple adapters concurrently."""
        targets = state.fan_out_targets
        state.fan_out_targets = []  # consume them

        async def _dispatch(adapter_name: str) -> PipelineResponse:
            branch = copy.deepcopy(state)
            branch.request.target_adapter = adapter_name
            branch = await self.pipeline.execute(branch)
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
