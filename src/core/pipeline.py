"""The Pipeline engine — runs a PipelineState through an ordered list of
middleware nodes, then dispatches to an adapter, and runs post-processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.state import PipelineState
from src.utils.errors import AdapterError, AdapterNotFoundError, RouterError

if TYPE_CHECKING:
    from src.adapters.base import AdapterBase
    from src.middleware.base import MiddlewareBase

logger = logging.getLogger(__name__)


class Pipeline:
    """Ordered execution of pre-middleware -> adapter -> post-middleware."""

    def __init__(
        self,
        pre_middleware: list[MiddlewareBase] | None = None,
        post_middleware: list[MiddlewareBase] | None = None,
        adapters: dict[str, AdapterBase] | None = None,
        default_adapter: str = "openai",
    ) -> None:
        self.pre_middleware = pre_middleware or []
        self.post_middleware = post_middleware or []
        self.adapters: dict[str, AdapterBase] = adapters or {}
        self.default_adapter = default_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, state: PipelineState) -> PipelineState:
        """Run the full pipeline once (one pass, no looping — the
        Orchestrator handles loops)."""

        # --- Pre-processing middleware ---
        for mw in self.pre_middleware:
            logger.debug("Running pre-middleware: %s", mw.name)
            state = await mw.process(state)
            if state.early_exit:
                logger.info("Early exit triggered by %s", mw.name)
                return state

        # --- Adapter dispatch + post-processing ---
        return await self.dispatch(state)

    async def dispatch(self, state: PipelineState) -> PipelineState:
        """Dispatch to the adapter and run post-middleware.

        Separated from ``execute()`` so the orchestrator's fallback path
        can retry different adapters *without* re-running pre-middleware.
        """
        adapter_name = state.request.target_adapter or self.default_adapter
        adapter = self.adapters.get(adapter_name)
        if adapter is None:
            raise AdapterNotFoundError(
                f"No adapter registered for '{adapter_name}'. "
                f"Available: {list(self.adapters.keys())}",
                adapter=adapter_name,
            )

        logger.debug("Dispatching to adapter: %s", adapter_name)
        try:
            state.response = await adapter.send(state.request)
        except RouterError:
            raise  # Already classified — propagate as-is
        except Exception as exc:
            raise AdapterError(
                f"Adapter '{adapter_name}' failed: {exc}",
                adapter=adapter_name,
            ) from exc
        state.response.request_id = state.request.id

        # --- Post-processing middleware ---
        for mw in self.post_middleware:
            logger.debug("Running post-middleware: %s", mw.name)
            state = await mw.process(state)
            if state.early_exit:
                logger.info("Early exit triggered by %s (post)", mw.name)
                return state

        return state
