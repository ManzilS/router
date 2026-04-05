"""Abstract base class for all middleware nodes."""

from __future__ import annotations

import abc

from src.core.state import PipelineState


class MiddlewareBase(abc.ABC):
    """Every middleware node must implement ``process``."""

    name: str = "base_middleware"

    @abc.abstractmethod
    async def process(self, state: PipelineState) -> PipelineState:
        """Inspect / mutate the pipeline state and return it.

        To trigger an early exit (skip remaining nodes + adapter), set
        ``state.early_exit = True`` and populate ``state.response``.
        """
