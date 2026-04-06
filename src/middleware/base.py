"""Abstract base classes for middleware nodes.

There are two distinct types of middleware, each with a clear contract:

- **PreMiddleware** — runs BEFORE the AI adapter is called.
  Use for: prompt injection, routing decisions, early exits, validation.

- **PostMiddleware** — runs AFTER the AI responds.
  Use for: logging, caching, response filtering, analytics.

Both receive a ``PipelineState`` and return it (possibly mutated).
"""

from __future__ import annotations

import abc

from src.core.state import PipelineState


class MiddlewareBase(abc.ABC):
    """Common contract for all middleware."""

    name: str = "base_middleware"
    phase: str = "unknown"  # "pre" or "post" — set by subclasses

    @abc.abstractmethod
    async def process(self, state: PipelineState) -> PipelineState:
        """Inspect / mutate the pipeline state and return it."""


class PreMiddleware(MiddlewareBase):
    """Middleware that runs before the adapter dispatch.

    To trigger an early exit (skip the AI entirely), set
    ``state.early_exit = True`` and populate ``state.response``.
    """

    phase = "pre"

    @abc.abstractmethod
    async def process(self, state: PipelineState) -> PipelineState: ...


class PostMiddleware(MiddlewareBase):
    """Middleware that runs after the adapter returns a response.

    ``state.response`` is guaranteed to be populated when this runs
    (unless a previous post-middleware triggered early exit).
    """

    phase = "post"

    @abc.abstractmethod
    async def process(self, state: PipelineState) -> PipelineState: ...
