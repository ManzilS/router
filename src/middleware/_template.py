"""
=============================================================================
MIDDLEWARE TEMPLATE — Copy this file to create a new middleware node.
=============================================================================

Steps:
  1. Copy this file:  cp _template.py myfeature.py
  2. Decide if it's Pre or Post middleware (see below)
  3. Implement the process() method
  4. Add an entry to plugins.yaml:

     middleware:
       pre:   # or `post:`
         - name: myfeature
           enabled: true
           module: src.middleware.myfeature
           settings:
             some_option: "value"

  That's it — the plugin registry handles the rest.

CHOOSING Pre vs Post:
  - PreMiddleware  — runs BEFORE the AI.  Use for: prompt modification,
                     routing, validation, early exits (answer without AI).
  - PostMiddleware — runs AFTER the AI.   Use for: logging, caching,
                     response filtering, analytics, database writes.

The __init__ kwargs must match the `settings:` keys in plugins.yaml.
=============================================================================
"""

from __future__ import annotations

import logging

from src.core.models import Choice, Message, PipelineResponse, Role
from src.core.state import PipelineState
from src.middleware.base import PreMiddleware, PostMiddleware

logger = logging.getLogger(__name__)


# ---- Example Pre-Middleware ------------------------------------------------

class TemplatePreMiddleware(PreMiddleware):
    """Pre-middleware template — runs before the AI adapter."""

    name = "template_pre"

    def __init__(self, some_option: str = "default") -> None:
        # These kwargs come from plugins.yaml `settings:` block
        self.some_option = some_option

    async def process(self, state: PipelineState) -> PipelineState:
        # You have access to the full state:
        #   state.request          — the PipelineRequest (messages, model, etc.)
        #   state.request.messages — the conversation so far
        #   state.request.metadata — dict for passing data between middleware
        #   state.extras           — dict for middleware-to-orchestrator comms

        # Example: modify the last user message
        # for m in reversed(state.request.messages):
        #     if m.role == Role.USER:
        #         m.content += " (enhanced by my middleware)"
        #         break

        # Example: early exit (answer without calling the AI)
        # state.response = PipelineResponse(
        #     model="local/myfeature",
        #     choices=[Choice(message=Message(
        #         role=Role.ASSISTANT, content="I handled this locally!"
        #     ), finish_reason="stop")],
        # )
        # state.early_exit = True

        return state


# ---- Example Post-Middleware -----------------------------------------------

class TemplatePostMiddleware(PostMiddleware):
    """Post-middleware template — runs after the AI responds."""

    name = "template_post"

    def __init__(self) -> None:
        pass

    async def process(self, state: PipelineState) -> PipelineState:
        # state.response is populated at this point
        # You can read/modify it:
        #
        #   text = state.response.text        — the assistant's reply
        #   model = state.response.model      — which model answered
        #   usage = state.response.usage      — token counts
        #
        # Example: log the exchange
        # logger.info("Model %s replied: %s", state.response.model, state.response.text[:80])

        return state
