"""Early-exit middleware — intercepts simple queries that don't need an AI
and returns instant local answers."""

from __future__ import annotations

import datetime
import logging
import platform
import re

from src.core.models import Choice, Message, PipelineResponse, Role
from src.core.state import PipelineState
from src.middleware.base import PreMiddleware

logger = logging.getLogger(__name__)


def _make_response(text: str, request_id: str) -> PipelineResponse:
    return PipelineResponse(
        request_id=request_id,
        model="local/early-exit",
        choices=[
            Choice(
                message=Message(role=Role.ASSISTANT, content=text),
                finish_reason="stop",
            )
        ],
    )


# Simple pattern matchers
_TIME_PATTERNS = re.compile(
    r"\b(what\s+time|current\s+time|what\'s\s+the\s+time|tell\s+me\s+the\s+time)\b",
    re.IGNORECASE,
)
_DATE_PATTERNS = re.compile(
    r"\b(what\s+date|today\'s\s+date|current\s+date|what\s+day)\b",
    re.IGNORECASE,
)
_PING_PATTERNS = re.compile(
    r"^(ping|hello|hi|hey|test)[\s!?.]*$",
    re.IGNORECASE,
)
_MATH_PATTERN = re.compile(
    r"^(?:what\s+is\s+|calculate\s+)?(\d[\d\s\+\-\*\/\.\(\)]+)[\s?]*$",
    re.IGNORECASE,
)


class EarlyExitMiddleware(PreMiddleware):
    """Pre-processing middleware that handles trivial queries locally."""

    name = "early_exit"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def process(self, state: PipelineState) -> PipelineState:
        if not self.enabled or state.response is not None:
            return state

        # Grab the last user message
        user_msgs = [m for m in state.request.messages if m.role == Role.USER]
        if not user_msgs:
            return state
        text = user_msgs[-1].content.strip()

        answer: str | None = None

        if _TIME_PATTERNS.search(text):
            now = datetime.datetime.now()
            answer = f"The current time is {now.strftime('%I:%M %p')}."

        elif _DATE_PATTERNS.search(text):
            today = datetime.date.today()
            answer = f"Today is {today.strftime('%A, %B %d, %Y')}."

        elif _PING_PATTERNS.match(text):
            answer = "Pong! The gateway is running."

        elif _MATH_PATTERN.match(text):
            expr = _MATH_PATTERN.match(text).group(1).strip()  # type: ignore[union-attr]
            try:
                # Safe eval: only allow digits and basic math operators
                if re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", expr):
                    result = eval(expr)  # noqa: S307 — validated safe subset
                    answer = f"The result is {result}."
            except Exception:
                pass  # Fall through to AI

        if answer:
            state.response = _make_response(answer, state.request.id)
            state.early_exit = True
            logger.info("Early exit: handled '%s' locally", text[:60])

        return state
