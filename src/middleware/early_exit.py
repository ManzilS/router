"""Early-exit middleware — intercepts simple queries that don't need an AI
and returns instant local answers."""

from __future__ import annotations

import ast
import datetime
import logging
import operator
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

# Safe AST-based math evaluator — no eval()
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_math_eval(expr: str) -> float | int | None:
    """Evaluate a simple arithmetic expression using the AST.

    Only supports numbers and +, -, *, / operators.  Returns None if
    the expression contains anything unexpected.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval_node(node: ast.expr) -> float | int:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError("Unsupported operator")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return op_fn(left, right)
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError("Unsupported operator")
            return op_fn(_eval_node(node.operand))
        raise ValueError("Unsupported node")

    try:
        return _eval_node(tree)
    except (ValueError, ZeroDivisionError):
        return None


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
                if re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", expr):
                    result = _safe_math_eval(expr)
                    if result is not None:
                        answer = f"The result is {result}."
            except Exception:
                pass  # Fall through to AI

        if answer:
            state.response = _make_response(answer, state.request.id)
            state.early_exit = True
            logger.info("Early exit: handled '%s' locally", text[:60])

        return state
