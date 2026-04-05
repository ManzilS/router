"""Universal Internal Format — the lingua franca of the gateway.

Every incoming request is immediately converted into a ``PipelineRequest``
and every outgoing response starts life as a ``PipelineResponse``.
Adapters and middleware only ever touch these types.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message-level models
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string, matching OpenAI convention


# ---------------------------------------------------------------------------
# Request / Response envelopes
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    """The universal representation of an inbound chat-completion request."""

    id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    messages: list[Message]
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None

    # --- Gateway bookkeeping (not sent to the AI) ---
    target_adapter: str = "openai"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PipelineResponse(BaseModel):
    """The universal representation of an outbound chat-completion response."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    request_id: str = ""
    choices: list[Choice] = Field(default_factory=list)
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Convenience helpers -----------------------------------------------
    @property
    def text(self) -> str:
        """Return the first choice's content, or empty string."""
        if self.choices:
            return self.choices[0].message.content
        return ""

    @property
    def has_tool_calls(self) -> bool:
        if self.choices:
            tc = self.choices[0].message.tool_calls
            return tc is not None and len(tc) > 0
        return False


# Forward-ref resolution (ToolCall referenced inside Message)
Message.model_rebuild()
