"""FastAPI route definitions — mimics the OpenAI API surface so any
client that supports a custom base URL can talk to us."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from src.core.models import (
    Choice,
    Message,
    PipelineRequest,
    PipelineResponse,
    Role,
    Usage,
)
from src.core.state import PipelineState

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Health / discovery endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    return {"status": "ok", "service": "ai-gateway-brain"}


@router.get("/v1/models")
async def list_models(request: Request):
    """Return the list of registered adapters as 'models'."""
    orchestrator = request.app.state.orchestrator
    adapters = orchestrator.pipeline.adapters
    models = [
        {
            "id": name,
            "object": "model",
            "owned_by": "ai-gateway-brain",
        }
        for name in adapters
    ]
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Main chat completions endpoint
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    orchestrator = request.app.state.orchestrator

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Convert incoming OpenAI-format request to our Universal Format
    pipeline_req = _parse_openai_request(body)

    # ----- STREAMING PATH -----
    if pipeline_req.stream:
        return await _handle_streaming(pipeline_req, orchestrator)

    # ----- NON-STREAMING PATH -----
    state = PipelineState(request=pipeline_req)

    try:
        state = await orchestrator.run(state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=502, detail="Upstream AI request failed")

    if state.response is None:
        raise HTTPException(status_code=502, detail="No response from AI")

    return JSONResponse(content=_format_openai_response(state.response))


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------

async def _handle_streaming(
    pipeline_req: PipelineRequest, orchestrator
) -> StreamingResponse:
    """Run pre-middleware, then stream from the adapter, then run
    post-middleware asynchronously after the stream finishes."""

    pipeline = orchestrator.pipeline
    state = PipelineState(request=pipeline_req)

    # Run pre-middleware (RAG injection, early exit, etc.)
    for mw in pipeline.pre_middleware:
        state = await mw.process(state)
        if state.early_exit:
            # Early exit still needs to respond — convert to a fake stream
            return StreamingResponse(
                _early_exit_stream(state),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    # Resolve adapter
    adapter_name = state.request.target_adapter or pipeline.default_adapter
    adapter = pipeline.adapters.get(adapter_name)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"No adapter registered for '{adapter_name}'",
        )

    async def _stream_and_log():
        """Yield SSE chunks to the client, accumulate the full text,
        then run post-middleware (logging, RAG ingest) after done."""
        full_text = []
        try:
            async for chunk in adapter.stream(state.request):
                full_text.append(_extract_content_from_sse(chunk))
                yield chunk
        except Exception as exc:
            logger.exception("Streaming failed from %s", adapter_name)
            error_chunk = {
                "id": state.request.id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": state.request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n[Gateway Error: {exc}]"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # --- Post-stream: run post-middleware for logging/RAG ---
        assembled_text = "".join(full_text)
        state.response = PipelineResponse(
            request_id=state.request.id,
            model=state.request.model,
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content=assembled_text),
                    finish_reason="stop",
                )
            ],
        )
        for mw in pipeline.post_middleware:
            try:
                await mw.process(state)
            except Exception:
                logger.exception("Post-stream middleware %s failed", mw.name)

    return StreamingResponse(
        _stream_and_log(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _early_exit_stream(state: PipelineState):
    """Convert an early-exit response into SSE chunks for streaming clients."""
    text = state.response.text if state.response else ""
    chunk = {
        "id": state.request.id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "local/early-exit",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _extract_content_from_sse(chunk: str) -> str:
    """Pull the text content out of an SSE data line for accumulation."""
    line = chunk.strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        return ""
    try:
        obj = json.loads(line[5:].strip())
        choices = obj.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_openai_request(body: dict[str, Any]) -> PipelineRequest:
    messages = []
    for m in body.get("messages", []):
        messages.append(
            Message(
                role=Role(m.get("role", "user")),
                content=m.get("content", ""),
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
            )
        )

    # Allow clients to specify target adapter via model name or header
    model = body.get("model", "default")
    target = "openai"  # default
    if "/" in model:
        # Convention: "ollama/llama3" means use the ollama adapter with model llama3
        parts = model.split("/", 1)
        target = parts[0]
        model = parts[1]

    metadata: dict[str, Any] = {}
    if "fan_out" in body:
        metadata["fan_out"] = body["fan_out"]

    return PipelineRequest(
        messages=messages,
        model=model,
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens"),
        stream=body.get("stream", False),
        stop=body.get("stop"),
        target_adapter=target,
        metadata=metadata,
    )


def _format_openai_response(resp: PipelineResponse) -> dict[str, Any]:
    choices = []
    for c in resp.choices:
        choice: dict[str, Any] = {
            "index": c.index,
            "message": {
                "role": c.message.role.value,
                "content": c.message.content,
            },
            "finish_reason": c.finish_reason,
        }
        if c.message.tool_calls:
            choice["message"]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in c.message.tool_calls
            ]
        choices.append(choice)

    return {
        "id": resp.id,
        "object": "chat.completion",
        "created": int(resp.created_at),
        "model": resp.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }
