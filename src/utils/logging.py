"""Structured logging setup with request ID context.

In production, logs are JSON.  In dev mode, logs are human-readable.
Every log line includes the request_id when one is set in context.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# Context variable for per-request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
request_start_var: ContextVar[float] = ContextVar("request_start", default=0.0)


def get_request_id() -> str:
    return request_id_var.get()


def get_request_elapsed_ms() -> float:
    start = request_start_var.get()
    if start == 0.0:
        return 0.0
    return (time.time() - start) * 1000


class StructuredFormatter(logging.Formatter):
    """JSON log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        elapsed = get_request_elapsed_ms()
        if elapsed > 0:
            log_entry["elapsed_ms"] = round(elapsed, 1)
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(log_entry)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for dev mode."""

    def format(self, record: logging.LogRecord) -> str:
        elapsed = get_request_elapsed_ms()
        elapsed_str = f" [{elapsed:.0f}ms]" if elapsed > 0 else ""
        req_id = request_id_var.get()
        req_str = f" req={req_id}" if req_id != "-" else ""

        base = (
            f"{self.formatTime(record)} | {record.levelname:<8s} | "
            f"{record.name}{req_str}{elapsed_str} | {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(log_level: str = "INFO", dev_mode: bool = False) -> None:
    """Configure the root logger."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if dev_mode:
        handler.setFormatter(DevFormatter())
    else:
        handler.setFormatter(StructuredFormatter())

    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
