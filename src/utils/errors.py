"""Structured error types for the router.

Every layer should raise these instead of raw exceptions so the router
can return meaningful error responses to clients.
"""

from __future__ import annotations


class RouterError(Exception):
    """Base for all router errors."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(self, message: str, *, adapter: str = "", details: str = "") -> None:
        self.message = message
        self.adapter = adapter
        self.details = details
        super().__init__(message)

    def to_dict(self) -> dict:
        d = {"error": {"type": self.error_type, "message": self.message}}
        if self.adapter:
            d["error"]["adapter"] = self.adapter
        if self.details:
            d["error"]["details"] = self.details
        return d


class AdapterError(RouterError):
    """An adapter failed to communicate with its upstream AI."""

    status_code = 502
    error_type = "adapter_error"


class AdapterTimeoutError(AdapterError):
    """The upstream AI did not respond in time."""

    status_code = 504
    error_type = "adapter_timeout"


class AdapterAuthError(AdapterError):
    """Authentication with the upstream AI failed (401/403)."""

    status_code = 401
    error_type = "adapter_auth_error"


class AdapterRateLimitError(AdapterError):
    """The upstream AI rate-limited us (429)."""

    status_code = 429
    error_type = "adapter_rate_limited"


class AdapterNotFoundError(RouterError):
    """The requested adapter does not exist."""

    status_code = 400
    error_type = "adapter_not_found"


class RequestValidationError(RouterError):
    """The incoming request is malformed or invalid."""

    status_code = 400
    error_type = "validation_error"


class RequestTooLargeError(RouterError):
    """The request body exceeds the configured limit."""

    status_code = 413
    error_type = "request_too_large"


class AuthenticationError(RouterError):
    """The client did not provide valid authentication."""

    status_code = 401
    error_type = "authentication_error"


class RateLimitError(RouterError):
    """The client has exceeded the rate limit."""

    status_code = 429
    error_type = "rate_limit_exceeded"
