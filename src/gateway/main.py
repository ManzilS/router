"""Application entry point — uses the plugin registry to wire up
adapters, middleware, pipeline, orchestrator, and the FastAPI server.

No concrete adapter or middleware classes are imported here.
Everything is discovered from plugins.yaml.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.core.registry import PluginRegistry, load_plugins_config
from src.utils.config import Settings
from src.utils.errors import (
    AuthenticationError,
    RateLimitError,
    RequestTooLargeError,
    RouterError,
)
from src.utils.logging import request_id_var, request_start_var, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: log config.  Shutdown: close adapter HTTP clients."""
    settings: Settings = app.state.settings
    logger.info(
        "AI Router starting — dev_mode=%s, default_adapter=%s, adapters=%s",
        settings.dev_mode,
        settings.default_adapter,
        list(app.state.orchestrator.pipeline.adapters.keys()),
    )
    yield
    # Shutdown — close any persistent HTTP clients
    for name, adapter in app.state.orchestrator.pipeline.adapters.items():
        if hasattr(adapter, "close"):
            await adapter.close()
            logger.info("Closed adapter client: %s", name)
    logger.info("AI Router shut down")


# ---------------------------------------------------------------------------
# Middleware classes
# ---------------------------------------------------------------------------

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects request ID and timing into every request."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request_id_var.set(req_id)
        request_start_var.set(time.time())

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id

        elapsed = (time.time() - request_start_var.get()) * 1000
        logger.info(
            "%s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token auth — skipped when api_key is empty."""

    async def dispatch(self, request: Request, call_next):
        settings: Settings = request.app.state.settings
        if not settings.api_key:
            return await call_next(request)

        # Skip auth for health endpoints
        if request.url.path in ("/", "/v1/models"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {settings.api_key}":
            err = AuthenticationError("Invalid or missing API key")
            return JSONResponse(status_code=err.status_code, content=err.to_dict())

        return await call_next(request)


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed max_body_size."""

    async def dispatch(self, request: Request, call_next):
        settings: Settings = request.app.state.settings
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_body_size:
            err = RequestTooLargeError(
                f"Request body exceeds {settings.max_body_size} bytes"
            )
            return JSONResponse(status_code=err.status_code, content=err.to_dict())
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory per-IP rate limiter (requests per minute)."""

    def __init__(self, app, rpm: int = 0):
        super().__init__(app)
        self.rpm = rpm
        self._buckets: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        if self.rpm <= 0:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = now - 60.0

        # Prune old entries
        hits = self._buckets.get(client_ip, [])
        hits = [t for t in hits if t > window]
        hits.append(now)
        self._buckets[client_ip] = hits

        if len(hits) > self.rpm:
            err = RateLimitError("Rate limit exceeded")
            response = JSONResponse(
                status_code=err.status_code, content=err.to_dict()
            )
            response.headers["Retry-After"] = "60"
            return response

        return await call_next(request)


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

async def router_error_handler(request: Request, exc: RouterError) -> JSONResponse:
    """Convert any RouterError into a structured JSON response."""
    settings: Settings = request.app.state.settings
    content = exc.to_dict()
    if settings.dev_mode and exc.details:
        content["error"]["details"] = exc.details
    return JSONResponse(status_code=exc.status_code, content=content)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory that builds and returns a fully wired FastAPI application."""
    settings = settings or Settings()

    # --- Structured logging ---
    setup_logging(log_level=settings.log_level, dev_mode=settings.dev_mode)

    # --- Discover plugins ---
    plugins_config = load_plugins_config(settings.plugins_config)
    registry = PluginRegistry(config=plugins_config)
    registry.discover()

    if not registry.adapters:
        logger.warning(
            "No adapters enabled! Check plugins.yaml. "
            "Requests will fail until at least one adapter is enabled."
        )

    # --- Pipeline & Orchestrator ---
    pipeline = Pipeline(
        pre_middleware=registry.pre_middleware,
        post_middleware=registry.post_middleware,
        adapters=registry.adapters,
        default_adapter=settings.default_adapter,
    )
    orchestrator = Orchestrator(pipeline=pipeline, max_loops=settings.max_loops)

    # --- FastAPI app ---
    app = FastAPI(
        title="AI Router",
        description="Modular AI routing engine with pluggable adapters and middleware",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.dev_mode else None,
        redoc_url=None,
    )

    # Store on app state so routes can access it
    app.state.orchestrator = orchestrator
    app.state.settings = settings
    app.state.registry = registry

    # --- Exception handler ---
    app.add_exception_handler(RouterError, router_error_handler)

    # --- Middleware stack (order matters — outermost first) ---
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(BodyLimitMiddleware)

    if settings.rate_limit_rpm > 0:
        app.add_middleware(RateLimitMiddleware, rpm=settings.rate_limit_rpm)

    if settings.api_key:
        app.add_middleware(AuthMiddleware)

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.gateway.server import router
    app.include_router(router)

    return app


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
