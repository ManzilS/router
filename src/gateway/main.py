"""Application entry point — wires up adapters, middleware, pipeline,
orchestrator, and starts the FastAPI server."""

from __future__ import annotations

import logging
import sys

import uvicorn
from fastapi import FastAPI

from src.adapters.gemini_ext import GeminiAdapter
from src.adapters.ollama_ext import OllamaAdapter
from src.adapters.openai_ext import OpenAIAdapter
from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.middleware.early_exit import EarlyExitMiddleware
from src.middleware.fallback import FallbackMiddleware
from src.middleware.logger import LoggerMiddleware
from src.middleware.rag_injector import RAGInjectorMiddleware
from src.middleware.router import RouterMiddleware
from src.utils.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory that builds and returns a fully wired FastAPI application."""
    settings = settings or Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    # --- Adapters ---
    adapters = {
        "openai": OpenAIAdapter(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        ),
        "ollama": OllamaAdapter(base_url=settings.ollama_base_url),
        "gemini": GeminiAdapter(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        ),
    }

    # --- Middleware ---
    pre_middleware = []
    post_middleware = []

    if settings.enable_early_exit:
        pre_middleware.append(EarlyExitMiddleware())

    pre_middleware.append(RouterMiddleware())

    # Fallback: if enabled, try adapters in order before giving up
    if settings.fallback_chain:
        pre_middleware.append(
            FallbackMiddleware(
                chain=settings.fallback_chain,
                failure_threshold=settings.fallback_failure_threshold,
                recovery_timeout=settings.fallback_recovery_timeout,
            )
        )

    if settings.enable_rag:
        rag = RAGInjectorMiddleware(persist_dir=settings.chroma_path)
        pre_middleware.append(rag)
        post_middleware.append(rag)  # also runs post to ingest

    if settings.enable_logger:
        post_middleware.append(LoggerMiddleware(db_path=settings.db_path))

    # --- Pipeline & Orchestrator ---
    pipeline = Pipeline(
        pre_middleware=pre_middleware,
        post_middleware=post_middleware,
        adapters=adapters,
        default_adapter=settings.default_adapter,
    )
    orchestrator = Orchestrator(pipeline=pipeline, max_loops=settings.max_loops)

    # --- FastAPI app ---
    app = FastAPI(
        title="AI Gateway Brain",
        description="Central nervous system for all your AI interactions",
        version="0.1.0",
    )

    # Store orchestrator on app state so routes can access it
    app.state.orchestrator = orchestrator
    app.state.settings = settings

    from src.gateway.server import router
    app.include_router(router)

    return app


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
