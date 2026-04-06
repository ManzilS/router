"""Application entry point — uses the plugin registry to wire up
adapters, middleware, pipeline, orchestrator, and the FastAPI server.

No concrete adapter or middleware classes are imported here.
Everything is discovered from plugins.yaml.
"""

from __future__ import annotations

import logging
import sys

import uvicorn
from fastapi import FastAPI

from src.core.orchestrator import Orchestrator
from src.core.pipeline import Pipeline
from src.core.registry import PluginRegistry, load_plugins_config
from src.utils.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory that builds and returns a fully wired FastAPI application."""
    settings = settings or Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    # --- Discover plugins ---
    plugins_config = load_plugins_config(settings.plugins_config)
    registry = PluginRegistry(config=plugins_config)
    registry.discover()

    if not registry.adapters:
        logging.warning(
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
    )

    # Store on app state so routes can access it
    app.state.orchestrator = orchestrator
    app.state.settings = settings
    app.state.registry = registry

    from src.gateway.server import router
    app.include_router(router)

    return app


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
