"""Listen-mode middleware — silently logs every request/response pair to a
local SQLite database so you can build your RAG brain over time."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

from src.core.state import PipelineState
from src.middleware.base import PostMiddleware

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".ai-router" / "history.db"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    model       TEXT,
    messages    TEXT,      -- JSON of the full message list
    response    TEXT,      -- assistant reply text
    metadata    TEXT,      -- JSON blob
    created_at  REAL
);
"""


class LoggerMiddleware(PostMiddleware):
    """Post-processing middleware that writes traffic to SQLite."""

    name = "logger"

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._initialized = False

    async def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(CREATE_TABLE)
            await db.commit()
        self._initialized = True

    async def process(self, state: PipelineState) -> PipelineState:
        # Only log when we have a response (post-processing pass)
        if state.response is None:
            return state

        if not self._initialized:
            await self._ensure_db()

        try:
            messages_json = json.dumps(
                [m.model_dump(mode="json") for m in state.request.messages]
            )
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(
                    """
                    INSERT INTO conversation_log
                        (request_id, model, messages, response, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state.request.id,
                        state.request.model,
                        messages_json,
                        state.response.text,
                        json.dumps(state.request.metadata),
                        state.request.created_at,
                    ),
                )
                await db.commit()
            logger.debug("Logged conversation %s", state.request.id)
        except Exception:
            logger.exception("Failed to log conversation")

        return state
