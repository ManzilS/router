"""Tests for the logger middleware."""

import tempfile
from pathlib import Path

import aiosqlite
import pytest

from src.core.models import Choice, Message, PipelineRequest, PipelineResponse, Role
from src.core.state import PipelineState
from src.middleware.logger import LoggerMiddleware


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_history.db"


@pytest.fixture
def middleware(tmp_db):
    return LoggerMiddleware(db_path=tmp_db)


def _state_with_response() -> PipelineState:
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="Hello")],
            model="test-model",
        ),
        response=PipelineResponse(
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content="Hi there!"),
                    finish_reason="stop",
                )
            ],
            model="test-model",
        ),
    )


@pytest.mark.asyncio
async def test_logs_conversation(middleware, tmp_db):
    state = _state_with_response()
    await middleware.process(state)

    async with aiosqlite.connect(str(tmp_db)) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM conversation_log")
        row = await cursor.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_skips_when_no_response(middleware, tmp_db):
    state = PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content="Hello")]
        )
    )
    await middleware.process(state)

    # DB should not even be created
    assert not tmp_db.exists()
