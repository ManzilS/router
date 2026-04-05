"""Tests for the Gemini adapter — translation logic only."""

import pytest

from src.adapters.gemini_ext import GeminiAdapter
from src.core.models import Message, PipelineRequest, Role


@pytest.fixture
def adapter():
    return GeminiAdapter(api_key="fake-key", model="gemini-2.0-flash")


@pytest.fixture
def sample_request():
    return PipelineRequest(
        messages=[
            Message(role=Role.SYSTEM, content="Be concise."),
            Message(role=Role.USER, content="What is 2+2?"),
        ],
        model="default",
        temperature=0.0,
        max_tokens=50,
    )


def test_translate_to_ai_separates_system(adapter, sample_request):
    payload = adapter.translate_to_ai(sample_request)

    # System message should become systemInstruction
    assert "systemInstruction" in payload
    assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise."

    # Only the user message should be in contents
    assert len(payload["contents"]) == 1
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "What is 2+2?"


def test_translate_to_ai_generation_config(adapter, sample_request):
    payload = adapter.translate_to_ai(sample_request)

    assert payload["generationConfig"]["temperature"] == 0.0
    assert payload["generationConfig"]["maxOutputTokens"] == 50


def test_translate_to_universal(adapter):
    raw = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "4"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6,
        },
    }
    resp = adapter.translate_to_universal(raw)

    assert resp.text == "4"
    assert resp.usage.total_tokens == 6
