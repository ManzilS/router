"""Tests for the OpenAI adapter — translation logic only (no network)."""

import pytest

from src.adapters.openai_ext import OpenAIAdapter
from src.core.models import Message, PipelineRequest, Role


@pytest.fixture
def adapter():
    return OpenAIAdapter(base_url="http://fake", api_key="test-key")


@pytest.fixture
def sample_request():
    return PipelineRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello!"),
        ],
        model="gpt-4",
        temperature=0.5,
        max_tokens=100,
    )


def test_translate_to_ai(adapter, sample_request):
    payload = adapter.translate_to_ai(sample_request)

    assert payload["model"] == "gpt-4"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"] == "Hello!"


def test_translate_to_universal(adapter):
    raw = {
        "id": "chatcmpl-123",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    resp = adapter.translate_to_universal(raw)

    assert resp.id == "chatcmpl-123"
    assert resp.text == "Hi there!"
    assert resp.usage.total_tokens == 15
    assert resp.choices[0].finish_reason == "stop"


def test_translate_to_universal_with_tool_calls(adapter):
    raw = {
        "id": "chatcmpl-456",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    resp = adapter.translate_to_universal(raw)

    assert resp.has_tool_calls
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.function.name == "get_weather"
    assert tc.id == "call_abc"
