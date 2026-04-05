"""Tests for the Ollama adapter — translation logic only."""

import pytest

from src.adapters.ollama_ext import OllamaAdapter
from src.core.models import Message, PipelineRequest, Role


@pytest.fixture
def adapter():
    return OllamaAdapter(base_url="http://fake:11434")


@pytest.fixture
def sample_request():
    return PipelineRequest(
        messages=[Message(role=Role.USER, content="Explain gravity.")],
        model="llama3",
        temperature=0.3,
        max_tokens=200,
    )


def test_translate_to_ai(adapter, sample_request):
    payload = adapter.translate_to_ai(sample_request)

    assert payload["model"] == "llama3"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["num_predict"] == 200
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == "Explain gravity."


def test_translate_to_universal(adapter):
    raw = {
        "model": "llama3",
        "message": {"role": "assistant", "content": "Gravity is..."},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 30,
    }
    resp = adapter.translate_to_universal(raw)

    assert resp.text == "Gravity is..."
    assert resp.model == "llama3"
    assert resp.usage.prompt_tokens == 12
    assert resp.usage.completion_tokens == 30
    assert resp.choices[0].finish_reason == "stop"
