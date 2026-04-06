"""Tests for adapter connection pooling and lifecycle."""

import pytest

from src.adapters.openai_ext import OpenAIAdapter
from src.adapters.ollama_ext import OllamaAdapter
from src.adapters.gemini_ext import GeminiAdapter
from src.adapters.lmstudio_ext import LMStudioAdapter


def test_openai_adapter_creates_client():
    adapter = OpenAIAdapter(base_url="http://fake", api_key="test")
    client = adapter._get_client()
    assert client is not None
    assert not client.is_closed


def test_openai_adapter_reuses_client():
    adapter = OpenAIAdapter(base_url="http://fake", api_key="test")
    c1 = adapter._get_client()
    c2 = adapter._get_client()
    assert c1 is c2


@pytest.mark.asyncio
async def test_openai_adapter_close():
    adapter = OpenAIAdapter(base_url="http://fake", api_key="test")
    adapter._get_client()
    await adapter.close()
    assert adapter._client is None


def test_ollama_adapter_creates_client():
    adapter = OllamaAdapter(base_url="http://fake:11434")
    client = adapter._get_client()
    assert client is not None


@pytest.mark.asyncio
async def test_ollama_adapter_close():
    adapter = OllamaAdapter()
    adapter._get_client()
    await adapter.close()
    assert adapter._client is None


def test_gemini_adapter_creates_client():
    adapter = GeminiAdapter(api_key="test")
    client = adapter._get_client()
    assert client is not None


@pytest.mark.asyncio
async def test_gemini_adapter_close():
    adapter = GeminiAdapter(api_key="test")
    adapter._get_client()
    await adapter.close()
    assert adapter._client is None


def test_lmstudio_inherits_pooling():
    adapter = LMStudioAdapter()
    client = adapter._get_client()
    assert client is not None
    assert adapter.base_url == "http://localhost:1234/v1"


def test_lmstudio_defaults():
    adapter = LMStudioAdapter()
    assert adapter.name == "lmstudio"
    assert adapter.api_key == "lm-studio"
