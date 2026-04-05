# AI Gateway Brain

A central "nervous system" for all your AI interactions. Acts as a local proxy that any OpenAI-compatible client can connect to, routing requests through a configurable pipeline of middleware (RAG injection, logging, early exits, fan-out routing) before dispatching to any backend AI (OpenAI, Ollama, Gemini).

## Architecture

```
Client (LMStudio, ChatGPT, etc.)
    |
    v
+-----------------------------+
|  Gateway (FastAPI)          |  localhost:8080/v1/chat/completions
|  OpenAI-compatible API      |
+----------+------------------+
           |
           v
+-----------------------------+
|  Pre-Middleware Pipeline    |
|  +- Early Exit (time/math)  |
|  +- Router (fan-out)        |
|  +- RAG Injector (context)  |
+----------+------------------+
           |
           v
+-----------------------------+
|  Adapter (format translator)|
|  openai | ollama | gemini   |
+----------+------------------+
           |
           v
+-----------------------------+
|  Post-Middleware Pipeline   |
|  +- RAG Injector (ingest)   |
|  +- Logger (SQLite)         |
+-----------------------------+
           |
           v
+-----------------------------+
|  Orchestrator (loop ctrl)   |
|  Handles tool-call loops    |
|  up to max_loops iterations |
+-----------------------------+
```

## Quick Start

```bash
# Clone and setup
uv sync

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
uv run python -m src.gateway.main
```

The server starts at `http://localhost:8080`. Point any OpenAI-compatible client to this URL as the base URL.

## Adapter Routing

Specify which backend to use via the model name:

| Model string | Adapter | Example |
|---|---|---|
| `gpt-4` | openai (default) | Standard OpenAI |
| `ollama/llama3` | ollama | Local Ollama model |
| `gemini/gemini-2.0-flash` | gemini | Google Gemini |

## Middleware

| Middleware | Phase | Description |
|---|---|---|
| **Early Exit** | Pre | Handles trivial queries (time, date, math, ping) locally |
| **Router** | Pre | Fan-out to multiple adapters simultaneously |
| **RAG Injector** | Pre + Post | Injects relevant past context; ingests new conversations |
| **Logger** | Post | Writes all traffic to SQLite for history |

## Configuration

All settings via environment variables (prefix `AIGW_`) or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `AIGW_PORT` | 8080 | Server port |
| `AIGW_DEFAULT_ADAPTER` | openai | Default backend |
| `AIGW_MAX_LOOPS` | 3 | Max orchestrator loops |
| `AIGW_OPENAI_API_KEY` | | Your OpenAI key |
| `AIGW_OLLAMA_BASE_URL` | http://localhost:11434 | Ollama endpoint |
| `AIGW_GEMINI_API_KEY` | | Your Gemini key |
| `AIGW_ENABLE_RAG` | true | Toggle RAG injection |
| `AIGW_ENABLE_EARLY_EXIT` | true | Toggle early exits |
| `AIGW_ENABLE_LOGGER` | true | Toggle conversation logging |

## Testing

```bash
uv run pytest tests/ -v
```

## Docker

```bash
docker build -t ai-gateway-brain .
docker run -p 8080:8080 --env-file .env ai-gateway-brain
```

## Adding a New Adapter

1. Create `src/adapters/myai_ext.py`
2. Subclass `AdapterBase` and implement `translate_to_ai()`, `translate_to_universal()`, and `send()`
3. Register it in `src/gateway/main.py` in the `adapters` dict

## Adding New Middleware

1. Create `src/middleware/myfeature.py`
2. Subclass `MiddlewareBase` and implement `process()`
3. Add it to the `pre_middleware` or `post_middleware` list in `src/gateway/main.py`
