# AI Router

A modular AI routing engine. Drop-in adapters and middleware, declarative config via `plugins.yaml`, OpenAI-compatible API surface.

Built as a foundation — use it standalone as a router, or extend it for larger projects (RAG brain, response evaluator, agent orchestration, etc.).

## Quick Start

```bash
uv sync
uv run python -m src.gateway.main
```

Server starts at `http://localhost:8080`. Point any OpenAI-compatible client to this URL.

LMStudio adapter is **on by default** (localhost:1234). Just have LMStudio running.

## Architecture

```
Client (LMStudio, Chatbox, any OpenAI client)
    |
    v
+-------------------------------+
|  Gateway (FastAPI)            |  localhost:8080/v1/chat/completions
|  OpenAI-compatible API        |
+-------------------------------+
    |
    v
+-------------------------------+
|  Plugin Registry              |  reads plugins.yaml
|  Auto-discovers adapters      |
|  and middleware at startup     |
+-------------------------------+
    |
    v
+-------------------------------+
|  Pre-Middleware Pipeline      |  (configurable, ordered)
|  e.g. early_exit, router      |
+-------------------------------+
    |
    v
+-------------------------------+
|  Adapter                      |  lmstudio | openai | ollama | gemini
|  Translates Universal Format  |
|  to/from AI-specific format   |
+-------------------------------+
    |
    v
+-------------------------------+
|  Post-Middleware Pipeline     |  (configurable, ordered)
|  e.g. logger                  |
+-------------------------------+
    |
    v
+-------------------------------+
|  Orchestrator                 |  tool-call loops, fan-out
+-------------------------------+
```

## plugins.yaml

All adapters and middleware are controlled declaratively:

```yaml
adapters:
  lmstudio:
    enabled: true
    module: src.adapters.lmstudio_ext
    settings:
      base_url: "http://localhost:1234/v1"

  openai:
    enabled: false
    module: src.adapters.openai_ext
    settings:
      api_key: "sk-..."

middleware:
  pre:
    - name: early_exit
      enabled: false
      module: src.middleware.early_exit
  post:
    - name: logger
      enabled: false
      module: src.middleware.logger
```

Enable a plugin by setting `enabled: true`. That's it.

## Adapter Routing

Use the model name to target specific adapters:

| Model string | Adapter | Example |
|---|---|---|
| `qwen3-0.6b` | lmstudio (default) | Any LMStudio model |
| `openai/gpt-4` | openai | Remote OpenAI API |
| `ollama/llama3` | ollama | Local Ollama |
| `gemini/gemini-2.0-flash` | gemini | Google Gemini |

## Adding Your Own Adapter

1. Copy `src/adapters/_template.py` to `src/adapters/myai_ext.py`
2. Implement `translate_to_ai()`, `translate_to_universal()`, `send()`
3. Add to `plugins.yaml`:
   ```yaml
   adapters:
     myai:
       enabled: true
       module: src.adapters.myai_ext
       settings:
         api_key: "..."
   ```

## Adding Your Own Middleware

1. Copy `src/middleware/_template.py` to `src/middleware/myfeature.py`
2. Subclass `PreMiddleware` or `PostMiddleware`
3. Implement `process()`
4. Add to `plugins.yaml` under `pre:` or `post:`

## Included Example Plugins

| Plugin | Type | Description | Default |
|---|---|---|---|
| **lmstudio** | Adapter | Local LMStudio (OpenAI-compatible) | ON |
| **openai** | Adapter | Remote OpenAI API | off |
| **ollama** | Adapter | Local Ollama | off |
| **gemini** | Adapter | Google Gemini | off |
| **early_exit** | Pre-middleware | Answer time/date/math locally | off |
| **router** | Pre-middleware | Fan-out to multiple adapters | off |
| **logger** | Post-middleware | Save conversations to SQLite | off |

## Production Configuration

All settings are configurable via environment variables (prefixed `ROUTER_`) or `.env` file. Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `ROUTER_DEV_MODE` | `false` | Enables `/docs`, verbose logging, error details in responses |
| `ROUTER_API_KEY` | _(empty)_ | If set, requires `Bearer <key>` on `/v1/*` endpoints |
| `ROUTER_RATE_LIMIT_RPM` | `0` | Requests per minute per IP (0 = disabled) |
| `ROUTER_MAX_BODY_SIZE` | `10485760` | Max request body in bytes (10 MB) |
| `ROUTER_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ROUTER_REQUEST_TIMEOUT` | `120.0` | Overall request timeout in seconds |

### Endpoints

| Path | Method | Description |
|---|---|---|
| `/` | GET | Basic status check |
| `/health` | GET | Detailed health check (lists adapters) |
| `/v1/models` | GET | List registered adapters as models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (streaming + non-streaming) |
| `/docs` | GET | Swagger UI (dev mode only) |

### Error Handling

All errors return structured JSON:

```json
{
  "error": {
    "type": "adapter_timeout",
    "message": "Upstream timed out",
    "adapter": "openai"
  }
}
```

Error types: `validation_error`, `authentication_error`, `rate_limit_exceeded`, `request_too_large`, `adapter_error`, `adapter_timeout`, `adapter_auth_error`, `adapter_rate_limited`, `adapter_not_found`, `internal_error`.

### Structured Logging

- **Production**: JSON logs with `request_id`, `elapsed_ms`, structured fields
- **Dev mode**: Human-readable `timestamp | LEVEL | logger | message` format
- Every response includes an `X-Request-ID` header for tracing

## Testing

```bash
uv run pytest tests/ -v
```

98 tests covering: pipeline execution, adapter translation, middleware logic, error handling, auth, rate limiting, streaming, security (safe math eval), connection pooling, and integration.

## Docker

```bash
docker build -t ai-router .
docker run -p 8080:8080 ai-router
```

Production Dockerfile: multi-stage build, non-root user, health check included.
