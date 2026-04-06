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

## Testing

```bash
uv run pytest tests/ -v
```

## Docker

```bash
docker build -t ai-router .
docker run -p 8080:8080 ai-router
```
