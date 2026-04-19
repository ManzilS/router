# router

**A modular AI routing engine. One OpenAI-compatible API in front of multiple LLM providers.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

---

## What it does

`router` sits between your application and any number of LLM providers — OpenAI, Gemini, Ollama, LMStudio — and exposes a **single OpenAI-compatible API surface**. Your app talks to `router` the same way it talks to OpenAI. `router` handles the rest: routing by model, auth, retries, request/response transforms, streaming, and fallback.

It's drop-in (swap your `base_url` and go), extensible (adapters and middleware are Python classes with a simple contract), and config-driven (`plugins.yaml`).

## Why

Modern AI applications don't want to be married to one provider. You want to:
- Route some requests to a local Ollama instance and others to GPT-4.
- Try Gemini for vision without rewriting your pipeline.
- Fall back to a secondary provider when a primary is down.
- A/B test two models without touching application code.

`router` exists so you can do all of that without forking your app for every provider.

## Architecture

```
┌────────────────┐
│ your app /     │ ── OpenAI-compatible requests ──┐
│ RAG pipeline   │                                  │
└────────────────┘                                  ▼
                                      ┌─────────────────────────┐
                                      │         router          │
                                      │                         │
                                      │  [middleware chain]     │
                                      │      ↓                  │
                                      │  [model → adapter map]  │
                                      │      ↓                  │
                                      └───────┬──────────┬──────┘
                                              │          │
                                         ┌────▼───┐  ┌───▼────┐
                                         │ OpenAI │  │ Ollama │  ...
                                         └────────┘  └────────┘
```

- **Adapters** normalize each provider's request/response shape (including streaming) to OpenAI's SSE format.
- **Middleware** runs in a declared order — auth, logging, retry, model mapping, rate limiting, etc.
- **plugins.yaml** wires everything together declaratively so adding a new provider or middleware is config, not code changes.

## Quickstart

```bash
git clone https://github.com/ManzilS/router
cd router
pip install -r requirements.txt
cp plugins.example.yaml plugins.yaml   # edit to add your providers
python -m router
```

Point your app at `http://localhost:8080/v1` instead of `https://api.openai.com/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")

r = client.chat.completions.create(
    model="gpt-4o-mini",        # or "llama3" to route to local Ollama
    messages=[{"role":"user","content":"hello"}]
)
```

## Example `plugins.yaml`

```yaml
providers:
  - name: openai
    adapter: openai
    api_key_env: OPENAI_API_KEY

  - name: ollama-local
    adapter: ollama
    base_url: http://localhost:11434

middleware:
  - name: log_requests
  - name: retry
    config: { max_attempts: 3 }
  - name: model_router
    routes:
      "gpt-*":  openai
      "llama*": ollama-local
```

## What's implemented today

- [x] OpenAI adapter (chat, streaming)
- [x] Ollama adapter
- [x] Gemini adapter
- [x] LMStudio adapter
- [x] Middleware chain (request and response phases)
- [x] Model → provider routing via plugins.yaml
- [ ] Per-provider rate limiting
- [ ] Metrics endpoint (Prometheus)
- [ ] Cost tracking per model

## Roadmap

- Tool-call normalization across providers
- Embedding endpoint parity (`/v1/embeddings`)
- Built-in cache middleware (semantic + exact)
- First-class Anthropic adapter

## License

MIT

## About

Built by [Manzil "Nick" Sapkota](https://github.com/ManzilS) — open to AI/ML Engineer roles. [Resume](mailto:manzilsapkota@gmail.com) · [LinkedIn](https://www.linkedin.com/in/manzilsapkota/).
