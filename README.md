# ContextRouter

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

ContextRouter is the **AI Gateway and Agent Orchestration** layer of the [ContextUnity](https://github.com/ContextUnity) ecosystem. Built on LangGraph, it routes LLM requests across 13 providers with automatic fallback, orchestrates multi-step agent workflows, and delegates memory to Brain.

---

## What is it for?

- **Agent Orchestration** — LangGraph state machines for complex reasoning workflows
- **LLM Provider Routing** — OpenAI, Anthropic, Vertex AI, Groq, Perplexity, local models with automatic fallback
- **RAG Pipeline** — Retrieval-Augmented Generation with Brain as backend
- **Tool Integration** — Federated tool execution via bidirectional gRPC streams
- **Protocol Adapters** — Telegram, AG-UI, A2A event formats

---

## Quick Start

```bash
# Start gRPC server
export REDIS_URL="redis://localhost:6379/0"
export CU_ROUTER_DEFAULT_LLM="openai/gpt-5-mini"
export OPENAI_API_KEY="sk-..."
uv run python -m contextunity.router

# Run tests
uv run --package contextunity-router pytest
```

---

## Architecture

```
src/contextunity/router/
├── modules/                      # Capability modules (models, tools, RAG)
├── cortex/                       # AI orchestration (LangGraph agents)
│   └── graphs/                   # dispatcher, rag_retrieval, commerce, news_engine, analytics
├── service/                      # gRPC service (mixin-based dispatcher)
├── core/                         # Config, plugins, registry
├── api/                          # REST/HTTP (FastAPI)
└── cli/                          # CLI commands
```

---

## Supported LLM Providers

| Provider | Key | Use Case |
|----------|-----|----------|
| **Vertex AI** | `vertex/gemini-2.5-flash` | Production, multimodal |
| **OpenAI** | `openai/gpt-5-mini` | General purpose |
| **Anthropic** | `anthropic/claude-sonnet-4` | Reasoning, analysis |
| **Perplexity** | `perplexity/sonar` | Web-grounded search |
| **Groq** | `groq/llama-3.3-70b-versatile` | Ultra-fast inference |
| **OpenRouter** | `openrouter/deepseek/deepseek-r1` | 100+ models |
| **Local** | `local/llama3.2` | Privacy, development |
| **RLM** | `rlm/gpt-5-mini` | Massive context (50k+ items) |

---

## Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_PORT` | `50052` | gRPC server port |
| `REDIS_URL` | — | Service discovery and persistence |
| `CU_ROUTER_DEFAULT_LLM` | `openai/gpt-5-mini` | Baseline LLM |
| `CU_ROUTER_FALLBACK_LLMS` | — | Comma-separated global fallback chain |
| `CU_BRAIN_GRPC_URL` | — | Override for Brain host |
| `CU_SHIELD_GRPC_URL` | — | Override for Shield host |

---

## Security

ContextRouter enforces multi-layer security:
- **SPOT Pattern** — `ContextToken` is the sole source of identity (no user_id/tenant_id in payloads)
- **Three-Layer Registration** — tenant binding → permission check → Redis ownership verification
- **SecureTool Wrapping** — every tool enforces `bound_tenant` and permission checks
- **Shield Pre-LLM Guard** — input scanning for prompt injection (Enterprise)

---

## Further Reading

- **Full Documentation**: [ContextRouter on Astro Site](../../docs/website/src/content/docs/router/)
- **Agent Boundaries & Golden Paths**: [AGENTS.md](AGENTS.md)

## License

This project is licensed under the terms specified in [LICENSE.md](LICENSE.md).
