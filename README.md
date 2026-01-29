# ContextRouter

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![GitHub](https://img.shields.io/badge/GitHub-ContextUnity-black.svg)](https://github.com/ContextUnity/contextrouter)
[![Docs](https://img.shields.io/badge/docs-contextrouter.dev-green.svg)](https://contextrouter.dev)

> ⚠️ **Early Version**: This is an early version of ContextRouter. Documentation is actively being developed, and the API may change.

## What is ContextRouter?

ContextRouter is the **AI Gateway and Agent Orchestration** layer of the [ContextUnity](https://github.com/ContextUnity) ecosystem. It's built on LangGraph and provides:

- **LLM Provider Routing** — OpenAI, Anthropic, Vertex AI, Groq, Perplexity, local models
- **Agent Orchestration** — LangGraph state machines for complex workflows
- **Fallback & Reliability** — automatic provider fallback with quota/rate limit handling
- **ContextUnit Protocol** — all data flows through the provenance-tracking ContextUnit format
- **Tool Integration** — exposes Brain (search) and Commerce (products) as LLM tools

Think of it as the **"Mind"** that processes requests, delegates memory to Brain, and orchestrates multi-step reasoning.

## Core Concepts

### ContextUnit — The Atomic Unit

All data flowing through ContextRouter uses the **ContextUnit** protocol from [ContextCore](https://github.com/ContextUnity/contextcore):

```python
from contextcore import ContextUnit, ContextToken

unit = ContextUnit(
    payload={"query": "What is RAG?"},
    provenance=["connector:telegram", "graph:rag"],
    security=SecurityScopes(read=["knowledge:read"])
)

# Authorization via capability-based tokens
token = ContextToken(permissions=("knowledge:read",))
if token.can_read(unit.security):
    # Process request
```

Every transformation adds to the provenance chain, enabling full traceability.

### Model Registry

All LLM usage goes through the central registry with automatic fallback:

```python
from contextrouter.modules.models import model_registry

model = model_registry.get_llm_with_fallback(
    key="openai/gpt-5-mini",
    fallback_keys=["anthropic/claude-sonnet-4", "vertex/gemini-2.5-flash"],
    strategy="fallback",
    config=config,
)

response = await model.generate(request)
```

## Integration with ContextUnity

ContextRouter is the orchestration layer that connects all ContextUnity services:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ContextRouter                               │
│                     (The "Mind" — Orchestration)                    │
├─────────────────────────────────────────────────────────────────────┤
│  • Receives requests from protocols (Telegram, Web, API)            │
│  • Routes to appropriate LLM providers                              │
│  • Orchestrates multi-step agent workflows                          │
│  • Delegates memory operations to Brain                             │
│  • Exposes tools for LLM function calling                           │
└───────────────┬─────────────────────────────────────┬───────────────┘
                │                                     │
                ▼                                     ▼
┌───────────────────────────┐           ┌───────────────────────────┐
│      ContextBrain         │           │     ContextCommerce       │
│  (The "Memory" — RAG)     │           │   (The "Store" — PIM)     │
├───────────────────────────┤           ├───────────────────────────┤
│ • Vector storage          │           │ • Product catalog         │
│ • Semantic search         │           │ • Taxonomy management     │
│ • Knowledge graph         │           │ • Supplier integration    │
│ • Episodic memory         │           │ • E-commerce backend      │
└───────────────────────────┘           └───────────────────────────┘
                ▲                                     ▲
                │                                     │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      ContextWorker        │
                    │  (The "Hands" — Tasks)    │
                    ├───────────────────────────┤
                    │ • Temporal workflows      │
                    │ • Background processing   │
                    │ • Scheduled jobs          │
                    └───────────────────────────┘
```

| Service | Role | How Router Uses It |
|---------|------|-------------------|
| **ContextCore** | Shared types, ContextUnit, gRPC contracts | Types, tokens, protos |
| **ContextBrain** | Knowledge storage and RAG | Search, memory, taxonomy via gRPC |
| **ContextWorker** | Background task execution | Triggers workflows via Temporal |
| **ContextCommerce** | E-commerce platform | Products, enrichment, matching |

> **What is gRPC?** [gRPC](https://grpc.io/) is a high-performance RPC framework that uses Protocol Buffers for serialization. It enables type-safe, efficient communication between services — faster than REST, with built-in streaming support.

### Memory & Retrieval (The Brain)

ContextRouter delegates all memory operations to **ContextBrain** via the `BrainProvider`:

```python
from contextrouter.modules.providers.storage import BrainProvider

brain = BrainProvider(config)
results = await brain.search("product taxonomy", limit=10)
```

Set your mode via `BRAIN_MODE=local` or `BRAIN_MODE=grpc`.

## Key Features

- **🧩 Modular Architecture** — swap components without changing agent logic
- **🎯 Agent Orchestration** — LangGraph state machines for complex workflows
- **🛡️ Production Ready** — ContextUnit protocol for data provenance and audit trails
- **🌐 Universal Model Support** — 15+ LLM providers with automatic fallback
- **⚡ Reliability** — quota exhaustion, rate limit, and timeout handling
- **🔧 Extensible** — add providers, graphs, tools via registry pattern

## Supported LLM Providers

| Provider | Key | Use Case |
|----------|-----|----------|
| **Vertex AI** | `vertex/gemini-2.0-flash` | Production, multimodal |
| **OpenAI** | `openai/gpt-5-mini` | General purpose |
| **Anthropic** | `anthropic/claude-sonnet-4` | Reasoning, analysis |
| **Perplexity** | `perplexity/sonar` | Web-grounded search |
| **Groq** | `groq/llama-3.3-70b-versatile` | Ultra-fast inference |
| **OpenRouter** | `openrouter/deepseek/deepseek-r1` | Access to 100+ models |
| **Local** | `local/llama3.2` | Privacy, development |
| **RLM** | `rlm/gpt-5-mini` | Massive context (50k+ items) |

## Quick Start

```python
from contextrouter.cortex import stream_agent

async for event in stream_agent(
    messages=[{"role": "user", "content": "How does RAG work?"}],
    session_id="session_123",
    platform="web",
):
    print(event)
```

## Installation

```bash
pip install contextrouter

# With all providers (recommended):
pip install contextrouter[vertex,storage,ingestion]

# Observability (optional):
pip install contextrouter[observability]
```

## Configuration

```bash
# LLM routing
export CONTEXTROUTER_DEFAULT_LLM="openai/gpt-5-mini"
export CONTEXTROUTER_FALLBACK_LLMS="anthropic/claude-sonnet-4,vertex/gemini-2.0-flash"

# Brain connection
export BRAIN_MODE="grpc"
export BRAIN_GRPC_HOST="localhost:50051"

# LLM providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_CLOUD_PROJECT="my-project"
export PERPLEXITY_API_KEY="pplx-..."
```

## Documentation

- [Full Documentation](https://contextrouter.dev) — complete guides and API reference
- [Technical Reference](./contextrouter-fulldoc.md) — architecture deep-dive
- [Contributing Guide](./CONTRIBUTING.md) — Golden Paths for adding functionality

## Contributing

We welcome contributions! See our [Contributing Guide](./CONTRIBUTING.md) for:

- **Golden Path: Adding LLM Providers** — full template with error handling
- **Golden Path: Adding Config Sections** — Pydantic settings pattern
- **Golden Path: Adding Cortex Graphs** — LangGraph agent workflows
- **Golden Path: Adding Tools** — LLM function calling

## License

This project is licensed under the terms specified in [LICENSE.md](LICENSE.md).
