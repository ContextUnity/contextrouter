# ContextRouter

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![GitHub](https://img.shields.io/badge/GitHub-ContextRouter-black.svg)](https://github.com/ContextRouter/contextrouter)
[![Docs](https://img.shields.io/badge/docs-contextrouter.org-green.svg)](https://contextrouter.org)

> ⚠️ **Early Version**: This is an early version of ContextRouter. Documentation is actively being developed, and the API may change.

## What is ContextRouter?

ContextRouter is a modular AI agent framework designed for building production-ready agent orchestration systems. It's built on top of LangGraph and provides a clean separation between your agent's decision logic and the technical implementation details.

Think of it as an **AI Gateway** that can:
- **Orchestrate multiple LLM providers** (OpenAI, Anthropic, Vertex AI, Groq, local models)
- **Route requests intelligently** based on latency, cost, and user tier
- **Manage agent workflows** using LangGraph state machines
- **Handle voice I/O** for speech-to-text and text-to-speech
- **Scale across instances** with shared state management

## What is it for?

ContextRouter is designed for developers and companies who want to:

- **Build complex AI agents** — from simple Q&A systems to sophisticated workflows
- **Orchestrate agent workflows** — multi-step tasks with state management and conditional routing
- **Create platform-independent solutions** — works with web, Telegram, API, or any other platform
- **Ensure security and traceability** — every piece of data uses ContextUnit protocol for full provenance tracking

### Typical use cases:
- AI Gateway and load balancing for LLM providers
- Agent orchestration for complex business workflows
- Voice-enabled personal assistants
- Multi-instance production deployments

## Key Features

- **🧩 Truly Modular** — every component can be swapped without changing your agent logic
- **🎯 Agent Orchestration** — build sophisticated agent workflows with LangGraph state machines
- **🛡️ Production Ready** — ContextUnit protocol for data provenance and audit trails, multi-instance safe state
- **🌐 Universal Model Support** — use any LLM provider: commercial (OpenAI, Anthropic, Vertex AI, Groq), aggregators (OpenRouter), or local (Ollama, vLLM)
- **🔧 Extensible by Design** — build custom agents, processing graphs, and integrations without touching core code

## Modules Overview

ContextRouter's architecture is built around specialized modules:

- **`modules/models/`** — LLM and embedding model abstractions (OpenAI, Anthropic, Vertex AI, Groq, local models)
- **`modules/protocols/`** — Platform adapters (AG-UI events, A2A/A2UI protocols)
- **`cortex/graphs/`** — LangGraph-based agent workflows:
  - `dispatcher.py` — Central graph selection (by config/registry)
  - `rag_retrieval.py` — RAG pipeline (retrieve → generate)
  - `commerce/` — Commerce domain (gardener, lexicon, matcher, chat)
- **`core/`** — ContextUnit protocol, token management, and core interfaces

## Integration with ContextUnity

ContextRouter is part of the ContextUnity ecosystem:

- **ContextCore** — Shared types and ContextUnit protocol
- **ContextCore** — Shared types and ContextUnit protocol
- **ContextBrain** — RAG retrieval and knowledge storage (**Centralized Brain**)
- **ContextWorker** — Background task execution
- **ContextCommerce** — E-commerce platform with agent integration

### Memory & Retrieval (The Brain)

ContextRouter no longer manages vector databases directly. It delegates all memory operations to **ContextBrain** via the `BrainProvider`.

| Mode | Description | Requirements |
|------|-------------|--------------|
| **Local** | Direct library import | `pip install contextbrain` |
| **gRPC** | Network call to remote service | `contextbrain` service running |

Set your mode via `BRAIN_MODE=local` or `BRAIN_MODE=grpc`. See [Storage Provider Docs](./src/contextrouter/modules/providers/storage/README.md) for details.

For RAG capabilities, knowledge storage, and ingestion pipelines, see [ContextBrain](https://contextbrain.dev).

## Roadmap

We're actively developing ContextRouter with focus on improving agent orchestration and developer experience:

### Near-term priorities:
- **Enhanced Voice I/O** — improved speech-to-text and text-to-speech capabilities
- **Advanced Routing** — smarter provider selection based on cost, latency, and quality
- **Plugin System** — comprehensive plugin architecture for extending functionality
- **Multi-instance Improvements** — better state synchronization and leader election

## Quick Start

```python
from contextrouter.cortex import stream_agent

# Initialize the shared brain
async for event in stream_agent(
    messages=[{"role": "user", "content": "How does RAG work?"}],
    session_id="session_123",
    platform="web",
    style_prompt="Be concise and technical."
):
    print(event)
```

For more examples, see the [`examples/`](./examples/) directory.

## Getting Started

1. **Install ContextRouter**:
   ```bash
   pip install contextrouter
   # For full functionality (recommended):
   pip install contextrouter[vertex,storage,ingestion]
   # Observability (optional):
   pip install contextrouter[observability]
   ```

2. **Configure your data sources** and LLM models
3. **Build your first agent** using the examples above
4. **Deploy** to your preferred platform (web, API, Telegram, etc.)

### Notes (Vertex / Gemini)

- **Vertex AI mode**: ContextRouter sets `GOOGLE_GENAI_USE_VERTEXAI=true` by default to avoid the
  Google GenAI SDK accidentally trying API-key auth. You can override it by exporting
  `GOOGLE_GENAI_USE_VERTEXAI=false` before importing/starting ContextRouter.

## Documentation

- [Full Documentation](https://contextrouter.org) — complete guides and API reference
- [Examples Directory](./examples/) — working code samples
- [Contributing Guide](./CONTRIBUTING.md) — how to contribute to the project

## Contributing

We welcome contributions! ContextRouter maintains strict coding standards with emphasis on:

- **Security First** — All contributions undergo security review and automated scanning
- **Code Quality** — Comprehensive linting, type checking, and automated testing
- **Clean Architecture** — Clear separation between business logic, infrastructure, and data layers
- **Type Safety** — Strict typing throughout the codebase with mypy validation

See our [Contributing Guide](./CONTRIBUTING.md) for detailed guidelines and current development priorities.

## License

This project is licensed under the terms specified in [LICENSE.md](LICENSE.md).
