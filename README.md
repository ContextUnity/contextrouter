# ContextRouter

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![GitHub](https://img.shields.io/badge/GitHub-ContextRouter-black.svg)](https://github.com/ContextRouter/contextrouter)
[![Docs](https://img.shields.io/badge/docs-contextrouter.org-green.svg)](https://contextrouter.org)

> ⚠️ **Early Version**: This is an early version of ContextRouter. Documentation is actively being developed, and the API may change.

## What is ContextRouter?

ContextRouter is a modular framework for building intelligent AI agents based on LangGraph. It acts as a "shared brain" that can handle complex tasks by combining information retrieval, text generation, and tool execution.

Unlike simple chatbots, ContextRouter can perform multi-step tasks: analyze queries, search for relevant information, apply logic, and provide structured responses.

## What is it for?

ContextRouter is designed for developers and companies who want to:

- **Build complex AI agents** — from simple Q&A systems to sophisticated workflows
- **Integrate RAG (Retrieval-Augmented Generation)** — search and generate responses based on your data
- **Create platform-independent solutions** — works with web, Telegram, API, or any other platform
- **Ensure security and traceability** — every piece of data has a provenance history

### Typical use cases:
- Corporate chatbots with knowledge bases
- AI assistants for document analysis
- Search-based recommendation systems
- Intelligent agents for business process automation

## Key Features

- **🧩 Fully Modular** — swap any component: LLM models, data stores, connectors, agents, and even entire processing graphs
- **🧠 Intelligent Orchestration** — sophisticated state management and conditional routing based on LangGraph
- **🛡️ Security and Tracing** — built-in Bisquit protocol for tracking data provenance
- **📡 Streaming-Oriented** — optimized for real-time and event-driven interfaces
- **🌍 Flexible Data Sources** — support for various storage solutions: Vertex AI Search, upcoming Postgres and local models support
- **🔧 Extensible by Design** — build custom agents, processing graphs, and integrations without touching core code

## Modules Overview

ContextRouter's architecture is built around specialized modules:

- **`modules/providers/`** — Data storage implementations (Vertex AI Search, Postgres, GCS)
- **`modules/connectors/`** — Raw data fetchers (Web search, RSS feeds, APIs, local files)
- **`modules/ingestion/`** — Data ingestion pipelines (ETL, indexing, RAG processing, deployment)
- **`modules/retrieval/`** — Search and RAG orchestration (pipelines, reranking, formatting)
- **`modules/models/`** — LLM and embedding model abstractions (Gemini, GPT, local models)
- **`modules/protocols/`** — Platform adapters (AG-UI events, A2A/A2UI protocols)

## RAG Capabilities

ContextRouter provides a complete RAG (Retrieval-Augmented Generation) pipeline powered by Vertex AI and Gemini:

### Ingestion Pipeline
- **Supported Content Types**: Books, articles, videos, Q&A pairs, web content, and custom structured data
- **Taxonomy & Ontology**: Automatic categorization and relationship mapping using AI-powered taxonomy builders
- **Knowledge Graph**: Semantic relationships and entity connections between ingested content
- **Citation System**: Precise source attribution with page numbers, timestamps, and context preservation

### Retrieval & Generation
- **Multi-stage Retrieval**: Initial search → reranking → context assembly
- **Citation Formatting**: Rich citations with source verification and confidence scores
- **Streaming Responses**: Real-time generation with source citations and reasoning traces

### Vertex AI + Gemini Integration
The RAG system runs on Google Cloud's Vertex AI Search for scalable vector storage and Gemini models for intelligent processing, ensuring enterprise-grade performance and security.

### Quick RAG Implementation
Build a production-ready RAG system in hours, not months. For custom integrations, enterprise deployments, or specialized RAG solutions, visit [contextrouter.dev](https://contextrouter.dev) to discuss your requirements.

## Roadmap

We're actively developing ContextRouter with focus on expanding data source support and improving developer experience:

### Near-term priorities:
- **PostgreSQL Integration** — native support for Postgres with pgvector for knowledge storage
- **Cognee Memory Integration** — advanced memory and knowledge graph capabilities
- **Local Model Support** — run AI models locally without cloud dependencies
- **Plugin System & Library** — comprehensive plugin architecture for extending functionality

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
