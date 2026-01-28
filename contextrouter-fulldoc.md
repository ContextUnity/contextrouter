# ContextRouter — Full Documentation

**The Reasoning Engine of ContextUnity**

ContextRouter is the AI Gateway and Agent Orchestration layer. It hosts LangGraph-based agents, manages LLM provider routing, and coordinates multi-step reasoning workflows.

---

## Overview

ContextRouter sits at the center of the ContextUnity ecosystem, receiving requests from various protocols (Telegram, Web, API) and orchestrating responses through AI agents. It delegates memory operations to ContextBrain and business logic to ContextCommerce.

### Key Responsibilities

1. **Agent Orchestration** — LangGraph state machines for complex workflows
2. **LLM Routing** — Intelligent provider selection (OpenAI, Anthropic, Vertex AI, local)
3. **RAG Pipeline** — Retrieval-Augmented Generation with Brain as backend
4. **Protocol Adapters** — Telegram, AG-UI, A2A event formats
5. **Tool Integration** — Exposes Brain (search) and Commerce (products) as LLM tools

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              ContextRouter                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  modules/                         cortex/                                  │
│  ├── models/                      ├── graphs/                              │
│  │   ├── registry.py         ────▶│   ├── dispatcher.py    (graph router) │
│  │   ├── llm/                     │   ├── rag_retrieval/   (RAG pipeline) │
│  │   │   ├── openai.py            │   ├── commerce/                       │
│  │   │   ├── anthropic.py         │   │   ├── gardener/    (taxonomy)     │
│  │   │   ├── vertex.py            │   │   ├── matcher/     (linking)      │
│  │   │   ├── perplexity.py        │   │   └── chat/        (product Q&A)  │
│  │   │   └── rlm.py (RLM)         │   └── news_engine/                    │
│  │   └── embeddings/              │       └── showrunner/  (curation)     │
│  │                                │                                        │
│  ├── protocols/                   └── steps/                              │
│  │   ├── agui.py                      ├── retrieval.py                    │
│  │   └── telegram.py                  ├── generation.py                   │
│  │                                    └── tool_use.py                     │
│  ├── providers/                                                            │
│  │   └── storage/                 core/                                   │
│  │       └── brain_provider.py    ├── config/                             │
│  │                                ├── registry.py                          │
│  └── tools/                       └── context_unit.py                      │
│      └── registry.py                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Agents (cortex/graphs/)

### 1. Dispatcher (`dispatcher.py`)

Central graph router. Selects which agent graph to execute based on:
- Request type (chat, enrichment, matching)
- Configuration flags
- User context

### 2. RAG Retrieval (`rag_retrieval/`)

Standard Retrieval-Augmented Generation pipeline:
1. **Retrieve** — Query Brain for relevant context
2. **Rerank** — Score and filter results  
3. **Generate** — LLM response with citations

### 3. Commerce Agents (`commerce/`)

#### Gardener Agent
Taxonomy classification and product enrichment.
- Input: Unclassified products from Commerce
- Process: Semantic matching against Brain taxonomy
- Output: Category assignments, attributes extraction

#### Matcher Agent  
Supplier-to-catalog product linking.
- Input: Supplier products + catalog products
- Process: Multi-factor matching (name, brand, attributes)
- Output: Matched pairs with confidence scores

#### Chat Agent
Product-aware conversational interface.
- Input: User questions about products
- Process: RAG with product context
- Output: Grounded responses with product references

### 4. News Engine (`news_engine/`)

#### Showrunner Agent
Content curation and editorial planning.
- Input: Raw news feeds, themes
- Process: Semantic filtering, deduplication, planning
- Output: Curated content with AI-generated summaries

---

## Model Registry

All LLM usage MUST go through the model registry:

```python
from contextrouter.modules.models import model_registry

# Configuration-driven selection
model = model_registry.get_llm_with_fallback(
    key=config.models.default_llm,  # From CONTEXTROUTER_DEFAULT_LLM
    fallback_keys=["gpt-4o", "claude-sonnet"],
    strategy="fallback",
    config=config,
)

# Specific provider (when needed)
model = model_registry.create_llm("perplexity/sonar", config=config)
```

### Supported Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **Vertex AI** | Gemini 2.0, 1.5 Pro | Primary (Google Cloud) |
| **OpenAI** | GPT-4o, o1, o3 | General purpose |
| **Anthropic** | Claude Sonnet, Haiku | Reasoning, analysis |
| **Perplexity** | Sonar | Web-grounded search |
| **RLM** | Recursive LM | Massive context (50k+ items) |
| **Local** | Ollama, vLLM | Development, privacy |

---

## Brain Integration

ContextRouter delegates all memory operations to ContextBrain:

### Mode Configuration

```bash
# Local mode (library import)
export BRAIN_MODE=local

# Remote mode (gRPC call)
export BRAIN_MODE=grpc
export BRAIN_GRPC_HOST=localhost:50051
```

### BrainProvider Interface

```python
from contextrouter.modules.providers.storage import BrainProvider

brain = BrainProvider(config)

# Semantic search
results = await brain.search("product taxonomy", limit=10)

# Graph traversal
related = await brain.graph_search(["entity:123"], depth=2)
```

---

## Protocols

### Telegram Protocol

```python
from contextrouter.modules.protocols.telegram import TelegramAdapter

adapter = TelegramAdapter(bot_token, router_config)
# Handles: messages → agent → response formatting
```

### AG-UI Protocol

Browser-based chat interface with streaming support:
- WebSocket events for real-time updates
- Structured message formats
- Citation rendering

---

## Configuration

### Environment Variables

```bash
# Core
CONTEXTROUTER_DEFAULT_LLM="gemini-2.0-flash"
CONTEXTROUTER_FALLBACK_LLMS="gpt-4o,claude-sonnet"

# Brain connection
BRAIN_MODE="grpc"
BRAIN_GRPC_HOST="localhost:50051"

# LLM providers
GOOGLE_CLOUD_PROJECT="my-project"
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="..."
PERPLEXITY_API_KEY="..."

# Observability
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
```

### Config Classes

```python
from contextrouter.core.config import RouterConfig

config = RouterConfig.from_env()
# Access: config.models, config.brain, config.providers
```

---

## CLI

```bash
# Run as API server
contextrouter serve --host 0.0.0.0 --port 8000

# Run specific agent
contextrouter run gardener --input products.json

# Check configuration
contextrouter config show
```

---

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest -m unit
uv run pytest -m integration

# With coverage
uv run pytest --cov=contextrouter
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `cortex/graphs/dispatcher.py` | Agent graph router |
| `cortex/graphs/rag_retrieval/graph.py` | RAG pipeline |
| `cortex/graphs/commerce/gardener/graph.py` | Taxonomy classifier |
| `modules/models/registry.py` | LLM provider registry |
| `modules/providers/storage/brain_provider.py` | Brain integration |
| `core/config/main.py` | Configuration management |

---

## Links

- **Documentation**: https://contextrouter.dev
- **Repository**: https://github.com/ContextUnity/contextrouter
- **API Reference**: https://contextrouter.dev/reference/

---

*Last updated: January 2026*
