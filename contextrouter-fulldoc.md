# ContextRouter — Full Documentation

**The Reasoning Engine of ContextUnity**

ContextRouter is the AI Gateway and Agent Orchestration layer. It hosts LangGraph-based agents, manages LLM provider routing with automatic fallback, and coordinates multi-step reasoning workflows using the ContextUnit protocol.

---

## Overview

ContextRouter sits at the center of the ContextUnity ecosystem, receiving requests from various protocols (Telegram, Web, API) and orchestrating responses through AI agents. It delegates memory operations to ContextBrain and business logic to ContextCommerce.

### Key Responsibilities

1. **Agent Orchestration** — LangGraph state machines for complex workflows
2. **LLM Routing** — Intelligent provider selection with fallback (OpenAI, Anthropic, Vertex AI, local)
3. **RAG Pipeline** — Retrieval-Augmented Generation with Brain as backend
4. **Protocol Adapters** — Telegram, AG-UI, A2A event formats
5. **Tool Integration** — Exposes Brain (search) and Commerce (products) as LLM tools

### ContextUnit Protocol

All data flowing through ContextRouter uses the ContextUnit protocol from ContextCore:

```python
from contextcore import ContextUnit, ContextToken

unit = ContextUnit(
    payload={"query": "What is RAG?"},
    provenance=["connector:telegram", "graph:rag"],
    security=SecurityScopes(read=["knowledge:read"])
)

# Every stage adds to provenance
unit.provenance.append("step:generation")
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ContextRouter                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  modules/                         cortex/                                    │
│  ├── models/                      ├── graphs/                                │
│  │   ├── registry.py              │   ├── dispatcher.py    (graph router)   │
│  │   ├── types.py                 │   ├── rag_retrieval/   (RAG pipeline)   │
│  │   └── llm/                     │   ├── commerce/                         │
│  │       ├── openai.py            │   │   ├── gardener/    (taxonomy)       │
│  │       ├── anthropic.py         │   │   └── matcher/     (linking)        │
│  │       ├── vertex.py            │   └── news_engine/                      │
│  │       ├── perplexity.py        │       ├── harvest/     (modular)        │
│  │       └── rlm.py               │       │   ├── steps.py                  │
│  │                                │       │   ├── json_parser.py            │
│  ├── retrieval/rag/               │       │   └── prompts.py                │
│  │   ├── pipeline.py              │       ├── archivist/   (modular)        │
│  │   ├── pipeline_helpers.py      │       │   ├── steps.py                  │
│  │   └── pipeline_retrieval.py    │       │   ├── filters.py                │
│  │                                │       │   └── json_utils.py             │
│  ├── providers/storage/           │       ├── showrunner/  (modular)        │
│  │   └── brain.py                 │       │   ├── steps.py                  │
│  │                                │       │   ├── heuristics.py             │
│  └── tools/                       │       │   └── prompts.py                │
│                                   │       └── agents/      (modular)        │
│  cortex/services/                 │           ├── generation.py             │
│  └── graph/       (modular)       │           ├── personas.py               │
│      ├── __init__.py              │           └── language_tool.py          │
│      ├── local.py                 │                                          │
│      └── postgres.py              core/                                     │
│                                   ├── config/                                │
│                                   └── tokens.py                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Modular Design (400-Line Code Scale)

All modules follow the 400-Line Code Scale standard:
- **cortex/services/graph/**: Split from monolithic graph.py (~540 lines → 3 modules)
- **modules/retrieval/rag/**: Pipeline split into focused modules (~330 lines → 3 modules)
- **news_engine/*/**: Each stage (harvest, archivist, showrunner, agents) has extracted prompts, utils, heuristics

---

## Integration with ContextUnity

ContextRouter is the orchestration hub that connects all ContextUnity services:

| Service | Role | How Router Uses It |
|---------|------|-------------------|
| **ContextCore** | Shared types, ContextUnit, gRPC contracts | Types, tokens, protos |
| **ContextBrain** | Knowledge storage and RAG | Search, memory, taxonomy via gRPC |
| **ContextWorker** | Background task execution | Triggers workflows via Temporal |
| **ContextCommerce** | E-commerce platform | Products, enrichment, matching |

### BrainProvider

All memory operations delegate to ContextBrain:

```python
from contextrouter.modules.providers.storage import BrainProvider

brain = BrainProvider(config)
results = await brain.search("product taxonomy", limit=10)
related = await brain.graph_search(["entity:123"], depth=2)
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
Supplier-to-catalog product linking with RLM support for massive datasets.
- Input: Supplier products + catalog products
- Process: Multi-factor matching (name, brand, attributes)
- Output: Matched pairs with confidence scores

### 4. News Engine (`news_engine/`)

Multi-stage news curation pipeline:

#### Harvest
- Fetches news from multiple sources via Perplexity Sonar
- Extracts facts with structured JSON output
- Deduplicates via semantic similarity

#### Showrunner
- Creates editorial plan based on themes
- Assigns stories to AI personas
- Curates content flow

#### Agents (Generation)
- Parallel generation with semaphore-based concurrency
- Each persona has unique voice prompts
- Outputs formatted posts with source attribution

---

## Model Registry

All LLM usage MUST go through the model registry:

```python
from contextrouter.modules.models import model_registry

# Configuration-driven selection with fallback
model = model_registry.get_llm_with_fallback(
    key=config.models.default_llm,
    fallback_keys=config.models.fallback_llms,
    strategy="fallback",
    config=config,
)

# Specific provider (when needed)
model = model_registry.create_llm("perplexity/sonar", config=config)
```

### Supported Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **Vertex AI** | Gemini 2.0, 2.5 Pro | Primary (Google Cloud) |
| **OpenAI** | GPT-5-mini, o1, o3 | General purpose, reasoning |
| **Anthropic** | Claude Sonnet 4, Haiku | Reasoning, analysis |
| **Perplexity** | Sonar | Web-grounded search |
| **Groq** | Llama 3.3 70B | Ultra-fast inference |
| **RLM** | Recursive LM | Massive context (50k+ items) |
| **Local** | Ollama, vLLM | Development, privacy |

### Error Handling & Fallback

The `FallbackModel` handles provider failures automatically:

```python
# Quota exhaustion → immediate fallback (no retries)
except ModelQuotaExhaustedError:
    logger.warning(f"Model {key} quota exhausted, trying fallback")
    continue

# Rate limiting → fallback with delay
except ModelRateLimitError:
    logger.warning(f"Model {key} rate limited")
    continue
```

### Reasoning Models

OpenAI reasoning models (gpt-5, o1, o3) require special handling:

```python
# Use max_completion_tokens, not max_tokens
# Include extra budget for chain-of-thought reasoning
if is_reasoning_model:
    bind_kwargs["max_completion_tokens"] = 8000  # 4k reasoning + 4k response
```

---

## Configuration

### Environment Variables

```bash
# Core
CONTEXTROUTER_DEFAULT_LLM="openai/gpt-5-mini"
CONTEXTROUTER_FALLBACK_LLMS="anthropic/claude-sonnet-4,vertex/gemini-2.0-flash"

# Brain connection
BRAIN_MODE="grpc"
BRAIN_GRPC_HOST="localhost:50051"

# LLM providers
GOOGLE_CLOUD_PROJECT="my-project"
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="..."
PERPLEXITY_API_KEY="pplx-..."

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

## Adding New Functionality

See `CONTRIBUTING.md` for complete Golden Paths:

1. **Adding LLM Provider** — Create module + BUILTIN_LLMS + config + tests
2. **Adding Config Section** — Pydantic BaseSettings with env aliases
3. **Adding Cortex Graph** — StateGraph + state TypedDict + dispatcher registration
4. **Adding Tool** — `@tool_registry.register` decorator

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
| `cortex/graphs/news_engine/graph.py` | News engine orchestration |
| `cortex/graphs/news_engine/harvest/steps.py` | News discovery |
| `cortex/graphs/news_engine/harvest/json_parser.py` | Robust JSON extraction |
| `cortex/graphs/news_engine/archivist/steps.py` | Content validation |
| `cortex/graphs/news_engine/archivist/filters.py` | Banned keywords, thresholds |
| `cortex/graphs/news_engine/showrunner/steps.py` | Editorial planning |
| `cortex/graphs/news_engine/showrunner/heuristics.py` | Content scoring |
| `cortex/graphs/news_engine/agents/generation.py` | Post generation |
| `cortex/graphs/news_engine/agents/personas.py` | Agent persona definitions |
| `cortex/graphs/news_engine/agents/language_tool.py` | Ukrainian proofreading |
| `cortex/services/graph/` | Modular GraphService |
| `cortex/services/graph/local.py` | LocalGraphService |
| `cortex/services/graph/postgres.py` | PostgresGraphService |
| `modules/models/registry.py` | LLM provider registry, BUILTIN_LLMS |
| `modules/models/types.py` | ModelRequest, error types |
| `modules/retrieval/rag/pipeline.py` | RAG pipeline core |
| `modules/retrieval/rag/pipeline_helpers.py` | Pipeline helper functions |
| `modules/retrieval/rag/pipeline_retrieval.py` | Retrieval logic |
| `modules/providers/storage/brain.py` | Brain integration |
| `core/config/main.py` | Configuration management |

---

## Links

- **Documentation**: https://contextrouter.dev
- **Repository**: https://github.com/ContextUnity/contextrouter
- **API Reference**: https://contextrouter.dev/reference/

---

*Last updated: January 2026*
