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
src/contextrouter/
├── modules/                             ← Capability modules
│   ├── models/
│   │   ├── registry.py                  ← LLM provider registry, BUILTIN_LLMS
│   │   ├── base.py                      ← BaseLLMProvider
│   │   ├── types.py                     ← ModelRequest, error types
│   │   ├── llm/                         ← 13 LLM providers
│   │   │   ├── openai.py                ← OpenAI (GPT-5, o1, o3)
│   │   │   ├── anthropic.py             ← Anthropic (Claude Sonnet/Haiku)
│   │   │   ├── vertex.py                ← Google Vertex AI (Gemini)
│   │   │   ├── perplexity.py            ← Perplexity (Sonar)
│   │   │   ├── groq.py                  ← Groq (Llama ultra-fast)
│   │   │   ├── rlm.py                   ← RLM (Recursive Language Model)
│   │   │   ├── huggingface.py           ← HuggingFace Inference
│   │   │   ├── hf_hub.py                ← HuggingFace Hub
│   │   │   ├── openrouter.py            ← OpenRouter (multi-provider)
│   │   │   ├── litellm.py               ← LiteLLM (universal adapter)
│   │   │   ├── local_openai.py          ← Local OpenAI-compat (Ollama, vLLM)
│   │   │   ├── openai_batch.py          ← OpenAI Batch API
│   │   │   ├── runpod.py                ← RunPod inference
│   │   │   └── _openai_compat.py        ← Shared OpenAI-compat base
│   │   └── embeddings/                  ← Embedding providers
│   │       ├── huggingface.py
│   │       └── vertex.py
│   │
│   ├── retrieval/rag/                   ← RAG pipeline (modular)
│   │   ├── pipeline.py
│   │   ├── pipeline_helpers.py
│   │   └── pipeline_retrieval.py
│   │
│   ├── providers/storage/brain.py       ← BrainProvider integration
│   ├── connectors/                      ← Protocol connectors (Telegram, etc.)
│   ├── tools/                           ← LLM tool registry
│   ├── transformers/                    ← Data transformers
│   ├── protocols/                       ← Communication protocols (A2A, AG-UI)
│   └── observability/                   ← Tracing/metrics integration
│
├── cortex/                              ← AI orchestration core
│   ├── graphs/                          ← LangGraph agent definitions
│   │   ├── dispatcher.py               ← Central graph router
│   │   ├── dispatcher_agent/           ← Dispatcher agent implementation
│   │   ├── rag_retrieval/              ← RAG pipeline graph
│   │   ├── commerce/
│   │   │   ├── gardener/               ← Taxonomy classifier
│   │   │   └── matcher/                ← Product linking (RLM-based)
│   │   ├── news_engine/                ← Multi-stage news pipeline
│   │   │   ├── harvest/                ← News discovery
│   │   │   ├── archivist/              ← Content validation/filtering
│   │   │   ├── showrunner/             ← Editorial planning
│   │   │   └── agents/                 ← Generation + personas
│   │   ├── analytics/                  ← Data analytics graphs
│   │   ├── sql_analytics/              ← SQL-based analytics
│   │   └── self_healing/               ← Auto-recovery graphs
│   │
│   ├── services/graph/                  ← Graph execution service (modular)
│   │   ├── local.py                    ← LocalGraphService
│   │   └── postgres.py                 ← PostgresGraphService
│   │
│   ├── runners/                         ← Graph entry points (invoke/stream)
│   ├── subagents/                       ← Sub-agent orchestration
│   ├── prompting/                       ← Prompt management
│   ├── callbacks/                       ← LangChain callbacks
│   ├── checkpointing/                   ← State persistence
│   ├── evals/                           ← Evaluation framework
│   └── utils/                           ← Cortex utilities
│
├── service/                             ← gRPC service layer (mixin-based)
│   ├── dispatcher_service.py           ← DispatcherService (composes mixins)
│   ├── server.py                       ← gRPC server setup
│   ├── payloads.py                     ← Request/response payloads
│   ├── helpers.py                      ← Service helpers (parse_unit, etc.)
│   ├── security.py                     ← validate_dispatcher_access, token checks
│   ├── decorators.py                   ← @require_security, @log_rpc, @handle_errors
│   ├── tool_factory.py                ← Dynamic tool creation (stream-only)
│   ├── stream_executors.py            ← StreamExecutorManager (bidi stream routing)
│   └── mixins/                         ← Service decomposition
│       ├── execution.py               ← ExecuteDispatcher, StreamDispatcher, ExecuteAgent
│       ├── registration.py            ← RegisterTools, DeregisterTools
│       ├── stream.py                  ← ToolExecutorStream (bidi callback handler)
│       └── persistence.py             ← Redis-backed registration persistence
│
├── core/                                ← Configuration and infrastructure
│   ├── config/                         ← Config management
│   ├── plugins.py                      ← Plugin manifest, context, loading
│   ├── agent_config_cache.py           ← Agent permission cache (TTL)
│   └── tokens.py                       ← Token utilities
│
├── api/                                 ← REST/HTTP API layer
├── cli/                                 ← CLI commands
│   └── commands/                       ← CLI subcommands
└── utils/                               ← Shared utilities
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

## Project Registration & ToolExecutorStream

External projects connect to Router via gRPC to register their tools and graphs.
SQL tools are executed remotely in the project's process via a **bidirectional gRPC stream** — database credentials never leave the project boundary.

### Registration Flow

```
Project (Acme)                         Router
     │                                    │
     ├── RegisterTools ────────────────►  │  ← Creates tool shell (no DSN)
     │   payload: {project_id, tools,     │
     │    graph, config}                  │
     │                                    │
     ├── ToolExecutorStream ◄──────────► │  ← Bidi stream opens
     │   ready: {project_id, tools}       │
     │                                    │
     │   ... user asks question ...       │
     │                                    │
     │  ◄── execute: {tool, sql, id}  ────┤  ← Router sends SQL via stream
     │                                    │
     │  ──► result: {columns, rows, id} ──┤  ← Project executes locally, returns
     │                                    │
```

### Key Design Decisions

1. **DSN Isolation**: `DATABASE_URL` never leaves the project process. Router has no access to the database.
2. **No new ports**: ToolExecutorStream runs on the same gRPC port as RegisterTools.
3. **Stream-only execution**: SQL tools require an active ToolExecutorStream. If the stream is disconnected, the tool returns an error.
4. **Auto-reconnect**: Projects maintain the stream with automatic reconnection and exponential backoff.
5. **Project-side validation**: SQL is validated (SELECT/WITH only, forbidden keywords) in the project before execution.

### Registration Code (Project Side)

```python
# 1. Register tools and graph
stub.RegisterTools(ContextUnit(payload={
    "project_id": "my_project",
    "tools": [{
        "name": "execute_sql",
        "type": "sql",
        "config": {
            "project_id": "my_project",  # for stream routing
            "schema_description": "...",
            "read_only": True,
        }
    }],
    "graph": {"name": "my_project", "template": "sql_analytics", ...}
}))

# 2. Open bidi stream — become remote SQL executor
start_executor_stream(router_url, database_url, "my_project")
```

### StreamExecutorManager (Router Side)

Singleton that tracks active project streams:

```python
from contextrouter.service.stream_executors import get_stream_executor_manager

manager = get_stream_executor_manager()

# Check if project stream is connected
if manager.is_available("acme", "execute_analytics_sql"):
    result = await manager.execute("acme", "execute_analytics_sql", {"sql": "..."})
```

### Security

- **RegisterTools** requires `tools:register` write scope
- **ToolExecutorStream** requires `tools:execute` permission in ContextToken
- SQL validation enforced on the project side (SELECT/WITH only)
- Statement timeout and row limits applied per tool config

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

### Supported Providers (13 LLM + 2 Embedding)

| Provider | Module | Models | Use Case |
|----------|--------|--------|----------|
| **Vertex AI** | `vertex.py` | Gemini 2.0, 2.5 Pro | Primary (Google Cloud) |
| **OpenAI** | `openai.py` | GPT-5-mini, o1, o3 | General purpose, reasoning |
| **Anthropic** | `anthropic.py` | Claude Sonnet 4, Haiku | Reasoning, analysis |
| **Perplexity** | `perplexity.py` | Sonar | Web-grounded search |
| **Groq** | `groq.py` | Llama 3.3 70B | Ultra-fast inference |
| **RLM** | `rlm.py` | Recursive LM | Massive context (50k+ items) |
| **OpenRouter** | `openrouter.py` | Multi-provider | Aggregated access |
| **LiteLLM** | `litellm.py` | Universal | Adapter for any provider |
| **Local OpenAI** | `local_openai.py` | Ollama, vLLM | Development, privacy |
| **RunPod** | `runpod.py` | Custom models | GPU inference |
| **HuggingFace** | `huggingface.py` | HF Inference | HuggingFace models |
| **HF Hub** | `hf_hub.py` | HF Hub models | Direct hub access |
| **OpenAI Batch** | `openai_batch.py` | GPT batch | Async batch processing |

**Embeddings**: HuggingFace (`embeddings/huggingface.py`), Vertex AI (`embeddings/vertex.py`)

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

## Plugin Architecture

ContextRouter supports a manifest-based plugin system for extending functionality without modifying core code.

### Plugin Structure

Each plugin is a directory containing:

```
plugins/
└── my-plugin/
    ├── plugin.yaml    # Manifest (required)
    └── plugin.py      # Entry point (default)
```

### Plugin Manifest (`plugin.yaml`)

```yaml
name: my-enrichment-plugin
version: 1.0.0
description: Custom enrichment tools for product data
author: Acme Corp
requires:
  contextrouter: ">=0.9.0"
capabilities:
  - tools
  - graphs
entry_point: plugin.py
```

**Capabilities** restrict what a plugin can register:

| Capability | Allows |
|------------|--------|
| `tools` | `ctx.register_tool()` |
| `graphs` | `ctx.register_graph()` |
| `connectors` | `ctx.register_connector()` |
| `providers` | `ctx.register_provider()` |
| `transformers` | `ctx.register_transformer()` |

### Entry Point

The entry point module must export an `on_load(ctx: PluginContext)` function:

```python
# plugin.py
from langchain_core.tools import tool

def on_load(ctx):
    @tool
    def product_enrichment(query: str) -> str:
        """Enrich product data with external sources."""
        return f"Enriched: {query}"

    ctx.register_tool(product_enrichment)
```

### Plugin Loading

Plugins are loaded at CLI startup from paths in config:

```toml
# settings.toml
[plugins]
paths = ["/path/to/plugins"]
```

The `scan()` function iterates plugin subdirectories, validates manifests, checks version compatibility, and calls `on_load(ctx)`. Attempting to register without the required capability raises `PermissionError`.

---

## Adding New Functionality

See `CONTRIBUTING.md` for complete Golden Paths:

1. **Adding LLM Provider** — Create module + BUILTIN_LLMS + config + tests
2. **Adding Config Section** — Pydantic BaseSettings with env aliases
3. **Adding Cortex Graph** — StateGraph + state TypedDict + dispatcher registration
4. **Adding Tool** — `@tool_registry.register` decorator
5. **Adding Plugin** — Create subdirectory with `plugin.yaml` + `on_load(ctx)` entry point

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
| `core/plugins.py` | Plugin manifest, context, loading |
| `core/config/main.py` | Configuration management |
| `core/config/security.py` | Security policies, default permissions |
| `core/agent_config_cache.py` | Agent permission cache (TTL, ContextView) |
| `cortex/runners/chat.py` | RAG entry points (invoke/stream) |
| `service/stream_executors.py` | StreamExecutorManager — bidi stream routing |
| `service/tool_factory.py` | Dynamic tool creation (stream-only) |
| `service/mixins/stream.py` | ToolExecutorStream bidi handler |

---

## Security & Token Minting

### Agent-Aware Token Minting

When security is enabled, Router mints a `ContextToken` for each request. The token's permissions are determined by:

1. **Default permissions** from `SecurityPoliciesConfig` (config-driven)
2. **Agent max permissions** from ContextView (cached in `AgentConfigCache`, TTL 5 min)
3. **Effective = intersection** — `intersect_permissions(agent_max, default)`

```python
# In chat runners (invoke_agent / stream_agent):
token = await _mint_access_token(user_ctx, agent_id="rag-agent")

# Without agent_id — uses default permissions from config
token = await _mint_access_token(user_ctx)
```

### AgentConfigCache

Fetches agent permission profiles from ContextView's `AdminService.GetAgentConfig` and caches them in-memory:

```python
from contextrouter.core.agent_config_cache import get_agent_config_cache

cache = get_agent_config_cache()
profile = await cache.get_agent_permissions("my-agent", default_permissions=(...))
# profile.permissions — expanded tuple
# profile.allowed_tools, profile.denied_tools — tool whitelists
# profile.profile_name — e.g. "rag_full", "commerce"
```

### Permission Profiles (from ContextCore)

Predefined profiles in `contextcore.permissions.PROJECT_PROFILES`:

| Profile | Permissions |
|---------|-------------|
| `rag_readonly` | `graph:rag`, `brain:read`, `memory:read` |
| `rag_full` | + `memory:write`, `trace:write` |
| `commerce` | `graph:commerce`, `brain:read`, `brain:write` |
| `medical` | `graph:medical`, + memory, + `zero:anonymize` |
| `admin` | `admin:all` (superadmin) |

---

## Service Notes

### Dispatcher Security and API Parity

- REST and gRPC dispatcher surfaces must expose equivalent security controls.
- Tool-level allow/deny lists are enforced by a graph-level guard prior to tool execution.
- The `security_guard_node` uses `has_tool_access()` from `contextcore.permissions` for token-based enforcement, falling back to `allowed_tools`/`denied_tools` state lists.
- `extract_tool_names(token.permissions)` pre-populates `allowed_tools` from the token at state initialization.
- `check_tool_scope()` classifies tool risk: `SAFE` → auto-execute, `CONFIRM` → HITL interrupt, `DENY` → block.
- HITL mechanism: when a `ToolRisk.CONFIRM` tool is invoked, `security_guard_node` pauses the graph via `langgraph.types.interrupt()` and waits for human approval (`hitl_approved`).
- All security events (blocked calls, HITL confirmations) are accumulated in `state['security_flags']` for trace enrichment.
- Security violation events are observable and auditable.

### Token as Single Point of Truth (SPOT)

Identity fields are **never** carried in request payloads. `ContextToken` is the exclusive source of truth for:

| Field | Source | Used By |
|-------|--------|---------|
| `user_id` | `token.user_id` | Trace metadata, memory personalization |
| `tenant_id` | `token.allowed_tenants[0]` | Tenant isolation, cache keys, Brain queries |

**Payload contract**: `ExecuteDispatcherPayload` and `ExecuteAgentPayload` contain **only** execution context — messages, agent_id, config, platform. No `user_id`, `tenant_id`, or `permissions`.

```python
# Router resolves tenant from token, never from payload
def _resolve_tenant_id(token) -> str:
    if token and token.allowed_tenants:
        return token.allowed_tenants[0]
    return "default"
```

This eliminates token-payload conflicts and prevents privilege escalation via payload injection.

### Graph Access Control

- RAG graph runner checks `has_graph_access(token.permissions, "rag")` before invocation.
- Tokens are minted with `SecurityPoliciesConfig.default_permissions` which includes `graph:rag` by default.
- Graph-level access denial raises `PermissionError` before execution.

### Dispatcher Runtime Memory and Caching

- Redis-backed session memory supports continuity and multi-instance behavior.
- Cache strategy is layered: in-memory hot cache plus Redis persistence.
- TTL selection must follow data freshness and sensitivity requirements.
- Tenant-aware cache keys are required for isolation.

### Dispatcher Integration Direction

- MCP/discovery and tool metadata/rate-limit/timeout capabilities are tracked as platform-hardening priorities.
- Any new tool execution path must include input validation and audit span coverage.

### Transformer and NER Guidance

- NER and transformer composition guidance remains part of service reference.
- New transformer additions should follow modular registration and envelope-based enrichment patterns.

### Trace Reflection Nodes

- Both RAG and Dispatcher graphs have `reflect` END nodes that log execution traces.
- `reflect_interaction` (RAG) captures tool_calls, token_usage, timing, and records episodes via `brain.add_episode()` and `brain.log_trace()`.
- `reflect_dispatcher` (Dispatcher) captures similar data with dispatcher-specific metadata (iterations, message count).
- Both reflect nodes enrich traces with `security_flags` from state — blocked tools, HITL events, and permission denials are visible in trace metadata.
- Reflection nodes never block — errors are caught and logged.

### Brain Memory Tools

- `brain_memory_tools.py` provides 4 LangChain tools for persistent memory:
  - `remember_episode` — record conversational episodes
  - `recall_episodes` — retrieve relevant past interactions
  - `learn_user_fact` — persist user preferences/facts
  - `recall_user_facts` — retrieve known user facts
- Tools are auto-registered in `discover_all_tools()` and available to the dispatcher agent.

---

## Links

- **Documentation**: https://contextrouter.dev
- **Repository**: https://github.com/ContextUnity/contextrouter
- **API Reference**: https://contextrouter.dev/reference/

---

*Last updated: February 2026*


---

## Component Docs: services/contextrouter/src/contextrouter/cortex/README.md

# Brain (LangGraph) pipeline

This package contains the shared agent “cortex” used by both the web UI and Telegram.

The cortex is implemented as a LangGraph `StateGraph`:

- **Graph wiring**: `contextrouter/cortex/graphs/brain.py` (central router) and `contextrouter/cortex/graphs/rag_retrieval.py` (RAG retrieval graph)
- **Nodes (graph steps)**: `contextrouter/cortex/nodes/`
- **Runners (host entrypoints)**: `contextrouter/cortex/runners/` (stream/invoke helpers)
- **State schema**: `contextrouter/cortex/state.py`
- **Generic prompts (cortex-owned)**: `contextrouter/cortex/prompting/`

## Ingestion graphs (dynamic recipes)

Ingestion can be wired as a full pipeline (`cortex/graphs/rag_ingestion.py`) or built dynamically from
a declarative recipe (`IngestionRecipe`) which selects a subset of stages (e.g. `preprocess` only,
`preprocess+taxonomy`, etc.).

The recipe is intentionally *not code* and is safe to accept from transports.

## High-level flow

The graph executes the following steps for each user message:

1) `extract_query`
2) `detect_intent`
3) Conditional routing via `should_retrieve`
4) `retrieve` (optional)
5) `suggest`
6) `generate`

In short:

- If the intent is **rag_and_web**, we retrieve from **Vertex AI Search** and (optionally) from **Google CSE** (site-limited).
- If retrieval yields **zero docs**, we return a **no-results response** generated with a dedicated no-results prompt.

## Nodes (what each one does)

### `extract_query`

- Reads the latest `HumanMessage` from `state.messages`.
- Writes:
  - `user_query` (string)
  - `should_retrieve` (bool)
  - initializes defaults (`intent=rag_and_web`, empty `retrieved_docs`, etc.)

### `detect_intent`

- Uses `gemini-2.0-flash-lite` to classify intent.
- Possible intents are:
  - `rag_and_web` - questions requiring retrieval from sources
  - `translate` - translation requests
  - `summarize` - summarization requests
  - `rewrite` - rewriting/editing requests
  - `identity` - questions about the assistant itself ("Who are you?", "What can you do?")

**Identity intent** (no retrieval):

- Detects self-referential questions about the assistant: "Who are you?", "What can you do?", "Tell me about yourself", "Are you an AI?"
- Skips RAG retrieval entirely (no Vertex Search, no web search)
- Uses `IDENTITY_PROMPT` with `style_prompt` context to generate a contextual response
- Prevents irrelevant RAG results when user asks about the assistant vs. philosophical concepts

Taxonomy + Graph + Ontology integration (runtime):

**Taxonomy enrichment**:

- **Before** the LLM call, `detect_intent` loads `taxonomy.json` (cached) and appends a **small taxonomy context** to the system prompt:
  - up to N top-level category names (default N=20)
  - a few example synonym mappings (from `canonical_map`)
- **After** the LLM call, `detect_intent` derives per-request taxonomy tags by matching the current user query against the taxonomy `canonical_map`:
  - `taxonomy_concepts`: canonical concepts detected in the query
  - `taxonomy_categories`: categories for those concepts (via graph service lookup)

**Graph facts (Path B retrieval)**:

- `retrieve` uses `GraphService.get_facts(taxonomy_concepts)` to fetch explicit relationship facts
- Facts are **ontology-filtered**: only relations marked as `runtime_fact_labels` in `ontology.json` are emitted
- Facts are **non-citation** background knowledge added to the RAG prompt (separate from Vertex Search citations)

**Retrieval query strengthening**:

- `detect_intent` returns `retrieval_queries` (1-3 short queries derived from `cleaned_query`)
- When `taxonomy_concepts` are detected, a compact concept query is added to strengthen retrieval
- This ensures Vertex Search benefits from taxonomy normalization even if the user query uses synonyms

Example (how it works):

- **taxonomy canonical_map**: `"pma" -> "Positive Mental Attitude"`, `"autosuggestion" -> "Autosuggestion"`
- **user query**: "How do I build PMA and use autosuggestion daily?"
- **result**:
  - `taxonomy_concepts=["Positive Mental Attitude", "Autosuggestion"]`
  - `taxonomy_categories=[...]` (resolved via GraphService)
  - `retrieval_queries=["How do I build Positive Mental Attitude", "use Autosuggestion daily", "PMA autosuggestion"]` (original + concept-strengthened)
  - `graph_facts=["Fact: Positive Mental Attitude CAUSES Success", "Fact: Autosuggestion REQUIRES Repetition"]` (ontology-filtered)

What if there’s no match with “top 20 categories”?

- The “top 20 categories” list is only a **prompt hint** for the LLM.
- Concept matching uses the **full** `canonical_map` (not limited to 20), so a concept can still be detected even if its category is not listed in the top 20.

### `should_retrieve` (routing)

This is the conditional router for the graph.

- If `intent!=rag_and_web` -> route directly to `suggest` (then `generate`)
- If `intent=rag_and_web` and `should_retrieve=True` and we have no docs yet -> route to `retrieve`
- Otherwise -> route to `suggest` (then `generate`)

### `retrieve` (Vertex AI Search + Graph Facts + Reranking)

**Path A (Vector Search)**:
- Runs searches in Vertex AI Search (book/video/qa) for each query in `retrieval_queries`.
- If web_allowed_domains is defined, runs web search for all domains.
- **Reranking**: After retrieval, documents are reranked per source type using the Vertex AI Ranking API. Reranking runs in parallel for all source types (book, video, qa, web).

**Path B (Graph Facts)**:
- Uses `GraphService.get_facts(taxonomy_concepts)` to fetch explicit relationship facts
- Facts are **ontology-filtered** (only `runtime_fact_labels` from `ontology.json` are emitted)
- Facts are added to state as `graph_facts` (non-citation background knowledge)
- Facts are injected into the RAG prompt via `build_rag_prompt(graph_facts=...)`

Web citations:

- Web search results are represented as `source_type="web"` docs.
- Citations include `type="web"` with:
  - `title`
  - `url`
  - `summary` (snippet or extracted page text)
- The citations builder limits web citations to **3 unique URLs** per run.

### `suggest`

- Generates optional “search suggestions” (used only for `intent=rag_and_web`).

### `generate`

- Produces the final assistant message.

Key behaviors:

- If `intent=rag_and_web` and there are **no retrieved docs**, we generate a **no-results response** using an LLM call with suggestions for alternative queries.

Web sources:

- We do **not** append a `Sources:` section to the answer text.
- URLs/titles are carried via `web` citations and rendered in the UI (Web tab).

No-results prompt:

- When `intent=rag_and_web` and there are 0 retrieved docs, the brain generates a no-results message using `gemini-2.0-flash-lite`.
- The prompt template can be overridden by the host via the input state field `no_results_prompt`.
- Host applications are expected to supply product tone/theme in `no_results_prompt`.

## Debugging

### `DEBUG_PIPELINE=1`

Emits structured per-node logs so you can see how the query is processed:

- `PIPELINE extract_query | ...`
- `PIPELINE detect_intent.in/out | taxonomy_concepts=... taxonomy_categories=... retrieval_queries=...`
- `PIPELINE route | ...`
- `PIPELINE retrieve.in | user_query=... retrieval_queries=...`
- `PIPELINE retrieve.graph_facts | facts=N concepts=[...] sample_facts=[...]`
- `PIPELINE retrieve.out | docs=N books=N videos=N qa=N web=N citations=N`
- `PIPELINE retrieve.fallback_to_web | ...`
- `PIPELINE suggest.in | ...`
- `PIPELINE generate.in | intent=... retrieved_docs=N citations=N`
- `PIPELINE generate.out | assistant_chars=N web_sources=N`
- `PIPELINE generate.no_results | ...`

### `DEBUG_WEB_SEARCH=1`

Logs a safe preview of raw CSE results (title/url/snippet) and the kept list.

### Filter-to-zero diagnostics (always on)

If CSE returns results but we keep 0 after filtering, we log a warning with:

- sample rejected host/url pairs
- sample invalid/missing links

This is logged even when `DEBUG_WEB_SEARCH=0`.



---

## Component Docs: services/contextrouter/src/contextrouter/cortex/graphs/README.md

# Cortex Graphs

This directory contains **graph definitions** (topology and wiring).

Business logic lives in:
- `contextrouter/cortex/steps/` (pure-ish step functions)
- `contextrouter/modules/` (capabilities: providers/connectors/transformers)

## Structure

```
graphs/
├── dispatcher.py         # Central graph selection (by config/registry)
├── rag_retrieval.py      # RAG pipeline (retrieve → generate)
│
└── commerce/             # Commerce domain (subgraph architecture)
    ├── graph.py          # CommerceGraph (main entry)
    ├── state.py          # CommerceState
    ├── chat/             # LLM intent detection
    ├── gardener/         # Taxonomy enrichment
    ├── lexicon/          # Content generation
    └── matcher/          # Product matching
```

## `dispatcher.py` (central graph selection)

Dispatches to the correct graph based on configuration.

```python
from contextrouter.cortex.graphs import compile_graph

# Use config (router.graph setting)
graph = compile_graph()

# Or explicit graph
graph = compile_graph("commerce")
```

**Priority:**
1. `router.override_path` — custom Python path (power-user)
2. `graph_registry` — registered via `@register_graph`
3. Built-in: `rag_retrieval`, `commerce`

## `commerce/` (Commerce domain)

Commerce graphs use subgraph architecture:

```python
from contextrouter.cortex.graphs.commerce import build_commerce_graph

# Programmatic access
graph = build_commerce_graph()
result = await graph.ainvoke({"intent": "enrich", ...})

# Chat mode (LLM intent detection)
from contextrouter.cortex.graphs.commerce import invoke_chat
result = await invoke_chat("Classify products from Vysota")
```

### Subgraphs

| Subgraph | Intent | Purpose |
|----------|--------|---------|
| `gardener` | `enrich` | Taxonomy, NER, KG enrichment |
| `lexicon` | `generate_content` | AI content generation |
| `matcher` | `match_products` | Product deduplication |
| `chat` | (wrapper) | LLM intent detection |

## `rag_retrieval.py` (RAG pipeline)

Handles chat/QA with retrieval:

```
START → extract_query → fetch_memory → detect_intent
                                           ↓
                              [should_retrieve?]
                              ↓              ↓
                          retrieve      →  generate
                              ↓              ↓
                          suggest       →  reflect → END
```

**Typical invocation** via runners:
```python
from contextrouter.cortex.runners.chat import stream_agent, invoke_agent
```

## Registering Custom Graphs

```python
from contextrouter.core.registry import register_graph

@register_graph("my_custom")
def build_my_graph():
    workflow = StateGraph(MyState)
    # ... add nodes, edges
    return workflow.compile()
```

Then use via config:
```
router.graph = "my_custom"
```



---

## Component Docs: services/contextrouter/src/contextrouter/modules/models/README.md

# Models (Multimodal LLMs + Embeddings)

This package defines the **multimodal model registry contract** used by the cortex and other modules. The interface supports text, image, and audio inputs with **strict capability-based fallback**.

## Multimodal Interface

Models use a **unified multimodal contract** that accepts text, image, and audio parts:

```python
from contextrouter.modules.models.types import ModelRequest, TextPart, ImagePart

# Text-only request
request = ModelRequest(
    parts=[TextPart(text="Hello, world!")],
    system="You are a helpful assistant",
    temperature=0.7,
)

# Multimodal request (text + image)
request = ModelRequest(
    parts=[
        TextPart(text="What's in this image?"),
        ImagePart(mime="image/jpeg", data_b64="...", uri="https://example.com/image.jpg")
    ]
)
```

## Model Registry & Fallback

### Model Keys

Models are selected by a **registry key** of the form: `"<provider>/<name>"`

Examples:
- `vertex/gemini-2.5-flash` (multimodal, Google)
- `openai/gpt-5.1` (multimodal, OpenAI)
- `anthropic/claude-opus-4.5` (text-only, Anthropic)
- `openrouter/openai/gpt-5.1` (OpenRouter, OpenAI-compatible)
- `local/llama3.1` (Ollama, OpenAI-compatible)
- `local-vllm/meta-llama/Llama-3.1-8B-Instruct` (vLLM, OpenAI-compatible)

### Fallback System

Models support **strict capability-based fallback**:

```python
from contextrouter.modules.models.registry import model_registry

# Get model with fallback
model = model_registry.get_llm_with_fallback(
    key="vertex/gemini-2.5-flash",
    fallback_keys=["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"],
    strategy="fallback",  # sequential
)
```

**Fallback Rules:**
- Only models supporting **all required modalities** are considered
- No automatic conversion (e.g., image → text description)

### Fallback Strategies

- **`fallback` (sequential)**: try candidates in order until one succeeds.
- **`parallel`**: run all candidates concurrently and return the first success (**generate only**).
- **`cost-priority`**: same mechanics as `fallback`; you must order your `fallback` list cheapest → most expensive.

**Streaming rule:** streaming always behaves like **sequential fallback** — we never switch providers mid-stream.

## Providers

### Built-in Providers

| Provider | Key Pattern | Modalities | Notes |
|----------|-------------|------------|-------|
| **Vertex AI** | `vertex/*` | Text + Image + Audio + Video | Default, requires GCP credentials. Gemini 1.5/2.5 models are multimodal. |
| **OpenAI** | `openai/*` | Text + Image + Audio (ASR) | Requires `contextrouter[models-openai]` + `OPENAI_API_KEY`. ASR via Whisper. |
| **Anthropic** | `anthropic/*` | Text + Image | Requires `contextrouter[models-anthropic]` + `ANTHROPIC_API_KEY`. Claude supports images natively. |
| **OpenRouter** | `openrouter/*` | Text + Image | Requires `contextrouter[models-openai]` + `OPENROUTER_API_KEY`. Model-dependent capabilities. |
| **Groq** | `groq/*` | Text + Image + Audio (ASR) | Requires `contextrouter[models-openai]` + `GROQ_API_KEY`. Ultra-fast Whisper ASR. |
| **RunPod** | `runpod/*` | Text + Image | Requires `contextrouter[models-openai]`. OpenAI-compatible chat; custom workers can do more. |
| **HF Hub (Remote)** | `hf-hub/*` | Text + Image + Audio (task-dependent) | Requires `contextrouter[models-hf-hub]`. Depends on task: ASR, VQA, image-to-text. |
| **Local (vLLM/Ollama)** | `local/*`, `local-vllm/*` | Text + Image | Requires `contextrouter[models-openai]`. Vision models support images. |
| **HuggingFace Transformers** | `hf/*` | Task-dependent (Text/Image/Audio) | Local inference. Task controls modality: text-gen, ASR, image-classification. |
| **RLM (Recursive)** | `rlm/*` | Text | Wraps any LLM with recursive REPL. For massive contexts (50k+ items). Requires `pip install rlm`. |
| **LiteLLM** | `litellm/*` | - | **Stub only** (not implemented). |

### HuggingFace Transformers ⚠️

**WARNING:** HuggingFace local inference requires heavy dependencies:

```bash
uv add contextrouter[hf-transformers]
```

**Limitations:**
- Requires `torch` + `transformers` (large installation)
- Heavy models can be very slow / memory-hungry depending on your hardware

**Use cases:**
- CPU-based development/testing
- Small / medium models for local dev (e.g., `hf/distilgpt2`, `hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Offline environments

**Do NOT use for:**
- Production inference
- Large models
- GPU workloads (use vLLM/TGI instead)

### RLM (Recursive Language Models) 🆕

**RLM** wraps any base LLM with recursive REPL capabilities for processing **massive contexts** (50k+ items) that would cause context degradation in standard LLM calls.

**Reference:** [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) | [GitHub](https://github.com/alexzhang13/rlm)

**Key Benefits:**
- GPT-5-mini with RLM **outperforms GPT-5** on long-context tasks
- Context stored as Python variable, not in prompt
- Model can `grep`, `filter`, `iterate`, and recursively analyze
- 60-70% cost reduction for bulk processing

**Installation:**
```bash
pip install rlm  # or: uv add rlm
```

**Usage:**
```python
from contextrouter.modules.models import model_registry

# Create RLM-wrapped model
model = model_registry.create_llm(
    "rlm/gpt-5-mini",  # Uses GPT-5-mini as base
    config=config,
    environment="docker",  # Isolated execution (recommended for production)
)

# Use for bulk operations
response = await model.generate(ModelRequest(
    system="You are a product matching expert.",
    parts=[TextPart(text="""
Variables available:
- supplier_products: 50,000 items
- site_products: 10,000 items

Write code to match products efficiently.
""")],
))
```

**Use Cases:**
- **Product Matching**: 50k supplier → 10k site catalog
- **Taxonomy Classification**: Navigate 1000+ category tree
- **Bulk Normalization**: Process 50k product names
- **Knowledge Graph Extraction**: Extract relations from large datasets

**Environment Options:**
| Environment | Use Case | Safety |
|-------------|----------|--------|
| `local` | Development | Low (same process) |
| `docker` | Production | High (isolated container) |
| `modal` | Cloud scaling | High |
| `prime` | High-performance cloud | High |

### LiteLLM (stub)

`litellm/*` exists as a **stub provider** (raises `NotImplementedError`).

Reasons:
- We prefer explicit providers (Vertex/OpenAI/Anthropic/OpenRouter/local) for clearer debugging and control.
- LiteLLM would add another abstraction layer that can complicate streaming/multimodal/error mapping.
- Observability/cost tracking is handled via Langfuse + our own normalized `UsageStats`.

### Running Local Models (vLLM / Ollama)

**vLLM** (OpenAI-compatible server):

You can run vLLM via Docker (recommended) or directly (`uv add vllm` + GPU drivers).

Example direct run:
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

- Set `LOCAL_VLLM_BASE_URL=http://localhost:8000/v1`
- Use `local-vllm/meta-llama/Llama-3.1-8B-Instruct` (any model ID supported by vLLM)

**Ollama** (OpenAI-compatible):

```bash
ollama serve
ollama pull llama3.1
```

- Set `LOCAL_OLLAMA_BASE_URL=http://localhost:11434/v1`
- Use `local/llama3.1`

+## Performance Considerations

### Local vs Remote Models

-   **Remote API models** (Vertex, OpenAI, Anthropic): High reliability, low setup effort, fast inference, multimodal support, and high accuracy for complex tasks (like JSON formatting).
-   **Aggregator API models** (OpenRouter): Access to hundreds of models via a single API; reliability depends on the specific provider.
-   **Local Model Servers** (vLLM, Ollama): High privacy, no per-token costs, but requires managing your own hardware (GPU/RAM). Good for text generation and random creative tasks.
-   **Local Libraries** (HuggingFace Transformers): Best for specialized small models (STT, classification, embeddings) running directly in your application process without an external server.

## Best Practices & Recommendations
+
+### Structured Output (JSON)
+
+For tasks that require high-quality structured output (e.g., `intent`, `suggestions`, ingestion stages):
+
+*   **Recommended**: `vertex/gemini-2.5-flash-lite` or better.
+*   **Warning**: Local models (vLLM/Ollama) often have difficulty maintaining strict JSON formatting for complex schemas. Using local models for these tasks may lead to parsing errors.
+
 ## Configuration

```toml
[models]
default = "vertex/gemini-2.5-flash"

[models.rag.intent]
model = "vertex/gemini-2.5-flash-lite"
fallback = ["anthropic/claude-haiku-4.5"]
strategy = "fallback"

[models.rag.generation]
model = "vertex/gemini-2.5-flash"
fallback = ["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"]
strategy = "fallback"

[models.rag.no_results]
model = "vertex/gemini-2.5-flash-lite"
fallback = ["anthropic/claude-haiku-4.5"]
strategy = "fallback"
```

### Environment Variables

API keys are **never stored in config**, only via environment:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `RUNPOD_API_KEY`
- `RUNPOD_BASE_URL`
- `HF_TOKEN`
- `OPENROUTER_BASE_URL` (optional)
- `LOCAL_OLLAMA_BASE_URL`, `LOCAL_VLLM_BASE_URL`

## Development & Testing

### Adding New Providers

1. Implement `BaseModel` subclass with proper `capabilities`
2. Register with `@model_registry.register_llm("provider", "name")`
3. Add optional dependencies to `pyproject.toml`
4. Update this README

### Testing Multimodal Features

Use the test utilities in `tests/unit/` for:
- Capability filtering validation
- Fallback strategy testing
- Stream safety verification



---

## Component Docs: services/contextrouter/src/contextrouter/modules/providers/storage/README.md

# Brain Storage Provider

The Brain provider delegates retrieval and storage operations to the `ContextBrain` service. It is the primary way `ContextRouter` interacts with indexed knowledge.

## Configuration

The provider supports two modes of operation, controlled by environment variables or configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `BRAIN_MODE` | Integration mode: `local` or `grpc` | `local` |
| `BRAIN_GRPC_ENDPOINT` | Address of the Brain gRPC service (required for `grpc` mode) | `localhost:50051` |
| `BRAIN_DATABASE_URL` | Database connection string (required for `local` mode) | - |

### Local Mode (`local`)

In this mode, `ContextRouter` imports `contextbrain` as a library and runs the `BrainService` logic within its own process. This is ideal for monolithic deployments or local development where you want to avoid network overhead.

**Requirements**:
-   `contextbrain` must be installed in the same environment.
-   Database credentials must be provided directly to the Router process.

### gRPC Mode (`grpc`)

In this mode, `ContextRouter` acts as a client and sends requests to a standalone `ContextBrain` service via gRPC. This is ideal for microservices architectures and allows scaling Brain independently from Router.

**Requirements**:
-   `ContextBrain` service must be running and accessible at the specified endpoint.
-   Protos must be compiled in `contextcore`.

## Usage

To use this provider in the RAG pipeline, set the following in your `.env`:

```bash
RAG_PROVIDER=brain
BRAIN_MODE=grpc
BRAIN_GRPC_ENDPOINT=10.0.0.5:50051
```



---

