# ContextRouter — Agent Instructions

AI Gateway and Agent Orchestration: LangGraph agents, LLM provider routing, RAG pipeline, tool dispatch, and protocol adapters.

## Entry & Execution
- **Workspace**: `services/router/`
- **Run**: `uv run python -m contextunity.router` (gRPC) or `contextunity.router serve` (API)
- **Tests**: `uv run --package contextunity-router pytest`
- **Lint**: `uv run ruff check .`

## Code Standards
You MUST adhere to [Code Standards](../../.agent/skills/code_standards/SKILL.md): 400-line limit, Pydantic strictness, `mise` sync, Ruff compliance.

## Architecture

```
src/contextunity/router/
├── modules/
│   ├── models/                    # LLM registry + providers
│   │   ├── registry.py            # BUILTIN_LLMS (CRITICAL!)
│   │   ├── types.py               # ModelRequest, error types
│   │   └── llm/                   # 12 providers: openai, anthropic, vertex, groq, perplexity, rlm, ...
│   ├── retrieval/rag/             # RAG pipeline (3 modules)
│   ├── providers/storage/         # BrainProvider
│   ├── tools/                     # LLM tool registry
│   │   ├── security_tools.py      # Shield tools (auto-discovered)
│   │   ├── privacy_tools.py       # Zero tools (auto-discovered)
│   │   └── brain_memory_tools.py  # Memory tools
│   ├── connectors/                # Data source connectors
│   └── protocols/                 # A2A, AG-UI adapters
│
├── cortex/                        # AI orchestration core
│   ├── graphs/                    # LangGraph agents
│   │   ├── dispatcher.py          # Central graph router
│   │   ├── dispatcher_agent/      # Dispatcher agent
│   │   ├── rag_retrieval/         # RAG pipeline graph
│   │   ├── commerce/gardener/     # Taxonomy classifier
│   │   ├── commerce/matcher/      # Product linking (RLM)
│   │   ├── news_engine/           # Multi-stage news pipeline
│   │   ├── analytics/             # Data analytics
│   │   └── sql_analytics/         # SQL-based analytics
│   ├── services/graph/            # GraphService (modular)
│   ├── runners/                   # Graph entry points
│   └── prompting/                 # Prompt management
│
├── service/                       # gRPC service (mixin-based)
│   ├── dispatcher_service.py      # DispatcherService (composes mixins)
│   ├── server.py                  # gRPC server setup
│   ├── payloads.py                # Request/response payloads
│   ├── security.py                # validate_dispatcher_access
│   ├── stream_executors.py        # StreamExecutorManager (bidi routing)
│   └── mixins/                    # execution, registration, stream, persistence
│
├── core/                          # Configuration and infrastructure
│   ├── config/                    # RouterConfig (Pydantic sections)
│   ├── plugins.py                 # Plugin manifest, loading
│   ├── registry.py                # Graph + tool registry
│   ├── flow_manager.py            # Conversation flow management
│   ├── memory.py                  # Memory management
│   ├── brain_token.py             # Brain token utilities
│   ├── state.py                   # Core state management
│   ├── types.py                   # Core type definitions
│   ├── interfaces.py              # Abstract interfaces
│   └── exceptions.py              # ContextrouterError hierarchy
│
├── api/                           # REST/HTTP API (FastAPI)
└── cli/                           # CLI commands
```

## Strict Boundaries
- **Model Registry ONLY**: All LLM calls go through `model_registry`. FORBIDDEN: direct provider imports in `cortex/graphs/`.
- **ContextUnit Protocol**: All data crossing module boundaries uses `ContextUnit` from `contextunity.core`.
- **Project Isolation**: No hardcoded project-specific logic. Use plugin framework or tool registrations.
- **Config-First**: Use `RouterConfig` classes. No direct `os.environ`.
- **No RAG Logic**: Router handles routing and orchestration; Brain handles retrieval storage.
- **Factory + Strict Config**: All graphs MUST use Pydantic config injection, never hardcoded models in nodes.

## Model Registry (MANDATORY)

```python
from contextunity.router.modules.models import model_registry
from contextunity.router.core import get_core_config

config = get_core_config()
model = model_registry.get_llm_with_fallback(
    key=config.models.default_llm,
    fallback_keys=config.models.fallback_llms,
    strategy="fallback",
    config=config,
)
```

### Error Handling for Fallback
Providers MUST convert SDK errors to typed exceptions:
```python
from ..types import ModelQuotaExhaustedError, ModelRateLimitError
# QuotaExhausted → immediate fallback, RateLimited → fallback with delay
```

### Reasoning Models (gpt-5, o1, o3)
```python
if is_reasoning_model:
    bind_kwargs = {"max_completion_tokens": 8000}  # Temperature must be 1
else:
    bind_kwargs = {"max_tokens": 2000, "temperature": 0.7}
```

## Security Architecture

### Token as Single Point of Truth (SPOT)
Identity fields (`user_id`, `tenant_id`) are **never** in request payloads. `ContextToken` is the exclusive source.

### Three-Layer Registration Security
| Layer | Check | Implementation |
|-------|-------|---------------|
| **A** | `project_id ∈ token.allowed_tenants` | Token tenant binding |
| **B** | `tools:register:{project_id}` permission | `has_registration_access()` |
| **C** | Redis project ownership record | `verify_project_owner()` |

### SecureTool
Every registered tool is wrapped in `SecureTool` with `bound_tenant` enforcement. Tools require `tool:{name}` permission.

### Shield Pre-LLM Guard
Dispatcher calls `self._guard.check_input()` before every LLM call (Enterprise). Uses SPOT pattern — end-user token propagated to Shield.

## Configuration

| Variable | Description |
|----------|-------------|
| `ROUTER_PORT` | gRPC server port (default 50052) |
| `REDIS_URL` | Service discovery and persistence |
| `CU_ROUTER_DEFAULT_LLM` | Baseline LLM (e.g. `openai/gpt-5-mini`) |
| `CU_ROUTER_FALLBACK_LLMS` | Comma-separated global fallback chain |
| `CU_BRAIN_GRPC_URL` | Override for Brain host |
| `CU_SHIELD_GRPC_URL` | Override for Shield host |

## Golden Paths

### Adding a New LLM Provider
1. Create `modules/models/llm/newprovider.py` — implement `generate()` and `stream()`
2. **CRITICAL**: Add to `BUILTIN_LLMS` in `modules/models/registry.py`
3. Add config section in `core/config/sections/`
4. Set `max_retries=0` in SDK — FallbackModel handles retries
5. Add tests: `tests/unit/test_newprovider_llm.py`

### Adding a Cortex Graph
1. Create `cortex/graphs/mygraph/` with `graph.py`, `state.py`, `steps.py`
2. Define `GraphConfig(BaseModel)` with Pydantic validation
3. Create node factories: `make_*(model_key, ...)` closures
4. Register via `@register_graph("mygraph")` or dispatcher config
5. Add tests using mock gRPC streams

### Adding a Config Section
1. Create Pydantic `BaseSettings` in `core/config/sections/`
2. Wire into `RouterConfig` composition
3. Add env var aliases

### Adding a Tool
1. Create tool function in `modules/tools/`
2. Register via `@tool_registry.register`
3. Ensure `SecureTool` wrapping with proper permissions

## Agent Wrapper Contract
- Node functions MUST return `dict` (partial state update)
- Async steps MUST be `await`ed — returning coroutine crashes LangGraph

## Further Reading
- [Astro Docs: ContextRouter](../../docs/website/src/content/docs/router/)
- [Router Operations Skill](../../.agent/skills/router_ops/SKILL.md)
