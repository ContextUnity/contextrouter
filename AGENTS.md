# ContextRouter — Agent Instructions

AI Gateway and Agent Orchestration service (LangGraph compiler, gRPC/REST execution, registration, privacy, model routing).

**Deep architecture:** [docs/architecture/router-architecture.md](../../docs/architecture/router-architecture.md)
**Types & payloads:** [docs/architecture/type-boundaries.md](../../docs/architecture/type-boundaries.md)
**Code quality:** [docs/architecture/code-quality.md](../../docs/architecture/code-quality.md)
**Tenants & registration:** [docs/architecture/project-tenant-registration.md](../../docs/architecture/project-tenant-registration.md)
**Monorepo agent rules:** [AGENTS.md](../../AGENTS.md)

---

## Entry & verification

Run from the **monorepo root** (`contextunity/`) unless noted.

| Task | Command |
|------|---------|
| gRPC server | `uv run python -m contextunity.router` or `uv run contextrouter serve` |
| Tests | `uv run --package contextunity-router pytest services/router/tests` |
| Lint | `cd services/router && uv run ruff check src tests` |
| Types (router scope) | `uv run basedpyright services/router/src/contextunity/router --warnings` |
| Monorepo gate | `uv run basedpyright --project pyrightconfig.json --warnings` |
| Core guards (if touching shared types) | See [type-boundaries.md §8.1](../../docs/architecture/type-boundaries.md) and `packages/core/AGENTS.md` |

Workspace for edits: `services/router/` (`src/contextunity/router/`).

---

## Skill routing

| Trigger | Skill |
|---------|-------|
| Typing, JSON/gRPC, `dict[str, object]`, basedpyright | **`contract-boundaries`** (primary) → then **`type-validation`** |

**Narrowing:** LangGraph state / open dict reads → `contextunity.core.narrowing`; SDK payload keys → `sdk.payload.get_*`. [type-boundaries.md §4.5](../../docs/architecture/type-boundaries.md).
| gRPC interceptors, scopes, Shield | **`security-implementation`** |
| New graph/platform tool / template | **`acdd-feature-development`** |
| Bug / test failure | **`diagnose`** |
| Implementation loop | **`tdd`** |
| Connect external project | workflow [/connect-project-to-router](../../.agents/workflows/connect-project-to-router.md) |

---

## Package layout (agent map)

```
services/router/src/contextunity/router/
├── langchain_boundaries.py   # LangChain/LangGraph vendor calls (NOT under service/)
├── cortex/
│   ├── compiler/             # YAML → LangGraph (node_factory, platform_tools, node_executors)
│   ├── dispatcher.py         # Built-in graph entry (stream/invoke agent)
│   ├── dispatcher_agent/     # Dispatcher LangGraph + nodes/tools.py
│   ├── services/             # GraphService, RedisCheckpointSaver; dispatcher submodule
│   ├── privacy/              # PII masking store, anonymizer
│   ├── checkpoint_guards.py  # LangGraph checkpoint tuple validation
│   ├── graph_boundary.py     # Narrow LangGraph Protocol surfaces
│   ├── types.py              # GraphState, merge_graph_state_update, message reducers
│   └── tenant_scope.py
├── modules/                  # models/registry, observability, retrieval/rag, tools
├── service/                  # gRPC layer (explicit submodule imports only)
│   ├── registration_redis.py, registration_projection.py
│   └── mixins/execution/     # StreamAgent, ExecuteAgent, dispatcher execution
├── api/                      # FastAPI REST
└── cli/                      # contextrouter (legacy; see cli/README.md)
```

---

## Import boundaries (mandatory)

These rules prevent **import cycles** between gRPC `service/`, cortex compiler, and checkpoint code.

| Rule | Rationale |
|------|-----------|
| **`service/__init__.py` exports nothing** | Import `service.dispatcher_service`, `service.mixins.execution.agent`, etc. explicitly. |
| **`cortex.services` exports only `GraphService` / `get_graph_service`** | Do **not** re-export `DispatcherService` here. Import from `contextunity.router.cortex.services.dispatcher`. |
| **No `__getattr__` lazy barrel on `cortex.services`** | That pattern hid a cycle: `service` → mixins → `dispatcher` → `cortex.services` while the package was still loading (e.g. `redis_saver` tests). |
| **`langchain_boundaries.py` lives at router package root** | Tools/modules call LangChain without importing the gRPC `service` package barrel. |
| **`import contextunity.router` may use `__getattr__`** | CLI/`--help` only — lazy `stream_agent`, `get_dispatcher_service`, Langfuse helpers. Different concern from `cortex.services`. |
| **Platform tools must not import domain packages** | No `commerce/`, etc. (invariant 7 below). |

---

## Platform invariants

Follow `packages/core/AGENTS.md` for proto, exceptions, and tokens. In this service:

1. **Model registry only** — LLM calls go through `modules/models/registry`; no direct provider imports in `cortex/graphs/` or ad-hoc nodes.
2. **ContextUnit protocol** — cross-module payloads use `contextunity.core` envelope types.
3. **Project isolation** — no hardcoded project logic; plugins (`PluginContext`) or manifest-driven tools.
4. **Config-first** — `RouterConfig` / `SharedConfig`; no bare `os.getenv()` / `os.environ` in logic.
5. **No retrieval storage** — Brain stores vectors; Router orchestrates.
6. **Factory + strict config** — graphs compile via factories with Pydantic `NodeConfig`, not hardcoded models in nodes.
7. **No domain imports** in platform tools (`cortex/compiler/platform_tools/`).
8. **Named execution types** — gRPC stays L3 (`dict[str, WireValue]`); coerce once to L4 (`GraphRunConfigInput`, `RegisteredProjectConfig`) at ingress. See [type-boundaries.md](../../docs/architecture/type-boundaries.md).
9. **Typing hygiene** — no `cast()`, `Any`, or `# type: ignore` for boundary fixes; use `TypeGuard`, `match`/`isinstance`, and vendor wrappers in `langchain_boundaries.py` / `graph_boundary.py`. SQLite/Postgres rows from `fetchone()` are `object` until validated (e.g. `(str(),) | (bytes(),)` for encrypted columns in `privacy/masking/store.py`).

---

## Graph compiler

- **Templates:** `cortex/compiler/templates/` → `build_from_template()` / `TemplateDefinition`.
- **Node types (compile-time):** `llm`, `embeddings`, `agent`, `tool` — see `cortex/compiler/types.py` `Literal[...]`.
- **Federated tools:** not a fifth node type. Tool nodes use `tool_binding`; federated tools use the `federated:` namespace (or `federated_tool_map` alias resolution in `node_factory._resolve_binding` / `_resolve_manifest_tool_name`). Agent `tools:` lists use the same map.
- **Config merge:** node override → graph default → `RouterConfig`.
- **Model resolution:** per-node model → graph `defaults.model` → `CU_ROUTER_DEFAULT_LLM`.
- **Worker callbacks:** remote `ExecuteNode` only for nodes listed in the graph `router_callbacks` manifest.
- **Security:** `validation.py` (reserved names, cycles, binding namespaces); every node wrapped with `secure_node.make_secure_node()`.
- **State I/O:** prefer `state_routing.read_state_input` / `write_state_output` (`dynamic` bucket + top-level reducer keys in `STATE_TOP_LEVEL_KEYS`).

### Federated executor behavior

- Soft-skip (upstream error, missing required args) returns via `_federated_skip_update()` — sets `_last_node`, merges `intermediate_results`, and writes structured skip payload (do not silently return `{}`).
- Platform nodes should also publish `intermediate_results` on returns for traceability (`node_executors/platform.py`).

### Routing helper note

- `should_continue` in `state_routing.py` is intentionally unused in compiled graphs today; routing uses template edges and conditional executors. Do not wire it without a template contract change.

---

## LangGraph state & checkpoints

- **`merge_graph_state_update`** (`cortex/types.py`) — dict keys in `GRAPH_MERGE_DICT_KEYS` (`intermediate_results`, `dynamic`) shallow-merge; `_steps` last-write-wins per chunk.
- **`is_graph_state`** — boundary guard on merged runtime dicts (honest `is_object_dict`, not a full per-key TypedDict proof).
- **`_merge_langgraph_messages`** — centralizes LangGraph `add_messages` with typed message lists.
- **Redis checkpoints:** validate pending writes / metadata in `checkpoint_guards.py` (`match` patterns, no bare `cast`) before `RedisCheckpointSaver` persists.

---

## Stream execution & tracing

- **StreamAgent** replays `graph.astream_events` v2 into `BrainAutoTracer` via `modules/observability/astream_tracer_replay.py` (callbacks alone are unreliable on the stream path).
- **InvokeAgent** uses LangChain callbacks on the invoke path.
- **Progress SSE** uses `sanitize_for_struct` like final results (`service/mixins/execution/agent.py`).
- **StreamDispatcher / ExecuteDispatcher:** `BrainAutoTracer` appended in `DispatcherService._build_config` (`cortex/services/dispatcher.py`) with Langfuse callbacks; thread id = `session_id`.

---

## Tenant vs project

- **`project_id`:** registration identity (manifest bundle owner).
- **`allowed_tenants`:** execution/security scope on tokens and nodes — intersect at ingress (`_intersect_tenant_with_project`) and in `secure_node` / `tenant_scope.py`.
- **Do not** use `allowed_tenants` for registration introspection. **IntrospectRegistrations** filters by permissions `router:introspect:{project_id}`, `tools:register:{project_id}`, or `admin:all` — not tenant membership.

---

## Registration persistence (SSOT)

| Layer | Module |
|-------|--------|
| Redis authoritative | `service/registration_redis.py` (`registration:*`, `:hash`, `:stream`) |
| Startup restore | `service/mixins/persistence.py` → in-process `RouterRegistry` |
| Compiler projection | `service/registration_projection.py` (malformed tool rows **skipped with warning** — lossy restore; re-register if tool lists look incomplete) |
| Idempotency (H12) | Hash match → skip full re-register; **`get_or_create_project_stream_secret`** reuses BiDi secret. Non-match → new `secrets.token_urlsafe(32)` on `:stream` |

Legacy file-based registration is not SSOT. Federated tools are graph-scoped; Redis key pattern `router:registrations:{project_id}`.

---

## View ↔ Router trace contract

- Trace ids in metadata from `prepare_execution` / token `trace_id`.
- Progress/final SSE: `sanitize_for_struct` on router → client.
- Introspection: project-scoped permissions (see Tenant vs project).

---

## LLM providers & agent wrapper

- **Providers:** subclass `BaseLLM`, `max_retries=0` on SDKs, map errors to `ModelRateLimitError` / `ModelQuotaExhaustedError`, register in `BUILTIN_LLMS`.
- **Reasoning models** (`gpt-5`, `o1`, `o3`): `temperature=1`, use `max_completion_tokens` not `max_tokens`.
- **Node output:** partial state `dict` only; await async steps.
- **`add_messages` nodes:** return **new** messages (delta), not full history — `node_executors/agent.py`, dispatcher `nodes/agent.py`.

---

## Exception hygiene

Broad `except Exception` is allowed only for **graceful degrade** paths (checkpoint fallback, scanner, dedup, optional Redis init). Mark with `# graceful-degrade:` and log; re-raise or surface `ContextUnityError` for contract violations.

---

## Co-deploy

Router tenant scope requires matching `core.manifest.tenants` helpers — [project-tenant-registration.md](../../docs/architecture/project-tenant-registration.md).

---

## Workflow routing (slash commands)

| Command | Workflow |
|---------|----------|
| Connect project | [/connect-project-to-router](../../.agents/workflows/connect-project-to-router.md) |
| Contract boundaries | [/contract-boundaries](../../.agents/workflows/contract-boundaries.md) |
