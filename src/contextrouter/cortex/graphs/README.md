# Cortex Graphs

This directory contains **graph wiring** (topology) only.

Business logic lives in:
- `contextrouter/cortex/steps/` (pure-ish step functions)
- `contextrouter/modules/` (capabilities: providers/connectors/transformers)

## `brain.py` (central dispatcher)

`brain.py` is the explicit entrypoint that selects which graph to compile/run.
It uses `Config.router.graph` as a simple key

## `rag_retrieval.py` (chat / RAG retrieval graph)

**Purpose**: handle a user chat turn end-to-end (intent → optional retrieval → generation).

- **Build/compile**: `build_graph()` / `compile_graph()`
- **Typical invocation**: via runners (preferred)
  - `contextrouter.cortex.runners.chat.stream_agent(...)` (LangGraph `astream_events(v2)` passthrough)
  - `contextrouter.cortex.runners.chat.invoke_agent(...)`
- **Steps** live in: `contextrouter/cortex/steps/rag_retrieval/`
- **Node wrappers** (thin) live in: `contextrouter/cortex/nodes/rag_retrieval/`

This graph is currently “named”/static (selected by `router.graph`), not recipe-built.

## `rag_ingestion.py` (ingestion graphs)

**Purpose**: build/update knowledge assets (clean text, taxonomy/ontology/graph, shadow/export/deploy/report).

### Two ways to wire ingestion

- **Default full pipeline**: `build_graph()` / `compile_graph()`
- **Dynamic recipe graph**: `build_graph_from_recipe(recipe)` / `compile_graph_from_recipe(recipe)`

### Why recipes exist

Transports (API/job runners) must not send Python objects/code. Instead, they can send a
JSON payload that selects a safe subset of stages.

#### `IngestionRecipe`

Recipe controls **topology** only:

- `stages`: subset of allowed stage names (in canonical order)
- `allow_unsafe`: bypass dependency validation (power-user mode)

#### `IngestionJobSpec` (API-friendly)

Job spec is a single JSON payload:

- `recipe`: `IngestionRecipe`
- `input`: input params (subset of `IngestionInputState`, e.g. `ingestion_config_path`, `only_types`, `overwrite`, `skip_*`, `workers`)

This format is:
- JSON-safe
- easy to persist as a job payload
- easy to replay

### Compiled graph cache

`compile_graph_from_recipe(recipe)` memoizes compiled graphs **in-process** using a stable
`recipe_cache_key(recipe)`. This avoids repeated compilation for common recipes like:
`preprocess-only`, `preprocess+taxonomy`, or `full`.
