# ContextRouter: Agent Development Guide

ContextRouter is a modular, LangGraph-powered "shared brain" designed for high-performance agentic workflows and multi-source knowledge orchestration.

## Core Philosophy

- **Knowledge Agnostic**: While RAG is a core capability, the brain is designed to orchestrate data from any source (Web, SQL, Vector, RSS) through bidirectional pipelines (Read/Write).
- **Platform Agnostic**: The brain should not know about HTTP, Telegram, or Web. It consumes messages and emits events.
- **Strict Separation**: Logic (cortex) vs. Infrastructure (modules/providers) vs. Raw Data (modules/connectors).
- **Bisquit Protocol**: All data passing through the pipeline is wrapped in `BisquitEnvelope` for provenance and security.
- **Registry-First**: Components are registered via decorators and loaded lazily.

## Architecture Guidelines

### Cortex (The Brain)
- Located in `src/contextrouter/cortex/`.
- Owns the `StateGraph` definition and node orchestration.
- **Node vs Step**: Nodes (classes in `nodes/`) are wrappers for registries; Steps (functions in `steps/`) contain pure business logic.

### Agent wrapper contract (agent-mode)
- **Return type**: `BaseAgent.process(...)` MUST return `dict[str, Any]` (partial state update).
- **Async steps**: If a wrapper calls an async step, it MUST `await` it. Returning a coroutine will crash LangGraph with `InvalidUpdateError`.
- **Guardrail**: Keep `tests/unit/test_agent_wrapper_contract.py` passing.

### Modules (The Capabilities)
- **Providers**: Database/Storage implementations (`IRead`, `IWrite`).
- **Connectors**: Raw data fetchers (Web, RSS, Files).
- **Models**: LLM and Embedding abstractions.
- **Protocols**: Mapping internal events to external formats (e.g., AG-UI).

## Engineering Principles

1. **No direct `os.environ`**: Use `core.config.Config` for all settings.
2. **Type Safety (Compromise Policy)**:
   - Use **TypedDict** for JSON-shaped contracts (ingestion `struct_data`, UI citation dict schemas).
   - Use **Pydantic** for runtime entities inside the brain (`cortex/models.py`: `RetrievedDoc`, `Citation`, etc.).
   - Avoid leaking `Any` across boundaries. If an external SDK returns loose objects, normalize at the boundary.
3. **Immutability**: Treat the LangGraph state as immutable; nodes return partial updates.
4. **Provenance**: Always use `envelope.add_trace("stage_name")` when transforming data.

## StructData (Ingestion + Retrieval Contracts)

### What `StructData` is
`StructData` is the canonical type for **JSON-serializable payloads** that are persisted or exported:
- In ingestion, `ShadowRecord.struct_data` (snake_case keys) and Vertex import JSONL `structData`.
- In retrieval, `RetrievedDoc.metadata` carries normalized, JSON-safe metadata from providers.

Types live in `src/contextrouter/core/types.py`:
- `StructDataPrimitive`, `StructDataValue`, `StructData`
- `coerce_struct_data(...)` for boundary normalization

### Rules
- **Use `source_type`** (snake_case) in `struct_data`. Do not introduce ad-hoc `"type"` keys.
- **Use TypedDict schemas** for per-source `struct_data` payloads:
  - `BookStructData`, `VideoStructData`, `QAStructData`, `WebStructData`, `KnowledgeStructData`
- **Boundary coercion only**: `coerce_struct_data(...)` must be called at integration boundaries (e.g., Vertex SDK parsing),
  not sprinkled through business logic.

## UI Citation Schema
The UI formatter returns camelCase dicts. Use the typed union `UICitation` (TypedDict union) from `core/types.py`
as the return type for the formatter in `modules/retrieval/formatting/`.

## Key Directories

- `core/`: Framework kernel (Config, Registry, Bisquit).
- `cortex/`: Graph orchestration and state nodes.
- `modules/retrieval/`: Search business logic (Orchestrator, Reranker, Formatter).
- `modules/protocols/agui/`: Unified UI event mapping.

## Documentation Reference
Full technical documentation is available in the `contextrouter_docs/` directory.
