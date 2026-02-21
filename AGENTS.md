# ContextRouter — Agent instructions

Gateway layer: Dispatcher, LangGraph orchestrations, RAG retrieval pipelines, LLM model registries, and the Tool Executor framework.

**License Context**: This service operates under the **Apache 2.0 Open-Source License**.

## Navigation & Entry
- **Workspace**: `services/contextrouter/`
- **Application Execution**: Run the server via `contextrouter serve`.
- **Tests**: run `uv run --package contextrouter pytest` from the monorepo root.

## Architecture Context (Current Code State)
- **Cortex (`cortex/`)**: The beating heart of the reasoning engine.
  - `graphs/`: LangGraph definitions (`dispatcher.py`, `rag_retrieval/`, `commerce/gardener/`).
  - `models/`: The central LLM registry wrapping diverse providers (`litellm`, `openai`, `anthropic`, `vertexai`, `runpod`, `hf`).
  - `tool_executor/`: Handles dynamic execution of tools defined by external projects via a bidirectional gRPC shell (`stream.py`, `shell.py`, `persistence.py`).
- **Core (`core/`)**: Configuration maps and the Plugin manifest loader (`plugins.py`).
- **Service (`service/`)**: gRPC implementation mappings, strictly validating inbound `ContextUnit` payloads in `dispatch.py` and `register.py`.
- **API (`api/`)**: Basic FastAPI layers for health checks or webhook events.

## Documentation Strategy
When modifying or extending this service, update documentation strictly across these boundaries:
1. **Technical Specifications**: `services/contextrouter/contextrouter-fulldoc.md`. Document any newly added LLM adapters, LangGraph states, or updates to the `ToolExecutorStream`.
2. **Public Website**: `docs/website/src/content/docs/router/`. For general deployment paradigms, orchestration concepts, or plugin development guides.
3. **Plans & Architecture**: `plans/router/`.

## Rules specific to ContextRouter
- **Model Agnosticism**: All LLM callsMUST be routed through `contextrouter.cortex.models`, never instantiating direct provider SDKs inside graph nodes.
- **Project Isolation**: Do not hardcode project-specific logic inside Router. Use the plugin framework (`plugins.py`) or external tool registrations to inject domain rules.
- Maintain `ContextUnit` provenance chains across all agent boundaries. Record every Graph node transition as a `trace` or `thought` inside the unit payload.


## AI Agent Rules (`rules/`)
ContextUnity uses strict AI assistant rules. You **MUST** review and adhere to the following rule files before modifying this service:
- `rules/global-rules.md` (General ContextUnity architecture and boundaries)
- `rules/contextrouter-rules.md` (Specific constraints for the **contextrouter** domain)
