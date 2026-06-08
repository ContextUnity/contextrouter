# ContextRouter

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

ContextRouter is the **AI Gateway and Agent Orchestration** layer of the [ContextUnity](https://github.com/ContextUnity) ecosystem. Built on LangGraph, it routes LLM requests across 13 providers with automatic fallback, orchestrates multi-step agent workflows, and delegates memory to Brain.

---

## What is it for?

- **Agent Orchestration** — LangGraph state machines for complex reasoning workflows
- **LLM Provider Routing** — OpenAI, Anthropic, Vertex AI, Groq, Perplexity, local models with automatic fallback
- **Graph Compiler** — Declarative YAML templates compiled into executable LangGraph pipelines
- **RAG Pipeline** — Retrieval-Augmented Generation with Brain as backend
- **SQL Analytics** — Natural language to SQL with auto-verification
- **Tool Integration** — Federated tool execution via bidirectional gRPC streams
- **Protocol Adapters** — Telegram, AG-UI, A2A event formats

---

## Quick Start

```bash
# Start gRPC server
export REDIS_URL="redis://localhost:6379/0"
export CU_ROUTER_DEFAULT_LLM="openai/gpt-5-mini"
export OPENAI_API_KEY="sk-..."
uv run python -m contextunity.router

# Run tests
uv run --package contextunity-router pytest
```

---

## Architecture

```
src/contextunity/router/
├── modules/                      # Capability modules (models, tools, RAG)
├── cortex/                       # AI orchestration (LangGraph agents)
│   ├── graphs/                   # Graph definitions
│   │   ├── dispatcher.py         # Central graph router
│   │   ├── compiler/             # ⭐ Graph compiler engine
│   │   │   ├── builder.py        # build_from_template(), build_local_graph()
│   │   │   ├── template_loader.py # YAML → Pydantic validation
│   │   │   ├── topology.py       # Edge auto-completion, tool ref parsing
│   │   │   ├── node_factory.py   # Node compilation (type → executor)
│   │   │   ├── platform_registry.py # PlatformToolRegistry
│   │   │   └── platform_tools/   # 22 modules, 29 registered bindings
│   │   └── rag_retrieval/        # RAG node implementations
│   └── templates/                # ⭐ YAML graph templates (5 built-in)
│       ├── retrieval_augmented.yaml  # Unified RAG + SQL
│       ├── gardener.yaml             # Product normalisation
│       ├── enricher.yaml             # Product enrichment (also used by writer)
│       ├── rlm_bulk_matcher.yaml     # RLM-based bulk matching
│       └── news_pipeline.yaml        # News processing
├── service/                      # gRPC service (mixin-based dispatcher)
├── core/                         # Config, plugin system, registry
│   └── plugins.py                # PluginManifest + PluginContext (capability-mediated)
├── api/                          # REST/HTTP (FastAPI)
└── cli/                          # CLI commands
```

---

## Graph Compiler

ContextRouter uses a **template-driven graph architecture**. Instead of writing imperative Python wiring for each pipeline, you describe the topology in YAML:

```yaml
# cortex/templates/gardener.yaml
name: gardener
nodes:
  - name: fetch_products
    type: tool
    tool_binding: federated:export_products_for_normalization
  - name: classify
    type: tool
    tool_binding: router_classify # universal LLM capability
    config:
      taxonomy_key: taxonomy
      response_format: json
  - name: write_results
    type: tool
    tool_binding: federated:update_normalized_products
edges:
  - from: __start__
    to: fetch_products
  - from: fetch_products
    to: classify
  - from: classify
    to: write_results
  - from: write_results
    to: __end__
```

The compiler transforms this into a LangGraph `StateGraph` with security wrapping, config validation, and cycle detection.

### Platform Tools

Universal LLM capabilities that any project can compose:

| Tool | Capability |
|------|-----------|
| `router_classify` | Taxonomy / intent classification |
| `router_generate_content` | Structured content generation |
| `router_review_content` | Quality review + correction |
| `router_filter_content` | Content filtering / validation |
| `router_plan_content` | Editorial / batch planning |
| `router_match_semantic` | Semantic similarity matching |
| `router_rlm_process` | Massive context RLM execution (50k+ items) |

All tools enforce `frozen=True` + `extra="forbid"` Pydantic configs and require `router:execute` scope.

---

## Federated Toolkits

Projects handle custom side-effects (e.g. database execution) by declaring `FederatedToolkit` classes in their SDK, which the Router invokes over a secure bidirectional gRPC stream. No credentials ever leave your infrastructure. 

```yaml
# In project's contextunity.project.yaml
router:
  toolkits:
    - name: DatabaseToolkit
  graph:
    commerce_pipeline:
      toolkits:
        - name: ContentToolkit
          exclude: ["destructive_tool"]
```

---

## Plugin System

ContextRouter can be extended with **declarative YAML plugins** — self-contained directories that register custom tools, LLM providers, data connectors, or transformers without modifying core code.

> **Plugins extend Router capabilities. Manifests define graphs.** These are complementary layers with zero overlap.

```
my-plugin/
  plugin.yaml       # Manifest: name, version, capabilities
  plugin.py          # Entry point with on_load(ctx)
```

```yaml
# plugin.yaml
name: "custom-weather-tool"
version: "1.0.0"
capabilities:
  - tools
entry_point: "plugin.py"
```

```python
# plugin.py
def on_load(ctx):
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"22°C in {city}"

    ctx.register_tool(get_weather)
```

### Capabilities

| Capability | Registration Method | Use Case |
|:-----------|:-------------------|:---------|
| `tools` | `ctx.register_tool()` | Custom LLM function-calling tools |
| `providers` | `ctx.register_provider()` | New LLM provider backends |
| `connectors` | `ctx.register_connector()` | Data source adapters |
| `transformers` | `ctx.register_transformer()` | Processing stages |
| `graphs` | `ctx.register_graph()` | Project graph state machines |

Plugins are scanned from directories listed in `settings.toml` under `[plugins].paths`. See the [Plugin System docs](../../website/src/content/docs/router/yaml-plugins.md) for the full API reference.

### Model Resolution

Models resolve through a 3-level hierarchy (first non-empty wins):

```
Per-node model  →  Graph defaults.model  →  CU_ROUTER_DEFAULT_LLM
```

Example — different models per graph, different models per node:

```yaml
default_graph: gardener
graph:
  gardener:
    template: "yaml:gardener"
    overrides:
      defaults:
        model: vertex/gemini-2.5-flash        # fast + cheap for classification

  enricher:
    template: "yaml:enricher"
    overrides:
      defaults:
        model: openai/gpt-4o                  # quality for content generation
      nodes:
        review:
          model: anthropic/claude-sonnet-4     # Claude is strong at review

  rlm_bulk_matcher:
    template: "yaml:rlm_bulk_matcher"
    overrides:
      defaults:
        model: rlm/gpt-5-mini                 # RLM wraps gpt-5-mini with REPL
```

### RLM (Recursive Language Models)

RLM is a **task-agnostic** inference paradigm that wraps any base LLM with a Python REPL environment. Instead of cramming 50k items into a context window, RLM stores data as Python variables and lets the model programmatically examine, filter, and recursively call itself.

The `rlm/` model prefix activates the wrapper: `rlm/gpt-5-mini`, `rlm/claude-sonnet`, `rlm/gemini-2.5-flash`.

**How Commerce uses RLM for product matching:**

```python
# Inside commerce matcher node — direct API call
model = model_registry.create_llm(
    "rlm/gpt-5-mini",
    config=config,
    environment="docker",
)

result = await model.generate(
    ModelRequest(
        system="You are a product matching expert...",
        parts=[TextPart(text=matching_prompt)],
        max_output_tokens=50000,
    ),
    custom_tools={
        "supplier_products": supplier_list,  # 50k items as Python variable
        "site_products": site_list,          # 10k items — NOT in prompt
    },
)
```

The same capability is available as a platform tool (`router_rlm_process`) in YAML templates — no Python required.

---

## Supported LLM Providers

| Provider | Key | Use Case |
|----------|-----|----------|
| **Vertex AI** | `vertex/gemini-2.5-flash` | Production, multimodal |
| **OpenAI** | `openai/gpt-5-mini` | General purpose |
| **Anthropic** | `anthropic/claude-sonnet-4` | Reasoning, analysis |
| **Perplexity** | `perplexity/sonar` | Web-grounded search |
| **Groq** | `groq/llama-3.3-70b-versatile` | Ultra-fast inference |
| **OpenRouter** | `openrouter/deepseek/deepseek-r1` | 100+ models |
| **Local** | `local/llama3.2` | Privacy, development |
| **RLM** | `rlm/gpt-5-mini` | Massive context (50k+ items) |

---

## Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_PORT` | `50050` | gRPC server port |
| `REDIS_URL` | — | Service discovery and persistence |
| `CU_ROUTER_DEFAULT_LLM` | `openai/gpt-5-mini` | Baseline LLM |
| `CU_ROUTER_FALLBACK_LLMS` | — | Comma-separated global fallback chain |
| `CU_BRAIN_GRPC_URL` | — | Override for Brain host |
| `CU_SHIELD_GRPC_URL` | — | Override for Shield host |

---

## Security

ContextRouter enforces multi-layer security:
- **SPOT Pattern** — `ContextToken` is the sole source of identity (no user_id/tenant_id in payloads)
- **SecureNode Wrapping** — every compiled node enforces token scope (attenuated per-node)
- **Config Immutability** — frozen Pydantic schemas block runtime injection
- **Three-Layer Registration** — tenant binding → permission check → Redis ownership verification
- **Shield Pre-LLM Guard** — input scanning for prompt injection (Enterprise)

---

## Further Reading

- **Full Documentation**: [ContextRouter on Astro Site](../../website/src/content/docs/router/)
- **Graph Compiler Deep Dive**: [Graph Compiler & Templates](../../website/src/content/docs/router/graph-compiler.md)
- **Plugin System**: [Plugin System & PluginContext API](../../website/src/content/docs/router/yaml-plugins.md)
- **Agent Boundaries & Golden Paths**: [AGENTS.md](AGENTS.md)

## License

This project is licensed under the terms specified in [LICENSE.md](LICENSE.md).
