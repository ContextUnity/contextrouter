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
