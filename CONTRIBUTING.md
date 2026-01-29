# Contributing to ContextRouter

Thanks for contributing to **ContextRouter**.

## Development Setup

```bash
cd contextrouter
uv sync --dev
```

If you work on ingestion code/CLI:

```bash
uv sync --dev --extra ingestion
```

## Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

## Linting & Tests

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest tests/ -v
```

## Branching & PR Flow

### Branch naming

- **Features**: `feat/<short-topic>`
- **Fixes**: `fix/<short-topic>`
- **Chores**: `chore/<short-topic>`
- **Docs**: `contextrouter_docs/<short-topic>`
- **Refactors**: `refactor/<short-topic>`

### Merge strategy

- Prefer **Squash & merge** into `main`
- Use **Conventional Commits** style: `feat:`, `fix:`, `docs:`, etc.

### Releases

- Bump version in `pyproject.toml` (SemVer)
- Tag releases as `vX.Y.Z`

---

## Architecture Overview

```
src/contextrouter/
├── core/                    # Kernel: Config, Registry, ContextToken
├── cortex/                  # LangGraph agents and orchestration
│   ├── graphs/              # Agent graphs (dispatcher, rag, commerce, news)
│   └── steps/               # Business logic functions
├── modules/
│   ├── models/              # LLM and embedding providers
│   │   ├── registry.py      # BUILTIN_LLMS, model_registry
│   │   └── llm/             # Provider implementations
│   ├── providers/           # Storage implementations (Brain)
│   ├── connectors/          # Raw data fetchers (Web, RSS)
│   ├── protocols/           # Platform adapters (AG-UI, Telegram)
│   └── tools/               # LLM tool registry
└── protocols/               # Communication layer
```

---

## Golden Path: Adding a New LLM Provider

### Step 1: Create the Provider Module

Create `src/contextrouter/modules/models/llm/newprovider.py`:

```python
"""NewProvider LLM provider."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import ContextToken

from ..base import BaseModel
from ..registry import model_registry
from ..types import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    ModelQuotaExhaustedError,
    ModelRateLimitError,
)

logger = logging.getLogger(__name__)


@model_registry.register_llm("newprovider", "*")
class NewProviderLLM(BaseModel):
    """NewProvider LLM implementation."""

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        **kwargs: object,
    ) -> None:
        self._cfg = config
        self._model_name = (model_name or "default-model").strip()
        self._capabilities = ModelCapabilities(
            supports_text=True,
            supports_image=False,
            supports_audio=False,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        provider_info = ProviderInfo(
            provider="newprovider",
            model_name=self._model_name,
            model_key=f"newprovider/{self._model_name}",
        )
        
        try:
            # Your implementation here
            ...
        except Exception as e:
            # Convert provider errors for proper fallback
            if "quota" in str(e).lower():
                raise ModelQuotaExhaustedError(str(e), provider_info=provider_info)
            if "rate" in str(e).lower():
                raise ModelRateLimitError(str(e), provider_info=provider_info)
            raise

    async def stream(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        # Streaming implementation
        ...
```

### Step 2: Add to BUILTIN_LLMS

⚠️ **CRITICAL**: Add the provider to `BUILTIN_LLMS` in `modules/models/registry.py`:

```python
BUILTIN_LLMS: dict[str, str] = {
    # ... existing providers ...
    "newprovider/*": "contextrouter.modules.models.llm.newprovider.NewProviderLLM",
}
```

Without this, the provider won't be found at runtime!

### Step 3: Add Configuration

Create `core/config/sections/newprovider.py`:

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class NewProviderConfig(BaseSettings):
    api_key: str = Field(default="", alias="NEWPROVIDER_API_KEY")
    base_url: str = Field(default="https://api.newprovider.com", alias="NEWPROVIDER_BASE_URL")
    
    model_config = {"env_prefix": ""}
```

Add to `core/config/__init__.py`:

```python
from .sections.newprovider import NewProviderConfig

class Config:
    # ...
    newprovider: NewProviderConfig = Field(default_factory=NewProviderConfig)
```

### Step 4: Add Tests & Docs

- Create `tests/unit/test_newprovider_llm.py`
- Update `docs/router/src/content/docs/models/llm.md` in docs repository

### Common Pitfalls

1. **Forgot BUILTIN_LLMS** → `KeyError: "llms: unknown key 'newprovider/model'"`
2. **Reasoning models** → Use `max_completion_tokens` (not `max_tokens`) for gpt-5/o1/o3
3. **Quota errors not caught** → Raise `ModelQuotaExhaustedError` for immediate fallback
4. **SDK retries conflict** → Set `max_retries=0` in SDK, let FallbackModel handle it

---

## Golden Path: Adding a Configuration Section

### Step 1: Create Config Section

Create `core/config/sections/myfeature.py`:

```python
from pydantic_settings import BaseSettings
from pydantic import Field


class MyFeatureConfig(BaseSettings):
    """Configuration for MyFeature."""
    
    enabled: bool = Field(default=False, alias="MYFEATURE_ENABLED")
    timeout_sec: int = Field(default=30, alias="MYFEATURE_TIMEOUT")
    
    model_config = {"env_prefix": ""}
```

### Step 2: Add to Main Config

In `core/config/__init__.py`:

```python
from .sections.myfeature import MyFeatureConfig

class Config:
    # ...
    myfeature: MyFeatureConfig = Field(default_factory=MyFeatureConfig)
```

### Step 3: Document

Add to README.md under Configuration section with env var examples.

---

## Golden Path: Adding a Cortex Graph

### Step 1: Create Graph Directory

```
cortex/graphs/mygraph/
├── __init__.py
├── graph.py          # StateGraph definition
├── state.py          # TypedDict state
└── steps/            # Business logic
    ├── __init__.py
    └── process.py
```

### Step 2: Define State

```python
# state.py
from typing import TypedDict

class MyGraphState(TypedDict, total=False):
    input: str
    context: list[str]
    output: str
    errors: list[str]
```

### Step 3: Build Graph

```python
# graph.py
from langgraph.graph import StateGraph, END
from .state import MyGraphState
from .steps import process_node

def create_graph():
    workflow = StateGraph(MyGraphState)
    
    workflow.add_node("process", process_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow.compile()
```

### Step 4: Register in Dispatcher

Add to `cortex/graphs/dispatcher.py`:

```python
from .mygraph import create_graph as create_mygraph

GRAPHS = {
    # ...
    "mygraph": create_mygraph,
}
```

---

## Golden Path: Connecting External Graphs

You can connect graphs from **external packages** without storing code inside Router. This is useful for:
- Domain-specific agents in separate repositories (ContextCommerce, ContextWorker)
- Client-specific customizations
- Keeping Router as a thin orchestration layer

### Option 1: Override Path (Config-Based)

Set `CONTEXTROUTER_OVERRIDE_PATH` to point to your graph builder:

```bash
export CONTEXTROUTER_OVERRIDE_PATH="mypackage.graphs:build_custom_graph"
```

Your external package:

```python
# mypackage/graphs.py
from langgraph.graph import StateGraph, END

def build_custom_graph():
    """Build graph from external package."""
    from .state import CustomState
    from .steps import process_step
    
    workflow = StateGraph(CustomState)
    workflow.add_node("process", process_step)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow  # Return uncompiled StateGraph
```

### Option 2: Registry-Based (Programmatic)

Register graph at application startup:

```python
# In your external package's __init__.py or startup script
from contextrouter.core.registry import register_graph

@register_graph("external_commerce")
def build_commerce_graph():
    """Commerce-specific graph from ContextCommerce."""
    from contextcommerce.graphs import create_enrichment_graph
    return create_enrichment_graph()
```

Then use via config:

```bash
export CONTEXTROUTER_GRAPH="external_commerce"
```

### Option 3: Direct Import (For Tight Integration)

When you control both packages:

```python
# In your application
from contextrouter.cortex.graphs import compile_graph
from contextrouter.core.registry import graph_registry

# Register external graph
from contextcommerce.agents.gardener import build_gardener_graph
graph_registry.register("gardener", build_gardener_graph)

# Compile and use
graph = compile_graph("gardener")
result = await graph.ainvoke({"products": [...]})
```

### Priority Resolution

The dispatcher resolves graphs in this order:
1. `router.override_path` — explicit Python path (highest priority)
2. `graph_registry` — registered graphs via `@register_graph`
3. Built-in graphs — `rag_retrieval`, `commerce`

### Best Practices

- **External graphs should be self-contained** — include state, steps, and logic
- **Use ContextUnit protocol** — all data passing through should use ContextUnit
- **Return uncompiled StateGraph** — let Router handle compilation and caching
- **Register at import time** — use `@register_graph` decorator for automatic registration

---

## Golden Path: Adding a Tool

Tools expose functionality to LLMs via function calling.

### Step 1: Create Tool

Create `modules/tools/mytool.py`:

```python
from contextrouter.modules.tools import tool_registry

@tool_registry.register("mytool")
async def mytool(query: str, limit: int = 10) -> dict:
    """
    Search for information.
    
    Args:
        query: Search query
        limit: Max results
        
    Returns:
        Search results
    """
    # Implementation
    return {"results": [...]}
```

### Step 2: Export

Add to `modules/tools/__init__.py`:

```python
from .mytool import mytool
```

---

## ContextUnit Protocol

All data flowing through ContextRouter MUST use the ContextUnit protocol from `contextcore`:

```python
from contextcore import ContextUnit, ContextToken

# Every piece of data has provenance
unit = ContextUnit(
    payload={"query": "..."},
    provenance=["connector:web", "graph:rag"],
    security=SecurityScopes(read=["knowledge:read"])
)

# Authorization via ContextToken
token = ContextToken(permissions=("knowledge:read",))
if token.can_read(unit.security):
    # Access granted
```

### Key Requirements

1. **Provenance** — Always add trace: `unit.provenance.append("stage:name")`
2. **Security** — Use ContextToken for authorization
3. **Metrics** — Track latency/cost in `unit.metrics`

---

## Error/Exception Conventions

- All exceptions inheriting from `ContextrouterError` must have a stable `code` string
- Model errors: Use `ModelQuotaExhaustedError`, `ModelRateLimitError` for proper fallback
- Raise typed errors at boundaries, let transports map `code` to protocols

---

## Architecture Rules

1. **No direct `os.environ`** — Use `Config` classes
2. **Registry-first** — Components registered via decorators, loaded lazily
3. **ContextUnit everywhere** — All cross-boundary data uses ContextUnit
4. **Separation** — Logic (cortex) vs Infrastructure (providers) vs Raw Data (connectors)
