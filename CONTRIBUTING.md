# Contributing

Thanks for contributing to **ContextRouter**.

## Development setup

```bash
cd contextrouter
uv pip install -e '.[dev]'
```

If you work on ingestion code/CLI:

```bash
uv pip install -e '.[dev,ingestion]'
```

## Pre-commit

Install the git hooks once:

```bash
pre-commit install
```

Run on-demand:

```bash
pre-commit run --all-files
```

## Linting & tests

```bash
uv run ruff check . --fix
uv run ruff format .
uv run python -m pytest -q
```

## Branching & GitHub workflow

### Branch naming

- **Features**: `feat/<short-topic>`
- **Fixes**: `fix/<short-topic>`
- **Chores**: `chore/<short-topic>`
- **Docs**: `contextrouter_docs/<short-topic>`
- **Refactors**: `refactor/<short-topic>`

Keep names lowercase, dash-separated. Example: `feat/ingestion-runner`.

### PR flow (recommended)

- Branch off `main`
- Open a PR early (Draft is fine)
- Keep PRs small and focused; avoid mixing unrelated changes
- Before requesting review:
  - `pre-commit run --all-files`
  - ensure CI is green (lint + tests)

### Merge strategy

- Prefer **Squash & merge** into `main` (keeps history clean)
- Use **Conventional Commits** style in the squash commit title:
  - `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `chore: ...`, `test: ...`

### Releases

- Bump version in `pyproject.toml` (SemVer)
- Tag releases as `vX.Y.Z`

## Error/exception conventions

- All internal exceptions that cross module/transport boundaries should inherit from
  `contextrouter.core.exceptions.ContextrouterError`.
- Every `ContextrouterError` must have a stable, **non-empty** `code` string.
- Prefer raising typed errors close to the boundary (providers/connectors/modules), and let
  transports map `code` to their own protocol.

## Architecture constraints (summary)

- `contextrouter.core` is the kernel: keep it knowledge-agnostic.
- RAG-specific types/settings live under `modules/retrieval/rag/`.
- Ingestion-specific code lives under `modules/ingestion/`.
- Graph wiring lives in `cortex/graphs/`; business logic in `cortex/steps/`.

## Golden Path: Adding a New LLM Provider

Follow these steps to add a new LLM provider (e.g., `newprovider`):

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
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
    UsageStats,
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
        # Implementation here
        ...

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

### Step 3: Add Configuration (if needed)

If the provider needs API keys or config, add to `core/config/`:

1. Add settings class in `core/config/sections/`:
   ```python
   class NewProviderConfig(BaseSettings):
       api_key: str = Field(default="", env="NEWPROVIDER_API_KEY")
   ```

2. Add to main Config class in `core/config/__init__.py`

### Step 4: Add Tests

Create `tests/unit/test_newprovider_llm.py`:

```python
import pytest
from contextrouter.modules.models import model_registry

def test_newprovider_registration():
    """Ensure newprovider is registered in BUILTIN_LLMS."""
    model = model_registry.create_llm("newprovider/default-model", config=config)
    assert model is not None
```

### Step 5: Document

Update `docs/router/src/content/docs/models/` with provider documentation.

### Common Pitfalls

1. **Forgot BUILTIN_LLMS** → `KeyError: "llms: unknown key 'newprovider/model'"`
2. **Temperature not supported** → Check if model supports custom temperature (like gpt-5-mini)
3. **Missing optional deps** → Add to `pyproject.toml` under `[project.optional-dependencies]`

