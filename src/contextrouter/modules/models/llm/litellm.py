"""LiteLLM provider (stub).

Why this is not implemented (yet):

- We already have explicit providers (Vertex, OpenAI, Anthropic, OpenRouter,
  local OpenAI-compatible). They are simpler to debug and keep behavior explicit.
- LiteLLM adds another abstraction layer ("combiner inside combiner") which
  can complicate: streaming semantics, multimodal mapping, error taxonomy.
- Cost/usage/observability can be handled by Langfuse + our `UsageStats`.

If you want LiteLLM, implement `litellm/*` using LiteLLM's OpenAI-compatible API:

```python
from litellm import completion

def generate(prompt: str) -> str:
    response = completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```
"""

from __future__ import annotations

from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseModel
from ..registry import model_registry
from ..types import ModelCapabilities, ModelRequest, ModelResponse, ModelStreamEvent


@model_registry.register_llm("litellm", "*")
class LiteLLMStub(BaseModel):
    """Not implemented on purpose."""

    def __init__(self, config: Config, *, model_name: str | None = None, **_: object) -> None:
        _ = config, model_name
        self._cap = ModelCapabilities(
            supports_text=True, supports_image=False, supports_audio=False
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._cap

    async def generate(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> ModelResponse:
        _ = request, token
        raise NotImplementedError(
            "LiteLLM provider is intentionally not implemented. "
            "Rationale: avoid stacking abstraction layers; use explicit providers. "
            "If you want LiteLLM, implement `litellm/*` as an optional provider."
        )

    async def stream(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request, token
        raise NotImplementedError(
            "LiteLLM provider is intentionally not implemented. "
            "If you want it, implement streaming mapping to ModelStreamEvent."
        )

    def get_token_count(self, text: str) -> int:
        # Conservative heuristic; actual counting depends on backend.
        return max(1, len(text) // 4) if text else 0


__all__ = ["LiteLLMStub"]
