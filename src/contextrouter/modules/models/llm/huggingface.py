"""HuggingFace LLM stub.

Functionality targets:
- transformers library (local inference)
- HuggingFace Inference API (remote)
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from contextrouter.core.config import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseLLM
from ..registry import model_registry


@model_registry.register_llm("hf", "transformers")
class HuggingFaceLLM(BaseLLM):
    def __init__(self, config: Config, **_: Any) -> None:
        self._cfg = config

    async def generate(
        self,
        prompt: str,
        tools: Sequence[Any] | None = None,
        *,
        token: BiscuitToken | None = None,
    ) -> str:
        _ = prompt, tools, token
        raise NotImplementedError("Install 'transformers' to use local HF models.")

    async def stream(self, prompt: str, *, token: BiscuitToken | None = None) -> AsyncIterator[str]:
        _ = prompt, token
        raise NotImplementedError("Install 'transformers' to use local HF models.")

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))

    def as_chat_model(self) -> Any:
        raise NotImplementedError("HuggingFaceLLM does not expose a chat model in this stub.")


__all__ = ["HuggingFaceLLM"]
