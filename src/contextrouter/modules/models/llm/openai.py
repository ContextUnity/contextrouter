"""OpenAI/Anthropic LLM stub.

Functionality targets:
- Wrapper for langchain-openai or direct SDK
- Must support function calling (tools) to be compatible with Router-style agents
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from contextrouter.core.config import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseLLM
from ..registry import model_registry


@model_registry.register_llm("openai", "gpt")
class OpenAILLM(BaseLLM):
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
        raise NotImplementedError(
            "OpenAILLM stub: install 'langchain-openai' or an OpenAI/Anthropic SDK and implement this provider."
        )

    async def stream(self, prompt: str, *, token: BiscuitToken | None = None) -> AsyncIterator[str]:
        _ = prompt, token
        raise NotImplementedError(
            "OpenAILLM stub: install 'langchain-openai' or an OpenAI/Anthropic SDK and implement this provider."
        )

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))

    def as_chat_model(self) -> Any:
        raise NotImplementedError("OpenAILLM does not expose a chat model in this stub.")


__all__ = ["OpenAILLM"]
