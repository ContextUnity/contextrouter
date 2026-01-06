"""LiteLLM-backed LLM provider (optional dependency).

This provider is intentionally opt-in:
- It is instantiated only when the model key starts with `litellm/`.
- It imports `langchain_litellm` lazily (only when actually used).

Key format:
    litellm/<provider>/<model>
Examples:
    litellm/openai/gpt-4o-mini
    litellm/anthropic/claude-3-5-haiku-latest
    litellm/vertex_ai/gemini-2.0-flash
"""

from __future__ import annotations

import logging
from inspect import signature
from typing import Any, AsyncIterator, Sequence

from langchain_core.messages import SystemMessage

from contextrouter.core.config import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseLLM

logger = logging.getLogger(__name__)


class LiteLLMLLM(BaseLLM):
    def __init__(
        self,
        config: Config,
        *,
        model: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        streaming: bool = True,
        **_: Any,
    ) -> None:
        self._cfg = config
        self._model_name = (model or "").strip()
        self._streaming = streaming
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._chat_model: Any | None = None

        if not self._model_name:
            raise ValueError("LiteLLMLLM requires a non-empty model string after 'litellm/'.")

    def _get_chat_model(self) -> Any:
        if self._chat_model is not None:
            return self._chat_model

        try:
            from langchain_litellm import ChatLiteLLM  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "LiteLLMLLM requires optional dependencies: install contextrouter with "
                "`litellm` extras (e.g. `pip install 'contextrouter[litellm]'`)."
            ) from e

        cfg = self._cfg
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "streaming": self._streaming,
            "temperature": cfg.llm.temperature if self._temperature is None else self._temperature,
            # Litellm / LangChain param naming varies by version; we filter by signature.
            "max_tokens": cfg.llm.max_output_tokens
            if self._max_output_tokens is None
            else self._max_output_tokens,
            "api_base": getattr(getattr(cfg, "litellm", None), "api_base", None),
            # Optional fallback list (if supported by the installed version).
            "fallbacks": getattr(getattr(cfg, "litellm", None), "fallback_models", None),
            "timeout": getattr(getattr(cfg, "litellm", None), "timeout_sec", None)
            or cfg.llm.timeout_sec,
        }

        # Only pass supported kwargs to avoid tight coupling to library versions.
        try:
            params = set(signature(ChatLiteLLM.__init__).parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
            self._chat_model = ChatLiteLLM(**filtered)
            return self._chat_model
        except TypeError:
            # Conservative fallback: model + streaming + temperature are the most stable knobs.
            minimal = {
                k: v
                for k, v in kwargs.items()
                if k in {"model", "streaming", "temperature"} and v is not None
            }
            self._chat_model = ChatLiteLLM(**minimal)
            return self._chat_model

    async def generate(
        self,
        prompt: str,
        tools: Sequence[Any] | None = None,
        *,
        token: BiscuitToken | None = None,
    ) -> str:
        _ = tools, token
        llm = self._get_chat_model()
        msg = await llm.ainvoke([SystemMessage(content=prompt)])
        content = getattr(msg, "content", "")
        return content if isinstance(content, str) else str(content)

    async def stream(self, prompt: str, *, token: BiscuitToken | None = None) -> AsyncIterator[str]:
        _ = token
        llm = self._get_chat_model()
        async for chunk in llm.astream([SystemMessage(content=prompt)]):
            c = getattr(chunk, "content", "")
            if isinstance(c, str) and c:
                yield c

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))

    def as_chat_model(self) -> Any:
        return self._get_chat_model()


__all__ = ["LiteLLMLLM"]
