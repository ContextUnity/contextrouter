"""Perplexity LLM provider (Perplexity Sonar API).

Perplexity provides LLM with built-in search capabilities.
Uses Llama models with real-time web search.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

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

# Default model
DEFAULT_MODEL = "sonar"


@model_registry.register_llm("perplexity", "*")
class PerplexityLLM(BaseModel):
    """Perplexity LLM provider with built-in search.

    Supports:
    - Real-time web search (sonar-*-online models)
    - Citations and sources
    - Search recency filtering
    """

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        search_recency_filter: str | None = "day",
        return_citations: bool = True,
        **kwargs: object,
    ) -> None:
        self._cfg = config
        self._model_name = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._search_recency_filter = search_recency_filter
        self._return_citations = return_citations
        self._base_url = "https://api.perplexity.ai"

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
        """Generate response with optional web search."""
        _ = token

        messages = self._build_messages(request)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": request.temperature or 0.7,
        }

        if request.max_output_tokens:
            payload["max_tokens"] = request.max_output_tokens

        # Search-specific options - enable for all models (Perplexity has built-in search)
        if self._search_recency_filter:
            payload["search_recency_filter"] = self._search_recency_filter
        if self._return_citations:
            payload["return_citations"] = self._return_citations

        logger.debug(
            f"Perplexity request: model={self._model_name}, recency={self._search_recency_filter}"
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._cfg.perplexity.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=request.timeout_sec or 60.0,
            )
            response.raise_for_status()
            data = response.json()

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        # Citations available in data.get("citations", []) for online models

        usage = self._extract_usage(data)

        return ModelResponse(
            text=content,
            usage=usage,
            raw_provider=ProviderInfo(
                provider="perplexity",
                model_name=self._model_name,
                model_key=f"perplexity/{self._model_name}",
            ),
        )

    async def stream(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream response tokens."""
        _ = token

        messages = self._build_messages(request)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": request.temperature or 0.7,
            "stream": True,
        }

        if request.max_output_tokens:
            payload["max_tokens"] = request.max_output_tokens

        # Search-specific options - enable for all models
        if self._search_recency_filter:
            payload["search_recency_filter"] = self._search_recency_filter
        if self._return_citations:
            payload["return_citations"] = self._return_citations

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._cfg.perplexity.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=request.timeout_sec or 60.0,
            ) as response:
                response.raise_for_status()
                full = ""

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            full += delta
                            yield TextDeltaEvent(delta=delta)
                    except json.JSONDecodeError:
                        continue

                yield FinalTextEvent(text=full)

    def _build_messages(self, request: ModelRequest) -> list[dict[str, str]]:
        """Convert ModelRequest to Perplexity messages format."""
        messages = []

        if request.system:
            messages.append(
                {
                    "role": "system",
                    "content": request.system,
                }
            )

        # Handle parts - only text supported
        user_content = ""
        for part in request.parts:
            if hasattr(part, "text"):
                user_content += part.text

        if user_content:
            messages.append(
                {
                    "role": "user",
                    "content": user_content,
                }
            )

        return messages

    def _extract_usage(self, data: dict[str, Any]) -> UsageStats | None:
        """Extract usage stats from response."""
        try:
            usage = data.get("usage", {})
            if usage:
                return UsageStats(
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                )
        except Exception:
            pass
        return None


__all__ = ["PerplexityLLM"]
