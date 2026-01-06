"""Vertex AI-backed LLM provider (ported from `contextrouter.cortex.llm`)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Sequence

from langchain_core.messages import SystemMessage

from contextrouter.core.config import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseLLM
from ..registry import model_registry

logger = logging.getLogger(__name__)


@model_registry.register_llm("vertex", "gemini-2.5-flash-lite")
@model_registry.register_llm("vertex", "gemini-2.5-flash")
@model_registry.register_llm("vertex", "gemini-2.5-pro")
class VertexLLM(BaseLLM):
    """Vertex Gemini via langchain-google-genai.

    IMPORTANT: This class MUST NOT access `os.environ` directly. All configuration
    must come from `Config` (which may be layered from env/TOML by the host).
    """

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        streaming: bool = True,
        **_: Any,
    ) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._cfg = config
        self._credentials = None

        # Initialize Google ADC credentials once per instance.
        try:
            import google.auth
            import google.auth.transport.requests

            self._credentials, _project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if hasattr(self._credentials, "refresh"):
                self._credentials.refresh(google.auth.transport.requests.Request())
        except (
            google.auth.exceptions.DefaultCredentialsError,
            google.auth.transport.requests.RequestError,
        ) as e:  # pragma: no cover
            logger.warning("VertexLLM: Failed to initialize credentials: %s", e)
            self._credentials = None
        except Exception as e:  # pragma: no cover
            logger.warning("VertexLLM: Unexpected credential error: %s", e)
            self._credentials = None

        chosen_model = (model_name or "").strip() or "gemini-2.5-flash"

        project_id = config.vertex.project_id
        location = config.vertex.location
        if not project_id or not location:
            # Keep error explicit and early (enterprise-friendly).
            raise ValueError("VertexLLM requires vertex.project_id and vertex.location in Config")

        self._model = ChatGoogleGenerativeAI(
            model=chosen_model,
            project=project_id,
            location=location,
            vertexai=True,
            temperature=config.llm.temperature if temperature is None else temperature,
            max_output_tokens=config.llm.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens,
            streaming=streaming,
            credentials=self._credentials,
            timeout=config.llm.timeout_sec,
            max_retries=config.llm.max_retries,
        )

    async def generate(
        self,
        prompt: str,
        tools: Sequence[Any] | None = None,
        *,
        token: BiscuitToken | None = None,
    ) -> str:
        # Tools/function calling is a future enhancement for this abstraction.
        _ = tools, token
        msg = await self._model.ainvoke([SystemMessage(content=prompt)])
        content = getattr(msg, "content", "")
        return content if isinstance(content, str) else str(content)

    async def stream(self, prompt: str, *, token: BiscuitToken | None = None) -> AsyncIterator[str]:
        _ = token
        async for chunk in self._model.astream([SystemMessage(content=prompt)]):
            c = getattr(chunk, "content", "")
            if isinstance(c, str) and c:
                yield c

    def get_token_count(self, text: str) -> int:
        """Count tokens using the underlying model's tokenizer."""
        if not text:
            return 0
        try:
            return self._model.get_num_tokens(text)
        except (AttributeError, TypeError, ValueError):
            # Fallback to a rough estimate (approx 4 chars per token)
            return max(1, len(text) // 4)

    def as_chat_model(self) -> Any:
        return self._model


__all__ = ["VertexLLM"]
