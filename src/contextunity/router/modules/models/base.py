"""Base interfaces for model providers (multimodal LLM + embeddings).
Hierarchy:
    BaseModel (identity, observability — model-type agnostic)
    ├── BaseLLM(BaseModel)  — text generation contract
    └── BaseEmbeddings(BaseModel) — vector embeddings contract
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from contextunity.core.manifest.router import RetryPolicy
from contextunity.core.parsing import json_loads

from .types import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelResponseFormatError,
    ModelStreamEvent,
    ModelType,
    ProviderInfo,
    ResponseFormat,
    UsageEvent,
)

# ---------------------------------------------------------------------------
# BaseModel — type-agnostic model identity & observability
# ---------------------------------------------------------------------------


class BaseModel(ABC):
    """Type-agnostic model base class.

    Owns identity (``model_key``, ``provider_info``) and observability hooks.
    Does NOT define text-generation methods — those live in ``BaseLLM``.

    Subclasses must set ``model_type`` as a ClassVar.

    Construction contract::

        super().__init__(provider="openai", model_name="gpt-5-mini")

    After ``__init__``:
    - ``self._model_key`` = ``"openai/gpt-5-mini"``
    - ``self._model_name`` = ``"gpt-5-mini"``
    - ``self._provider_info`` = ``ProviderInfo(...)``
    """

    model_type: ClassVar[ModelType]

    def __init__(self, *, provider: str, model_name: str) -> None:
        """Set identity: build ``_model_key`` and ``_provider_info`` from provider/model_name."""
        self._model_name: str = model_name
        self._model_key: str = f"{provider}/{model_name}"
        self._provider_info: ProviderInfo = ProviderInfo(
            provider=provider,
            model_name=model_name,
            model_key=self._model_key,
        )

    @property
    def model_key(self) -> str:
        """Provider-qualified model identifier (e.g. ``"openai/gpt-5.1"``)."""
        return getattr(self, "_model_key", type(self).__name__)

    @property
    def provider_info(self) -> ProviderInfo:
        """Normalized provider information."""
        return self._provider_info

    # ── Retry infrastructure ──────────────────────────────────────────

    _RETRYABLE_TRIGGERS: ClassVar[set[str]] = {"network", "timeout"}
    """Base-level retryable error triggers.  Subclasses extend this set."""

    def _is_retryable(self, error: Exception, policy: RetryPolicy) -> bool:
        """Return ``True`` if *error* maps to a trigger listed in *policy.retry_on*
        and the trigger is in this model class's ``_RETRYABLE_TRIGGERS`` set.
        """
        trigger = self._error_to_trigger(error)
        if trigger is None:
            return False
        return trigger in policy.retry_on and trigger in self._RETRYABLE_TRIGGERS

    @staticmethod
    def _error_to_trigger(error: Exception) -> str | None:
        """Map a runtime exception to a retry trigger key."""
        # Import locally to avoid circular deps — these types live in
        # the same package but may import BaseModel themselves.
        from .types import (
            ModelRateLimitError,
            ModelResponseFormatError,
            ModelTimeoutError,
        )

        if isinstance(error, ModelTimeoutError):
            return "timeout"
        if isinstance(error, ModelRateLimitError):
            return "rate_limit"
        if isinstance(error, ModelResponseFormatError):
            return "response_format"
        if isinstance(error, (ConnectionError, OSError)):
            return "network"
        return None

    @staticmethod
    def _compute_retry_delay(policy: RetryPolicy, attempt: int) -> float:
        """Compute delay in seconds for the given zero-based *attempt*.

        ``"fixed"`` returns ``base_delay_ms`` every time; ``"exponential"``
        doubles per attempt and caps at ``max_delay_ms``.
        """
        if policy.backoff == "none":
            return 0.0
        base = policy.base_delay_ms / 1000.0
        if policy.backoff == "fixed":
            return base
        # exponential: base * 2^attempt, capped at max_delay_ms
        multiplier = 1.0
        for _ in range(attempt):
            multiplier *= 2.0
        max_delay_sec = float(policy.max_delay_ms) / 1000.0
        return min(base * multiplier, max_delay_sec)


# ---------------------------------------------------------------------------
# BaseLLM — text-generation contract
# ---------------------------------------------------------------------------


class BaseLLM(BaseModel, ABC):
    """Multimodal LLM interface.

    Brain events (``llm_start`` / ``llm_end``) are emitted automatically
    by ``generate()`` and ``stream()``.  Cost estimation is applied
    automatically after every generation if ``total_cost`` is absent.

    Retry loop lives in ``generate()`` — controlled by ``RetryPolicy``.
    When all retry attempts are exhausted the last error propagates to
    ``FallbackModel`` which tries the next candidate.

    Providers must implement:
    - ``_generate()`` async method
    - ``_stream()`` async generator

    Optionally override:
    - ``capabilities`` property for multimodal support declaration
    - ``get_token_count()`` if a real tokenizer is available
    - ``generate_batch()`` for optimized batch processing
    """

    model_type: ClassVar[ModelType] = ModelType.LLM

    _default_capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(supports_text=True)
    _RETRYABLE_TRIGGERS: ClassVar[set[str]] = {
        "network",
        "timeout",
        "rate_limit",
        "response_format",
    }

    _logger: ClassVar[logging.Logger] = logging.getLogger("contextunity.router.models")

    @property
    def capabilities(self) -> ModelCapabilities:
        """Declare what modalities this model supports."""
        return getattr(self, "_capabilities", self._default_capabilities)

    @abstractmethod
    async def _generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Provider-specific single-shot generation.

        Called by ``generate()`` inside the retry loop.  Must translate
        ``request.parts`` into the provider's native API call and return a
        populated ``ModelResponse`` with at minimum ``.text`` and ``.usage``.
        """
        raise NotImplementedError

    @abstractmethod
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Provider-specific streaming generation.

        Yields ``TextDelta`` events for incremental text, followed by a
        ``UsageEvent`` when the stream completes.  Called directly by
        ``stream()`` which auto-enriches cost data on the final event.
        """
        raise NotImplementedError

    async def generate(
        self,
        request: ModelRequest,
        *,
        retry_policy: RetryPolicy | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        """Generate a response with automatic retry."""
        import asyncio

        policy = retry_policy or RetryPolicy()
        start = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(policy.max_attempts):
            try:
                response = await self._generate(request, **kwargs)

                # Auto-enrich cost if the provider didn't set it
                if response.usage and response.usage.total_cost is None:
                    _ = response.usage.estimate_cost(self._model_name)

                # Validate response format
                self._validate_response_format(request, response)

                return response

            except Exception as e:
                last_error = e

                # Check if this error qualifies for retry
                if not self._is_retryable(e, policy):
                    raise

                # Check if we've exhausted attempts
                if attempt >= policy.max_attempts - 1:
                    raise

                # Check time budget
                elapsed = time.monotonic() - start
                if elapsed >= policy.timeout_sec:
                    self._logger.warning(
                        "Retry budget exhausted for %s after %.1fs (%d attempts)",
                        self.model_key,
                        elapsed,
                        attempt + 1,
                    )
                    raise

                delay = self._compute_retry_delay(policy, attempt)
                self._logger.info(
                    "Retrying %s (attempt %d/%d) after %s in %.1fs",
                    self.model_key,
                    attempt + 2,
                    policy.max_attempts,
                    type(e).__name__,
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        # Should not reach here, but just in case
        assert last_error is not None  # noqa: S101
        raise last_error

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream tokens from the model."""
        async for event in self._stream(request):
            if isinstance(event, UsageEvent):
                # Auto-enrich cost for stream too
                if event.usage and event.usage.total_cost is None:
                    _ = event.usage.estimate_cost(self._model_name)
            yield event

    async def generate_batch(
        self,
        requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        """Generate responses for multiple requests."""
        import asyncio

        results = await asyncio.gather(
            *[self._generate(req) for req in requests],
            return_exceptions=True,
        )
        responses: list[ModelResponse] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            responses.append(result)
        return responses

    def get_token_count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return max(1, len(text.split()))

    @staticmethod
    def _validate_response_format(request: ModelRequest, response: ModelResponse) -> None:
        """Validate that the response matches the requested format."""
        fmt = request.response_format
        if fmt is None or fmt == ResponseFormat.TEXT or fmt == "text":
            return

        if fmt == ResponseFormat.JSON_OBJECT or fmt == "json_object":
            text = response.text.strip()
            try:
                _ = json_loads(text)
            except ValueError as exc:
                raise ModelResponseFormatError(
                    (
                        f"Model returned invalid JSON for response_format=json_object: "
                        f"{exc!s}. Response text (first 200 chars): {text[:200]!r}"
                    ),
                    provider_info=response.raw_provider,
                ) from exc


# ---------------------------------------------------------------------------
# BaseEmbeddings — vectorization contract
# ---------------------------------------------------------------------------


class BaseEmbeddings(BaseModel, ABC):
    """Vectorization model interface."""

    model_type: ClassVar[ModelType] = ModelType.EMBEDDINGS

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Return a dense embedding vector for a single query string."""
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of document strings."""
        raise NotImplementedError


__all__ = ["BaseModel", "BaseLLM", "BaseEmbeddings"]
