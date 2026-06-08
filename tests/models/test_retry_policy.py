"""Tests for RetryPolicy manifest validation and BaseLLM retry loop.

Architecture contract:
  - RetryPolicy is a Pydantic model in the manifest with sensible defaults
  - BaseLLM.generate() retries retryable errors up to max_attempts
  - Non-retryable errors (ModelQuotaExhaustedError) propagate immediately
  - Budget exhaustion stops retries even if max_attempts not reached
  - BaseModel only retries network/timeout; BaseLLM adds rate_limit/response_format
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from contextunity.core.manifest.router import RetryPolicy

from contextunity.router.modules.models.base import BaseLLM
from contextunity.router.modules.models.types import (
    ModelBudgetExceededError,
    ModelQuotaExhaustedError,
    ModelRateLimitError,
    ModelRequest,
    ModelResponse,
    ModelResponseFormatError,
    ModelStreamEvent,
    ModelTimeoutError,
    ProviderInfo,
    TextPart,
    UsageStats,
)

# ── Mock model ──────────────────────────────────────────────────────


class MockLLM(BaseLLM):
    """Minimal LLM for testing retry behaviour."""

    def __init__(self) -> None:
        super().__init__(provider="test", model_name="mock-v1")
        self._generate_fn: AsyncMock = AsyncMock()

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        return await self._generate_fn(request)

    async def _stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield  # type: ignore[misc]  # pragma: no cover


def _ok_response() -> ModelResponse:
    return ModelResponse(
        text="hello",
        usage=UsageStats(input_tokens=10, output_tokens=5),
        raw_provider=ProviderInfo(provider="test", model_name="mock-v1", model_key="test/mock-v1"),
    )


def _make_request(fmt: str | None = None) -> ModelRequest:
    parts = [TextPart(text="hi")]
    return ModelRequest(parts=parts, response_format=fmt)


# ── Manifest validation tests ───────────────────────────────────────


class TestRetryPolicyManifest:
    def test_defaults(self):
        """Default RetryPolicy has sensible values."""
        p = RetryPolicy()
        assert p.max_attempts == 2
        assert p.backoff == "exponential"
        assert p.base_delay_ms == 500
        assert p.max_delay_ms == 8000
        assert p.timeout_sec == 30.0
        assert "rate_limit" in p.retry_on
        assert "timeout" in p.retry_on
        assert "network" in p.retry_on
        assert "response_format" not in p.retry_on


# ── Backoff calculation tests ────────────────────────────────────────


class TestBackoffCalculation:
    def test_no_backoff(self):
        p = RetryPolicy(backoff="none")
        assert BaseLLM._compute_retry_delay(p, 0) == 0.0
        assert BaseLLM._compute_retry_delay(p, 5) == 0.0

    def test_fixed_backoff(self):
        p = RetryPolicy(backoff="fixed", base_delay_ms=1000)
        assert BaseLLM._compute_retry_delay(p, 0) == 1.0
        assert BaseLLM._compute_retry_delay(p, 3) == 1.0

    def test_exponential_backoff(self):
        p = RetryPolicy(backoff="exponential", base_delay_ms=500, max_delay_ms=8000)
        assert BaseLLM._compute_retry_delay(p, 0) == 0.5  # 500ms * 2^0
        assert BaseLLM._compute_retry_delay(p, 1) == 1.0  # 500ms * 2^1
        assert BaseLLM._compute_retry_delay(p, 2) == 2.0  # 500ms * 2^2
        assert BaseLLM._compute_retry_delay(p, 5) == 8.0  # capped at max_delay_ms


# ── Retry loop tests ────────────────────────────────────────────────


class TestRetryLoop:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Successful generation returns immediately."""
        model = MockLLM()
        model._generate_fn.return_value = _ok_response()

        result = await model.generate(_make_request())
        assert result.text == "hello"
        assert model._generate_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """ModelRateLimitError triggers retry."""
        model = MockLLM()
        model._generate_fn.side_effect = [
            ModelRateLimitError("429 too many", provider_info=None),
            _ok_response(),
        ]
        policy = RetryPolicy(max_attempts=3, backoff="none", retry_on=["rate_limit"])

        result = await model.generate(_make_request(), retry_policy=policy)
        assert result.text == "hello"
        assert model._generate_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_response_format_error(self):
        """ModelResponseFormatError triggers retry when configured."""
        model = MockLLM()
        model._generate_fn.side_effect = [
            ModelResponseFormatError("bad json", provider_info=None),
            _ok_response(),
        ]
        policy = RetryPolicy(
            max_attempts=3,
            backoff="none",
            retry_on=["rate_limit", "response_format"],
        )

        result = await model.generate(_make_request(), retry_policy=policy)
        assert result.text == "hello"
        assert model._generate_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_response_format_not_retried_by_default(self):
        """ModelResponseFormatError propagates when not in retry_on."""
        model = MockLLM()
        model._generate_fn.side_effect = ModelResponseFormatError("bad json", provider_info=None)
        # Default policy does NOT include response_format in retry_on
        policy = RetryPolicy(max_attempts=3, backoff="none")

        with pytest.raises(ModelResponseFormatError):
            await model.generate(_make_request(), retry_policy=policy)
        assert model._generate_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self):
        """ModelQuotaExhaustedError never retries."""
        model = MockLLM()
        model._generate_fn.side_effect = ModelQuotaExhaustedError("billing", provider_info=None)
        policy = RetryPolicy(max_attempts=5, backoff="none")

        with pytest.raises(ModelQuotaExhaustedError):
            await model.generate(_make_request(), retry_policy=policy)
        assert model._generate_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_attempts_raise(self):
        """All attempts exhausted → last error propagates."""
        model = MockLLM()
        model._generate_fn.side_effect = ModelTimeoutError("timeout", provider_info=None)
        policy = RetryPolicy(max_attempts=3, backoff="none", retry_on=["timeout"])

        with pytest.raises(ModelTimeoutError):
            await model.generate(_make_request(), retry_policy=policy)
        assert model._generate_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_budget_exhaustion(self):
        """Retry stops when timeout_sec is exceeded."""
        model = MockLLM()
        model._generate_fn.side_effect = ModelTimeoutError("timeout", provider_info=None)
        policy = RetryPolicy(
            max_attempts=10,
            backoff="none",
            retry_on=["timeout"],
            timeout_sec=1.0,
        )

        # Fake time so budget is exceeded after first attempt
        with patch("time.monotonic", side_effect=[0.0, 0.0, 2.0]):
            with pytest.raises(ModelTimeoutError):
                await model.generate(_make_request(), retry_policy=policy)
        # Should stop after 2 attempts (first attempt + one retry before budget check)
        assert model._generate_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_with_max_attempts_one(self):
        """max_attempts=1 means no retry (single attempt only)."""
        model = MockLLM()
        model._generate_fn.side_effect = ModelRateLimitError("429", provider_info=None)
        policy = RetryPolicy(max_attempts=1, retry_on=["rate_limit"])

        with pytest.raises(ModelRateLimitError):
            await model.generate(_make_request(), retry_policy=policy)
        assert model._generate_fn.call_count == 1


# ── Error-to-trigger mapping tests ──────────────────────────────────


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("error", "expected_trigger"),
        [
            (ModelTimeoutError("t", provider_info=None), "timeout"),
            (ModelRateLimitError("r", provider_info=None), "rate_limit"),
            (ModelResponseFormatError("j", provider_info=None), "response_format"),
            (ConnectionError("conn refused"), "network"),
            (OSError("socket"), "network"),
            (ModelQuotaExhaustedError("q", provider_info=None), None),
            (RuntimeError("boom"), None),
        ],
    )
    def test_error_maps_to_trigger(self, error, expected_trigger):
        assert BaseLLM._error_to_trigger(error) == expected_trigger


# ── Cost budget tests (FallbackModel level) ───────────────────────────


def _ok_response_with_cost(cost: float, key: str = "test/mock-v1") -> ModelResponse:
    return ModelResponse(
        text="ok",
        usage=UsageStats(input_tokens=100, output_tokens=50, total_cost=cost),
        raw_provider=ProviderInfo(provider="test", model_name="mock-v1", model_key=key),
    )


class CostMockLLM(BaseLLM):
    """Mock LLM that returns a fixed cost."""

    def __init__(self, cost: float, key: str = "test/mock-v1") -> None:
        provider, model = key.split("/")
        super().__init__(provider=provider, model_name=model)
        self._cost = cost

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        return _ok_response_with_cost(self._cost, self.model_key)

    async def _stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield  # type: ignore[misc]  # pragma: no cover


class TestCostBudget:
    @pytest.mark.asyncio
    async def test_budget_exceeded_on_primary(self):
        """Primary model response exceeds budget → ModelBudgetExceededError."""
        from contextunity.router.core.config import RouterConfig
        from contextunity.router.modules.models.registry import FallbackModel

        config = RouterConfig()
        model = CostMockLLM(cost=0.10, key="openai/gpt-5")
        fallback = FallbackModel(
            None,
            ["openai/gpt-5"],
            "fallback",
            config,
            budget_usd=0.05,
        )
        fallback._candidates = [("openai/gpt-5", model)]

        with pytest.raises(ModelBudgetExceededError, match="0.1000"):
            await fallback.generate(_make_request())

    @pytest.mark.asyncio
    async def test_no_budget_limit(self):
        """When budget_usd=None, expensive responses succeed."""
        from contextunity.router.core.config import RouterConfig
        from contextunity.router.modules.models.registry import FallbackModel

        config = RouterConfig()
        model = CostMockLLM(cost=999.99, key="openai/gpt-5")
        fallback = FallbackModel(
            None,
            ["openai/gpt-5"],
            "fallback",
            config,  # budget_usd=None
        )
        fallback._candidates = [("openai/gpt-5", model)]

        result = await fallback.generate(_make_request())
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_within_budget_succeeds(self):
        """Response under budget returns normally."""
        from contextunity.router.core.config import RouterConfig
        from contextunity.router.modules.models.registry import FallbackModel

        config = RouterConfig()
        model = CostMockLLM(cost=0.001, key="openai/gpt-5")
        fallback = FallbackModel(
            None,
            ["openai/gpt-5"],
            "fallback",
            config,
            budget_usd=1.0,
        )
        fallback._candidates = [("openai/gpt-5", model)]

        result = await fallback.generate(_make_request())
        assert result.text == "ok"
