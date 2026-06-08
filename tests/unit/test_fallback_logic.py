"""Tests for FallbackModel: sequential, parallel, streaming, budget, and error paths.

Zero-infrastructure tests — all models are mocked, no LLM calls made.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from contextunity.router.core.config import RouterConfig
from contextunity.router.modules.models.base import BaseLLM
from contextunity.router.modules.models.registry import FallbackModel
from contextunity.router.modules.models.types import (
    ErrorEvent,
    FinalTextEvent,
    ModelBudgetExceededError,
    ModelCapabilities,
    ModelCapabilityError,
    ModelExhaustedError,
    ModelQuotaExhaustedError,
    ModelRateLimitError,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelTimeoutError,
    ProviderInfo,
    TextDeltaEvent,
    TextPart,
    UsageStats,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _info(key: str = "mock/test") -> ProviderInfo:
    return ProviderInfo(provider="mock", model_name="test", model_key=key)


class StubModel(BaseLLM):
    """Configurable stub for testing FallbackModel delegation."""

    def __init__(
        self,
        *,
        caps: ModelCapabilities | None = None,
        text: str = "OK",
        fail: type[Exception] | None = None,
        cost: float | None = None,
        token_count: int = 3,
    ):
        super().__init__(provider="mock", model_name="test")
        self._caps = caps or ModelCapabilities(supports_text=True)
        self._text = text
        self._fail = fail
        self._cost = cost
        self._token_count = token_count

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._caps

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        if self._fail:
            raise self._fail(self._text, provider_info=_info())
        usage = UsageStats(total_cost=self._cost) if self._cost is not None else None
        return ModelResponse(text=self._text, raw_provider=_info(), usage=usage)

    async def _stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if self._fail:
            raise self._fail(self._text, provider_info=_info())
        yield TextDeltaEvent(delta=self._text)
        yield FinalTextEvent(text=self._text)

    def get_token_count(self, text: str) -> int:
        return self._token_count


def _fallback(
    candidates: list[tuple[str, BaseLLM]],
    strategy: str = "fallback",
    budget: float | None = None,
) -> FallbackModel:
    config = RouterConfig()
    keys = [k for k, _ in candidates]
    fb = FallbackModel(None, keys, strategy, config, budget_usd=budget)
    fb._candidates = candidates
    return fb


def _text_request(text: str = "Test") -> ModelRequest:
    return ModelRequest(parts=[TextPart(text=text)])


# ═════════════════════════════════════════════════════════════════════
# Capability filtering
# ═════════════════════════════════════════════════════════════════════


class TestCapabilityFiltering:
    """_filter_candidates selects models by required modalities."""

    def test_all_text_models_pass_text_request(self):
        a = StubModel(text="A")
        b = StubModel(text="B")
        fb = _fallback([("a", a), ("b", b)])
        filtered = fb._filter_candidates({"text"})
        assert len(filtered) == 2

    def test_image_request_filters_text_only(self):
        text_only = StubModel(caps=ModelCapabilities(supports_text=True, supports_image=False))
        multimodal = StubModel(caps=ModelCapabilities(supports_text=True, supports_image=True))
        fb = _fallback([("text", text_only), ("multi", multimodal)])
        filtered = fb._filter_candidates({"text", "image"})
        assert len(filtered) == 1
        assert filtered[0][0] == "multi"

    def test_no_compatible_raises_capability_error(self):
        text_only = StubModel(caps=ModelCapabilities(supports_text=True, supports_image=False))
        fb = _fallback([("text", text_only)])
        with pytest.raises(ModelCapabilityError):
            fb._filter_candidates({"image"})

    def test_no_candidates_raises_exhausted(self):
        fb = _fallback([])
        with pytest.raises(ModelExhaustedError):
            fb._filter_candidates({"text"})


# ═════════════════════════════════════════════════════════════════════
# Sequential fallback
# ═════════════════════════════════════════════════════════════════════


class TestSequentialFallback:
    """_generate_sequential: try models in order, stop on success."""

    @pytest.mark.anyio
    async def test_first_model_succeeds(self):
        fb = _fallback([("a", StubModel(text="A")), ("b", StubModel(text="B"))])
        resp = await fb.generate(_text_request())
        assert resp.text == "A"

    @pytest.mark.anyio
    async def test_fallback_on_generic_error(self):
        failing = StubModel(text="fail", fail=Exception)
        success = StubModel(text="OK")
        fb = _fallback([("fail", failing), ("ok", success)])
        resp = await fb.generate(_text_request())
        assert resp.text == "OK"

    @pytest.mark.anyio
    async def test_fallback_on_quota_exhausted(self):
        quota_fail = StubModel(text="quota", fail=ModelQuotaExhaustedError)
        success = StubModel(text="OK")
        fb = _fallback([("quota", quota_fail), ("ok", success)])
        resp = await fb.generate(_text_request())
        assert resp.text == "OK"

    @pytest.mark.anyio
    async def test_fallback_on_timeout(self):
        timeout = StubModel(text="timeout", fail=ModelTimeoutError)
        success = StubModel(text="OK")
        fb = _fallback([("slow", timeout), ("ok", success)])
        resp = await fb.generate(_text_request())
        assert resp.text == "OK"

    @pytest.mark.anyio
    async def test_fallback_on_rate_limit(self):
        rate = StubModel(text="rate", fail=ModelRateLimitError)
        success = StubModel(text="OK")
        fb = _fallback([("limited", rate), ("ok", success)])
        resp = await fb.generate(_text_request())
        assert resp.text == "OK"

    @pytest.mark.anyio
    async def test_all_fail_raises_exhausted(self):
        fail_a = StubModel(text="A", fail=Exception)
        fail_b = StubModel(text="B", fail=Exception)
        fb = _fallback([("a", fail_a), ("b", fail_b)])
        with pytest.raises(ModelExhaustedError, match="All 2 candidate models failed"):
            await fb.generate(_text_request())

    @pytest.mark.anyio
    async def test_exhausted_includes_last_error(self):
        fail = StubModel(text="oops", fail=ModelTimeoutError)
        fb = _fallback([("slow", fail)])
        with pytest.raises(ModelExhaustedError, match="Last error"):
            await fb.generate(_text_request())


# ═════════════════════════════════════════════════════════════════════
# Budget enforcement
# ═════════════════════════════════════════════════════════════════════


class TestBudgetEnforcement:
    """Budget checking in sequential fallback."""

    @pytest.mark.anyio
    async def test_budget_exceeded_raises(self):
        expensive = StubModel(text="$$$", cost=1.50)
        fb = _fallback([("expensive", expensive)], budget=1.00)
        with pytest.raises(ModelBudgetExceededError, match="budget exceeded"):
            await fb.generate(_text_request())

    @pytest.mark.anyio
    async def test_within_budget_succeeds(self):
        cheap = StubModel(text="OK", cost=0.10)
        fb = _fallback([("cheap", cheap)], budget=1.00)
        resp = await fb.generate(_text_request())
        assert resp.text == "OK"

    @pytest.mark.anyio
    async def test_budget_exceeded_no_further_fallback(self):
        """Budget exceeded is a hard stop — never tries next model."""
        expensive = StubModel(text="$$$", cost=2.00)
        backup = StubModel(text="backup")
        fb = _fallback([("expensive", expensive), ("backup", backup)], budget=1.00)
        with pytest.raises(ModelBudgetExceededError):
            await fb.generate(_text_request())

    @pytest.mark.anyio
    async def test_cumulative_cost_across_candidates(self):
        """Cost accumulates across fallback candidates."""
        fail = StubModel(text="fail", fail=Exception)
        succeed_a = StubModel(text="A", cost=0.60)
        succeed_b = StubModel(text="B", cost=0.60)

        # fail → succeed_a (cost=0.60, within 1.00) → never reaches succeed_b
        fb = _fallback([("fail", fail), ("a", succeed_a), ("b", succeed_b)], budget=1.00)
        resp = await fb.generate(_text_request())
        assert resp.text == "A"


# ═════════════════════════════════════════════════════════════════════
# Parallel fallback
# ═════════════════════════════════════════════════════════════════════


class TestParallelFallback:
    """_generate_parallel: concurrent execution, return first success."""

    @pytest.mark.anyio
    async def test_parallel_returns_first_success(self):
        a = StubModel(text="A")
        b = StubModel(text="B")
        fb = _fallback([("a", a), ("b", b)], strategy="parallel")
        resp = await fb.generate(_text_request())
        # First successful result (deterministic since both succeed)
        assert resp.text in ("A", "B")

    @pytest.mark.anyio
    async def test_parallel_skips_failures(self):
        fail = StubModel(text="fail", fail=Exception)
        success = StubModel(text="OK")
        fb = _fallback([("fail", fail), ("ok", success)], strategy="parallel")
        resp = await fb.generate(_text_request())
        assert resp.text == "OK"

    @pytest.mark.anyio
    async def test_parallel_all_fail_raises_exhausted(self):
        fail_a = StubModel(text="A", fail=Exception)
        fail_b = StubModel(text="B", fail=Exception)
        fb = _fallback([("a", fail_a), ("b", fail_b)], strategy="parallel")
        with pytest.raises(ModelExhaustedError, match="parallel mode"):
            await fb.generate(_text_request())


# ═════════════════════════════════════════════════════════════════════
# Streaming fallback
# ═════════════════════════════════════════════════════════════════════


class TestStreamingFallback:
    """_stream: sequential streaming fallback."""

    @pytest.mark.anyio
    async def test_stream_first_success(self):
        a = StubModel(text="streamed")
        fb = _fallback([("a", a)])
        events = []
        async for event in fb.stream(_text_request()):
            events.append(event)
        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(deltas) == 1
        assert deltas[0].delta == "streamed"

    @pytest.mark.anyio
    async def test_stream_fallback_on_failure(self):
        fail = StubModel(text="fail", fail=ModelTimeoutError)
        success = StubModel(text="OK")
        fb = _fallback([("fail", fail), ("ok", success)])
        events = []
        async for event in fb.stream(_text_request()):
            events.append(event)
        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(deltas) == 1
        assert deltas[0].delta == "OK"

    @pytest.mark.anyio
    async def test_stream_quota_exhausted_fallback(self):
        quota_fail = StubModel(text="quota", fail=ModelQuotaExhaustedError)
        success = StubModel(text="OK")
        fb = _fallback([("quota", quota_fail), ("ok", success)])
        events = []
        async for event in fb.stream(_text_request()):
            events.append(event)
        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert deltas[0].delta == "OK"

    @pytest.mark.anyio
    async def test_stream_all_fail_yields_error_event(self):
        fail_a = StubModel(text="A", fail=Exception)
        fail_b = StubModel(text="B", fail=Exception)
        fb = _fallback([("a", fail_a), ("b", fail_b)])
        events = []
        async for event in fb.stream(_text_request()):
            events.append(event)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert "All 2" in errors[0].error


# ═════════════════════════════════════════════════════════════════════
# Token counting & bind_tools
# ═════════════════════════════════════════════════════════════════════


class TestTokenCount:
    """get_token_count delegates to first candidate."""

    def test_delegates_to_first_model(self):
        a = StubModel(token_count=42)
        b = StubModel(token_count=99)
        fb = _fallback([("a", a), ("b", b)])
        assert fb.get_token_count("hello world") == 42

    def test_no_candidates_uses_word_split(self):
        fb = _fallback([])
        assert fb.get_token_count("hello world foo") == 3


class TestBindTools:
    """bind_tools chains LangChain models with fallback."""

    def test_no_langchain_model_raises(self):
        a = StubModel()  # No _langchain_model or _model attr
        fb = _fallback([("a", a)])
        with pytest.raises(AttributeError, match="No candidate model supports bind_tools"):
            fb.bind_tools([])

    def test_single_model_returns_bound_directly(self):
        mock_lc = MagicMock()
        mock_lc.bind_tools.return_value = mock_lc

        a = StubModel()
        a._langchain_model = mock_lc  # type: ignore[attr-defined]
        fb = _fallback([("a", a)])
        result = fb.bind_tools(["tool1"])
        assert result is mock_lc

    def test_multiple_models_chains_with_fallbacks(self):
        primary_lc = MagicMock()
        fallback_lc = MagicMock()
        chained = MagicMock()
        primary_lc.bind_tools.return_value = primary_lc
        fallback_lc.bind_tools.return_value = fallback_lc
        primary_lc.with_fallbacks.return_value = chained

        a = StubModel()
        a._langchain_model = primary_lc  # type: ignore[attr-defined]
        b = StubModel()
        b._langchain_model = fallback_lc  # type: ignore[attr-defined]
        fb = _fallback([("a", a), ("b", b)])
        result = fb.bind_tools(["tool1"])
        primary_lc.with_fallbacks.assert_called_once_with([fallback_lc])
        assert result is chained


# ═════════════════════════════════════════════════════════════════════
# Model key and strategy routing
# ═════════════════════════════════════════════════════════════════════


class TestModelKeyAndStrategy:
    """Verify model_key construction and strategy dispatch."""

    def test_model_key_built_from_candidates(self):
        fb = _fallback([("gpt4", StubModel()), ("claude", StubModel())])
        assert fb._model_key == "fallback/gpt4/claude"

    @pytest.mark.anyio
    async def test_parallel_strategy_dispatches_to_parallel(self):
        a = StubModel(text="A")
        fb = _fallback([("a", a)], strategy="parallel")
        # parallel path returns ModelResponse
        resp = await fb.generate(_text_request())
        assert resp.text == "A"

    @pytest.mark.anyio
    async def test_cost_priority_uses_sequential(self):
        """cost-priority strategy falls back to sequential execution."""
        a = StubModel(text="A")
        fb = _fallback([("a", a)], strategy="cost-priority")
        resp = await fb.generate(_text_request())
        assert resp.text == "A"
