"""Fallback strategies -- automatic provider failover on rate-limit, timeout, or model unavailability."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, override

from contextunity.core import get_contextunit_logger
from contextunity.core.manifest.router import RetryPolicy

# Using native anext directly
from contextunity.router.core import RouterConfig

from ..base import BaseLLM
from ..langchain_boundary import LangchainToolBinder, langchain_tool_binder
from ..types import (
    ErrorEvent,
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
)
from .types import ModelSelectionStrategy

if TYPE_CHECKING:
    from .main import ModelRegistry

logger = get_contextunit_logger(__name__)


class FallbackModel(BaseLLM):
    """Model wrapper that implements fallback strategies."""

    def __init__(
        self,
        registry: ModelRegistry,
        candidate_keys: list[str],
        strategy: ModelSelectionStrategy,
        config: RouterConfig,
        budget_usd: float | None = None,
        **kwargs: object,
    ) -> None:
        """Store candidate keys, selection strategy, and optional per-node cost budget."""
        resolved_name = "/".join(candidate_keys)
        super().__init__(provider="fallback", model_name=resolved_name)
        self._registry: ModelRegistry = registry
        self._candidate_keys: list[str] = candidate_keys
        self._strategy: ModelSelectionStrategy = strategy
        self._config: RouterConfig = config
        self._budget_usd: float | None = budget_usd
        self._kwargs: dict[str, object] = dict(kwargs)
        self._candidates: list[tuple[str, BaseLLM]] | None = None

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Capabilities are determined by filtering candidates."""
        # This will be checked during generation
        return ModelCapabilities()

    def _get_candidates(self) -> list[tuple[str, BaseLLM]]:
        """Lazy initialization of candidate models."""
        if self._candidates is None:
            self._candidates = []

            tenant_id = self._kwargs.get("tenant_id")
            if not tenant_id:
                try:
                    from contextunity.router.core.context import get_current_access_token

                    ctx_token = get_current_access_token()
                    if ctx_token and getattr(ctx_token, "allowed_tenants", ()):
                        tenant_id = ctx_token.allowed_tenants[0]
                except Exception:
                    pass

            for key in self._candidate_keys:
                try:
                    model_kwargs = dict(self._kwargs)

                    if tenant_id:
                        model_kwargs["tenant_id"] = tenant_id

                    model = self._registry.create_llm(key, config=self._config, **model_kwargs)
                    self._candidates.append((key, model))
                except Exception as e:
                    logger.warning("Failed to initialize model %s: %s", key, e)
                    continue
        return self._candidates

    def _filter_candidates(self, required_modalities: set[str]) -> list[tuple[str, BaseLLM]]:
        """Filter candidates that support all required modalities."""
        candidates = self._get_candidates()
        if not candidates:
            raise ModelExhaustedError(
                (
                    "No candidate models could be initialized. "
                    "Check optional dependencies for the selected provider(s) and your config."
                ),
                provider_info=None,
            )
        filtered: list[tuple[str, BaseLLM]] = []

        for key, model in candidates:
            caps = model.capabilities
            if caps.supports(required_modalities):
                filtered.append((key, model))

        if not filtered:
            available = [(key, model.capabilities) for key, model in candidates]
            raise ModelCapabilityError(
                (
                    f"No model supports required modalities {required_modalities}. "
                    f"Available: {available}"
                ),
                provider_info=None,
            )

        return filtered

    @override
    async def generate(
        self,
        request: ModelRequest,
        *,
        retry_policy: RetryPolicy | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        """Bypass wrapper-level brain events — delegates directly to ``_generate``."""
        _ = retry_policy, kwargs
        return await self._generate(request)

    @override
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Yield stream events via ``super().stream()`` which enriches usage cost."""
        async for event in super().stream(request):
            yield event

    @override
    async def _generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Route to sequential or parallel strategy depending on ``_strategy``."""
        required = request.required_modalities()
        candidates = self._filter_candidates(required)

        if self._strategy == "parallel":
            return await self._generate_parallel(candidates, request)
        else:  # "fallback" or "cost-priority"
            return await self._generate_sequential(candidates, request)

    async def _generate_sequential(
        self,
        candidates: list[tuple[str, BaseLLM]],
        request: ModelRequest,
    ) -> ModelResponse:
        """Try each candidate in order; skip on rate-limit/timeout/quota exhaustion.

        Raises:
            ModelBudgetExceededError: If cumulative cost exceeds ``_budget_usd``.
            ModelExhaustedError: If every candidate fails.
        """
        last_error = None
        cumulative_cost: float = 0.0

        for key, model in candidates:
            try:
                if key == candidates[0][0]:
                    logger.info("Executing Primary Model: %s", key)
                else:
                    logger.warning("Executing Fallback Model: %s", key)

                response = await model.generate(request)

                # Track cumulative cost across all candidates
                if response.usage and response.usage.total_cost is not None:
                    cumulative_cost += response.usage.total_cost

                # Check per-node cost budget (hard stop — no further fallback)
                if self._budget_usd is not None and cumulative_cost > self._budget_usd:
                    raise ModelBudgetExceededError(
                        (
                            f"Node cost budget exceeded: ${cumulative_cost:.4f} > "
                            f"${self._budget_usd:.4f} across {candidates.index((key, model)) + 1} "
                            f"candidate(s)"
                        ),
                        provider_info=response.raw_provider,
                    )

                if response.usage:
                    logger.debug(
                        "Generation succeeded with model %s, usage: %s", key, response.usage
                    )
                else:
                    logger.debug("Generation succeeded with model %s", key)
                return response
            except ModelBudgetExceededError:
                raise  # Hard stop — never fallback on budget exceeded
            except ModelQuotaExhaustedError as e:
                # Quota exhausted = billing issue, fallback immediately (no retries help)
                logger.warning("Model %s quota exhausted, trying fallback: %s", key, e)
                last_error = e
                continue
            except (ModelTimeoutError, ModelRateLimitError) as e:
                logger.warning("Model %s failed (%s): %s", key, type(e).__name__, e)
                last_error = e
                continue
            except Exception as e:
                logger.error("Model %s failed with unexpected error: %s", key, e)
                last_error = e
                continue

        # All candidates failed
        error_msg = f"All {len(candidates)} candidate models failed"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise ModelExhaustedError(error_msg, provider_info=None)

    async def _generate_parallel(
        self,
        candidates: list[tuple[str, BaseLLM]],
        request: ModelRequest,
    ) -> ModelResponse:
        """Fire all candidates concurrently and return the first successful response.

        Raises:
            ModelExhaustedError: If every candidate fails.
        """

        async def try_model(_key: str, model: BaseLLM) -> ModelResponse:
            """Attempt generation on a single candidate; exceptions propagate to ``gather``."""
            return await model.generate(request)

        tasks = [try_model(key, model) for key, model in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Find first successful result
        for i, result in enumerate(results):
            key, _ = candidates[i]
            if isinstance(result, ModelResponse):
                logger.debug("Parallel generation succeeded with model %s", key)
                return result
            else:
                logger.debug("Model %s failed in parallel mode: %s", key, result)

        # All failed
        raise ModelExhaustedError(
            f"All {len(candidates)} models failed in parallel mode",
            provider_info=None,
        )

    @override
    async def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Sequential-only stream fallback: try models in order, commit to the first that yields."""
        required = request.required_modalities()
        candidates = self._filter_candidates(required)

        # For streaming, only sequential fallback makes sense
        # Try models in order, commit to first that yields content
        last_error = None

        for key, model in candidates:
            try:
                logger.debug("Trying model %s for streaming", key)
                event_iterator = model.stream(request)
                first_event = await anext(event_iterator, None)

                if first_event is not None:
                    # Model started streaming successfully
                    logger.debug("Streaming succeeded with model %s", key)
                    yield first_event

                    # Continue yielding from this successful model
                    async for event in event_iterator:
                        yield event
                    return  # Success, don't try fallbacks
                else:
                    # Model completed without yielding anything - try next
                    continue

            except ModelQuotaExhaustedError as e:
                logger.warning(
                    "Model %s quota exhausted during streaming, trying fallback: %s", key, e
                )
                last_error = e
                continue
            except (ModelTimeoutError, ModelRateLimitError) as e:
                logger.warning(
                    "Model %s failed during streaming (%s): %s", key, type(e).__name__, e
                )
                last_error = e
                continue
            except Exception as e:
                logger.error("Model %s failed streaming with unexpected error: %s", key, e)
                last_error = e
                continue

        # All candidates failed
        error_msg = f"All {len(candidates)} candidate models failed for streaming"
        if last_error:
            error_msg += f". Last error: {last_error}"
        yield ErrorEvent(error=error_msg)

    @override
    def get_token_count(self, text: str) -> int:
        """Use first available model for token counting."""
        candidates = self._get_candidates()
        if not candidates:
            return len(text.split())  # Fallback estimate

        _, first_model = candidates[0]
        return first_model.get_token_count(text)

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> object:
        """Bind tools to all candidate langchain models with fallback chain."""
        bound_models: list[tuple[str, LangchainToolBinder]] = []
        for key, model in self._get_candidates():
            langchain_attr: object = getattr(model, "_langchain_model", None)
            model_attr: object = getattr(model, "_model", None)
            lc_raw = langchain_attr if langchain_attr is not None else model_attr
            lc_model = langchain_tool_binder(lc_raw) if lc_raw is not None else None
            if lc_model is not None:
                try:
                    bound_raw = lc_model.bind_tools(tools, **kwargs)
                    bound = langchain_tool_binder(bound_raw)
                    if bound is not None:
                        bound_models.append((key, bound))
                except Exception as e:
                    logger.warning("Failed to bind tools for %s: %s", key, e)
                    continue

        if not bound_models:
            raise AttributeError(
                (
                    "No candidate model supports bind_tools. "
                    "Ensure at least one LLM provider with LangChain integration is configured."
                )
            )

        primary_key, primary = bound_models[0]
        if len(bound_models) == 1:
            logger.debug("FallbackModel.bind_tools using single model: %s", primary_key)
            return primary

        fallbacks: list[LangchainToolBinder] = [m for _, m in bound_models[1:]]
        logger.debug(
            "FallbackModel.bind_tools: primary=%s, fallbacks=%s",
            primary_key,
            [k for k, _ in bound_models[1:]],
        )
        return primary.with_fallbacks(fallbacks)
