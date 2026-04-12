"""Fallback strategies for models."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

from contextunity.core import get_contextunit_logger

try:
    from asyncio import anext  # Python 3.10+
except ImportError:
    from typing import AsyncIterator as _AI
    from typing import TypeVar

    T = TypeVar("T")

    async def anext(iterator: _AI[T], default: T | None = None) -> T | None:
        """Polyfill for asyncio.anext on older Python versions."""
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            return default


from contextunity.core.tokens import ContextToken

from contextunity.router.core import Config

from ..types import (
    BaseModel,
    ErrorEvent,
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


class FallbackModel(BaseModel):
    """Model wrapper that implements fallback strategies."""

    def __init__(
        self,
        registry: ModelRegistry,
        candidate_keys: list[str],
        strategy: ModelSelectionStrategy,
        config: Config,
        **kwargs: object,
    ) -> None:
        self._registry = registry
        self._candidate_keys = candidate_keys
        self._strategy = strategy
        self._config = config
        self._kwargs = kwargs
        self._candidates: list[tuple[str, BaseModel]] | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Capabilities are determined by filtering candidates."""
        # This will be checked during generation
        return ModelCapabilities()

    def _get_candidates(self) -> list[tuple[str, BaseModel]]:
        """Lazy initialization of candidate models."""
        if self._candidates is None:
            self._candidates = []

            tenant_id = self._kwargs.get("tenant_id")
            if not tenant_id:
                try:
                    from contextunity.router.cortex.runtime_context import get_current_access_token

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

    def _filter_candidates(self, required_modalities: set[str]) -> list[tuple[str, BaseModel]]:
        """Filter candidates that support all required modalities."""
        candidates = self._get_candidates()
        if not candidates:
            raise ModelExhaustedError(
                "No candidate models could be initialized. "
                "Check optional dependencies for the selected provider(s) and your config.",
                provider_info=None,
            )
        filtered = []

        for key, model in candidates:
            caps = model.capabilities
            if caps.supports(required_modalities):
                filtered.append((key, model))

        if not filtered:
            available = [(key, model.capabilities) for key, model in candidates]
            raise ModelCapabilityError(
                f"No model supports required modalities {required_modalities}. "
                f"Available: {available}",
                provider_info=None,
            )

        return filtered

    async def generate(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
    ) -> ModelResponse:
        required = request.required_modalities()
        candidates = self._filter_candidates(required)

        if self._strategy == "parallel":
            return await self._generate_parallel(candidates, request, token)
        else:  # "fallback" or "cost-priority"
            return await self._generate_sequential(candidates, request, token)

    async def _generate_sequential(
        self,
        candidates: list[tuple[str, BaseModel]],
        request: ModelRequest,
        token: ContextToken | None,
    ) -> ModelResponse:
        """Sequential fallback: try models in order until success."""
        last_error = None

        for key, model in candidates:
            try:
                if key == candidates[0][0]:
                    logger.info("Executing Primary Model: %s", key)
                else:
                    logger.warning("Executing Fallback Model: %s", key)

                response = await model.generate(request, token=token)
                if response.usage:
                    logger.debug(
                        "Generation succeeded with model %s, usage: %s", key, response.usage
                    )
                else:
                    logger.debug("Generation succeeded with model %s", key)
                return response
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
        candidates: list[tuple[str, BaseModel]],
        request: ModelRequest,
        token: ContextToken | None,
    ) -> ModelResponse:
        """Parallel fallback: try all models concurrently, return first success."""

        async def try_model(key: str, model: BaseModel) -> ModelResponse:
            try:
                return await model.generate(request, token=token)
            except Exception:
                raise  # Will be caught by gather

        tasks = [try_model(key, model) for key, model in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Find first successful result
        for i, result in enumerate(results):
            key, _ = candidates[i]
            if not isinstance(result, Exception):
                logger.debug("Parallel generation succeeded with model %s", key)
                return result
            else:
                logger.debug("Model %s failed in parallel mode: %s", key, result)

        # All failed
        raise ModelExhaustedError(
            f"All {len(candidates)} models failed in parallel mode",
            provider_info=None,
        )

    async def stream(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        required = request.required_modalities()
        candidates = self._filter_candidates(required)

        # For streaming, only sequential fallback makes sense
        # Try models in order, commit to first that yields content
        last_error = None

        for key, model in candidates:
            try:
                logger.debug("Trying model %s for streaming", key)
                event_iterator = model.stream(request, token=token)
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

    def get_token_count(self, text: str) -> int:
        """Use first available model for token counting."""
        candidates = self._get_candidates()
        if not candidates:
            return len(text.split())  # Fallback estimate

        _, first_model = candidates[0]
        return first_model.get_token_count(text)

    def bind_tools(self, tools, **kwargs):
        """Bind tools to all candidate LangChain models with fallback chain.

        Finds the underlying LangChain model from each candidate,
        calls bind_tools on each, then chains them with with_fallbacks()
        so ainvoke retries with the next model on failure.
        """
        bound_models = []
        for key, model in self._get_candidates():
            lc_model = getattr(model, "_langchain_model", None) or getattr(model, "_model", None)
            if lc_model is not None and hasattr(lc_model, "bind_tools"):
                try:
                    bound = lc_model.bind_tools(tools, **kwargs)
                    bound_models.append((key, bound))
                except Exception as e:
                    logger.warning("Failed to bind tools for %s: %s", key, e)
                    continue

        if not bound_models:
            raise AttributeError(
                "No candidate model supports bind_tools. "
                "Ensure at least one LLM provider with LangChain integration is configured."
            )

        primary_key, primary = bound_models[0]
        if len(bound_models) == 1:
            logger.debug("FallbackModel.bind_tools using single model: %s", primary_key)
            return primary

        # Chain with LangChain's built-in fallback mechanism
        fallbacks = [m for _, m in bound_models[1:]]
        logger.debug(
            "FallbackModel.bind_tools: primary=%s, fallbacks=%s",
            primary_key,
            [k for k, _ in bound_models[1:]],
        )
        return primary.with_fallbacks(fallbacks)
