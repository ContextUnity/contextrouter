"""Model registry for LLMs and embeddings."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Generic, Literal, TypeVar, cast

try:
    from asyncio import anext  # Python 3.10+
except ImportError:
    from typing import AsyncIterator as _AI

    async def anext(iterator: _AI[T], default: T | None = None) -> T | None:
        """Polyfill for asyncio.anext on older Python versions."""
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            return default


from contextrouter.core import Config, get_core_config
from contextrouter.core.tokens import ContextToken

from .base import BaseEmbeddings
from .types import (
    BaseModel,
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

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Model selection strategies:
# - fallback: sequentially try candidates in order until one succeeds.
# - parallel: run candidates concurrently and return the first success (generate-only).
# - cost-priority: sequential fallback where you order candidates cheapest → most expensive.
ModelSelectionStrategy = Literal["fallback", "parallel", "cost-priority"]

# Built-in model mappings
BUILTIN_LLMS: dict[str, str] = {
    "vertex/*": "contextrouter.modules.models.llm.vertex.VertexLLM",
    "openai/*": "contextrouter.modules.models.llm.openai.OpenAILLM",
    "openrouter/*": "contextrouter.modules.models.llm.openrouter.OpenRouterLLM",
    "local/*": "contextrouter.modules.models.llm.local_openai.LocalOllamaLLM",
    "local-vllm/*": "contextrouter.modules.models.llm.local_openai.LocalVllmLLM",
    # Anthropic is provider-wildcarded like OpenAI/OpenRouter: any model name becomes `model_name`.
    "anthropic/*": "contextrouter.modules.models.llm.anthropic.AnthropicLLM",
    "groq/*": "contextrouter.modules.models.llm.groq.GroqLLM",
    "runpod/*": "contextrouter.modules.models.llm.runpod.RunPodLLM",
    "hf-hub/*": "contextrouter.modules.models.llm.hf_hub.HuggingFaceHubLLM",
    # HuggingFace transformers: allow `hf/<model_id>`.
    "hf/*": "contextrouter.modules.models.llm.huggingface.HuggingFaceLLM",
    # Perplexity Sonar: built-in search LLM
    "perplexity/*": "contextrouter.modules.models.llm.perplexity.PerplexityLLM",
    # LiteLLM: intentionally a stub (not implemented) to avoid adding another abstraction layer.
    "litellm/*": "contextrouter.modules.models.llm.litellm.LiteLLMStub",
    # Recursive Language Models: wraps any LLM with REPL-based recursive context processing
    # for handling massive contexts (50k+ items). Uses `rlm/<base_model>` format.
    # Example: "rlm/gpt-5-mini" for GPT-5-mini with recursive capabilities.
    "rlm/*": "contextrouter.modules.models.llm.rlm.RLMLLM",
}

BUILTIN_EMBEDDINGS: dict[str, str] = {
    "vertex/text-embedding": ("contextrouter.modules.models.embeddings.vertex.VertexEmbeddings"),
    "hf/sentence-transformers": (
        "contextrouter.modules.models.embeddings.huggingface.HuggingFaceEmbeddings"
    ),
}


# Local Registry class for models
TItem = TypeVar("TItem")


class Registry(Generic[TItem]):
    """Minimal registry for model components."""

    def __init__(self, *, name: str, builtin_map: dict[str, str] | None = None) -> None:
        self._name = name
        self._items: dict[str, TItem] = {}
        self._builtin_map: dict[str, str] = builtin_map or {}

    def get(self, key: str) -> TItem:
        import importlib

        k = key.strip()
        if k not in self._items:
            raw: str | None = None

            # Exact builtin
            if k in self._builtin_map:
                raw = self._builtin_map[k]
            else:
                # Wildcard provider registration: allow `provider/*` to match any `provider/<name>`.
                if "/" in k:
                    provider, _name = k.split("/", 1)
                    wildcard = f"{provider}/*"
                    if wildcard in self._items:
                        return self._items[wildcard]
                    if wildcard in self._builtin_map:
                        raw = self._builtin_map[wildcard]

            if raw is not None:
                # Lazy import
                if ":" in raw:
                    mod_name, attr = raw.split(":", 1)
                elif "." in raw:
                    mod_name, attr = raw.rsplit(".", 1)
                else:
                    mod_name = raw
                    attr = raw
                mod = importlib.import_module(mod_name)
                self._items[k] = cast(TItem, getattr(mod, attr))

        # If exact key is missing, retry wildcard from explicit registrations
        if k not in self._items and "/" in k:
            provider, _name = k.split("/", 1)
            wildcard = f"{provider}/*"
            if wildcard in self._items:
                return self._items[wildcard]

        if k not in self._items:
            raise KeyError(f"{self._name}: unknown key '{k}'")
        return self._items[k]

    def register(self, key: str, value: TItem, *, overwrite: bool = False) -> None:
        k = key.strip()
        if not k:
            raise ValueError(f"{self._name}: registry key must be non-empty")
        if not overwrite and k in self._items:
            raise KeyError(f"{self._name}: '{k}' already registered")
        self._items[k] = value


@dataclass(frozen=True)
class ModelKey:
    provider: str
    name: str

    def as_str(self) -> str:
        return f"{self.provider}/{self.name}"


class ModelRegistry:
    def __init__(self) -> None:
        # Lazy builtin import maps: keep startup fast and avoid importing optional deps.
        self._llms: Registry[type[BaseModel]] = Registry(name="llms", builtin_map=BUILTIN_LLMS)
        self._embeddings: Registry[type[BaseEmbeddings]] = Registry(
            name="embeddings", builtin_map=BUILTIN_EMBEDDINGS
        )

    def register_llm(
        self, provider: str, name: str
    ) -> Callable[[type[BaseModel]], type[BaseModel]]:
        key = ModelKey(provider=provider, name=name).as_str()

        def decorator(cls: type[BaseModel]) -> type[BaseModel]:
            self._llms.register(key, cls)
            return cls

        return decorator

    def register_embeddings(
        self, provider: str, name: str
    ) -> Callable[[type[BaseEmbeddings]], type[BaseEmbeddings]]:
        key = ModelKey(provider=provider, name=name).as_str()

        def decorator(cls: type[BaseEmbeddings]) -> type[BaseEmbeddings]:
            self._embeddings.register(key, cls)
            return cls

        return decorator

    def get_llm(self, key: str | None = None, *, config: Config | None = None) -> BaseModel:
        cfg = config or get_core_config()
        k = key or cfg.models.default_llm
        if "/" not in k:
            raise ValueError(
                "LLM key must be 'provider/name' (e.g. 'vertex/gemini-2.5-flash-lite')"
            )
        cls = self._llms.get(k)
        _provider, name = k.split("/", 1)
        kwargs: dict[str, object] = {"model_name": name}
        ctor = cast(Callable[..., BaseModel], cls)
        return ctor(cfg, **kwargs)

    def create_llm(self, key: str, *, config: Config | None = None, **kwargs: object) -> BaseModel:
        cfg = config or get_core_config()
        k = key
        if "/" not in k:
            raise ValueError(
                "LLM key must be 'provider/name' (e.g. 'vertex/gemini-2.5-flash-lite')"
            )
        cls = self._llms.get(k)
        if "model_name" not in kwargs and "/" in k:
            _provider, name = k.split("/", 1)
            kwargs = dict(kwargs)
            kwargs["model_name"] = name
        ctor = cast(Callable[..., BaseModel], cls)
        return ctor(cfg, **kwargs)

    def get_embeddings(
        self, key: str | None = None, *, config: Config | None = None
    ) -> BaseEmbeddings:
        cfg = config or get_core_config()
        k = key or cfg.models.default_embeddings
        cls = self._embeddings.get(k)
        ctor = cast(Callable[..., BaseEmbeddings], cls)
        return ctor(cfg)

    def create_embeddings(
        self, key: str, *, config: Config | None = None, **kwargs: object
    ) -> BaseEmbeddings:
        """Create embeddings model by explicit key, optionally passing provider kwargs."""
        cfg = config or get_core_config()
        k = key.strip()
        cls = self._embeddings.get(k)
        ctor = cast(Callable[..., BaseEmbeddings], cls)
        return ctor(cfg, **kwargs)

    def get_llm_with_fallback(
        self,
        key: str | None = None,
        *,
        fallback_keys: list[str] | None = None,
        strategy: ModelSelectionStrategy = "fallback",
        config: Config | None = None,
    ) -> BaseModel:
        """Get a model with fallback support.

        Args:
            key: Primary model key (provider/name)
            fallback_keys: List of fallback model keys
            strategy: Fallback strategy ("fallback", "parallel", "cost-priority")
            config: Configuration object

        Returns:
            Model instance with fallback capabilities

        Raises:
            ModelCapabilityError: If no model supports required modalities
        """
        cfg = config or get_core_config()

        # Build candidate list: primary + fallbacks
        primary_key = key or cfg.models.default_llm
        candidate_keys = [primary_key]
        if fallback_keys:
            candidate_keys.extend(fallback_keys)

        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for k in candidate_keys:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)

        logger.debug("Model fallback candidates: %s, strategy: %s", unique_keys, strategy)

        return FallbackModel(
            registry=self,
            candidate_keys=unique_keys,
            strategy=strategy,
            config=cfg,
        )


class FallbackModel(BaseModel):
    """Model wrapper that implements fallback strategies."""

    def __init__(
        self,
        registry: ModelRegistry,
        candidate_keys: list[str],
        strategy: ModelSelectionStrategy,
        config: Config,
    ) -> None:
        self._registry = registry
        self._candidate_keys = candidate_keys
        self._strategy = strategy
        self._config = config
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
            for key in self._candidate_keys:
                try:
                    model = self._registry.create_llm(key, config=self._config)
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
                logger.debug("Trying model %s for generation", key)
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
        from .types import ErrorEvent

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


model_registry = ModelRegistry()

__all__ = ["ModelRegistry", "model_registry", "ModelKey", "FallbackModel", "ModelSelectionStrategy"]
