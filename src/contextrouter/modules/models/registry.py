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
from contextrouter.core.tokens import BiscuitToken

from .base import BaseEmbeddings
from .types import (
    BaseModel,
    ModelCapabilities,
    ModelCapabilityError,
    ModelExhaustedError,
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
    # LiteLLM: intentionally a stub (not implemented) to avoid adding another abstraction layer.
    "litellm/*": "contextrouter.modules.models.llm.litellm.LiteLLMStub",
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

        logger.debug(f"Model fallback candidates: {unique_keys}, strategy: {strategy}")

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
                    logger.warning(f"Failed to initialize model {key}: {e}")
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
        token: BiscuitToken | None = None,
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
        token: BiscuitToken | None,
    ) -> ModelResponse:
        """Sequential fallback: try models in order until success."""
        last_error = None

        for key, model in candidates:
            try:
                logger.debug(f"Trying model {key} for generation")
                response = await model.generate(request, token=token)
                if response.usage:
                    logger.info(f"Generation succeeded with model {key}, usage: {response.usage}")
                else:
                    logger.info(f"Generation succeeded with model {key}")
                return response
            except (ModelTimeoutError, ModelRateLimitError) as e:
                logger.warning(f"Model {key} failed ({type(e).__name__}): {e}")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"Model {key} failed with unexpected error: {e}")
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
        token: BiscuitToken | None,
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
                logger.info(f"Parallel generation succeeded with model {key}")
                return result
            else:
                logger.debug(f"Model {key} failed in parallel mode: {result}")

        # All failed
        raise ModelExhaustedError(
            f"All {len(candidates)} models failed in parallel mode",
            provider_info=None,
        )

    async def stream(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        required = request.required_modalities()
        candidates = self._filter_candidates(required)

        # For streaming, only sequential fallback makes sense
        # Try models in order, commit to first that yields content
        last_error = None

        for key, model in candidates:
            try:
                logger.debug(f"Trying model {key} for streaming")
                event_iterator = model.stream(request, token=token)
                first_event = await anext(event_iterator, None)

                if first_event is not None:
                    # Model started streaming successfully
                    logger.info(f"Streaming succeeded with model {key}")
                    yield first_event

                    # Continue yielding from this successful model
                    async for event in event_iterator:
                        yield event
                    return  # Success, don't try fallbacks
                else:
                    # Model completed without yielding anything - try next
                    continue

            except (ModelTimeoutError, ModelRateLimitError) as e:
                logger.warning(f"Model {key} failed during streaming ({type(e).__name__}): {e}")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"Model {key} failed streaming with unexpected error: {e}")
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


model_registry = ModelRegistry()

__all__ = ["ModelRegistry", "model_registry", "ModelKey", "FallbackModel", "ModelSelectionStrategy"]
