"""Main Model Registry."""

from __future__ import annotations

import logging
from typing import Callable, cast

from contextrouter.core import Config, get_core_config

from ..base import BaseEmbeddings, BaseModel
from .core import ModelKey, Registry
from .fallback import FallbackModel
from .types import BUILTIN_EMBEDDINGS, BUILTIN_LLMS, ModelSelectionStrategy

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for AI models."""

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
        **kwargs: object,
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
        primary_key = key or cfg.models.default_llm

        # Build candidate list: primary + fallbacks
        candidate_keys = [primary_key]
        if fallback_keys:
            candidate_keys.extend(fallback_keys)
        elif cfg.models.allow_global_fallback and cfg.models.fallback_llms:
            candidate_keys.extend(cfg.models.fallback_llms)

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
            **kwargs,
        )


model_registry = ModelRegistry()
