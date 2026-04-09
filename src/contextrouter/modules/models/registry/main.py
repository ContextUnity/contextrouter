"""Main Model Registry."""

from __future__ import annotations

from typing import Callable, cast

from contextcore import get_context_unit_logger

from contextrouter.core import Config, get_core_config

from ..base import BaseEmbeddings, BaseModel
from .core import ModelKey, Registry
from .fallback import FallbackModel
from .types import BUILTIN_EMBEDDINGS, BUILTIN_LLMS, ModelSelectionStrategy

logger = get_context_unit_logger(__name__)


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

        # intercept and process tenant_id/shield_key_name for api_key retrieval
        tenant_id = kwargs.pop("tenant_id", None)
        if tenant_id and "api_key" not in kwargs:
            shield_key_name = kwargs.pop("shield_key_name", None)

            # Auto-infer shield_key_name from ContextToken capabilities
            # SecureNode capability stripping leaves exactly one allowed path if a node has a model_secret_ref.
            if not shield_key_name:
                try:
                    from contextrouter.cortex.runtime_context import get_current_access_token

                    ctx_token = get_current_access_token()
                    if ctx_token:
                        prefix = f"shield:secrets:read:{tenant_id}/api_keys/"
                        allowed_paths = [
                            p[len(prefix) :] for p in ctx_token.permissions if p.startswith(prefix)
                        ]
                        if len(allowed_paths) == 1:
                            shield_key_name = allowed_paths[0]
                            logger.debug(
                                "Inferred single shield_key_name from token: %s", shield_key_name
                            )
                        elif k in allowed_paths:
                            shield_key_name = k
                except Exception as e:
                    logger.debug("Token inference failed for shield_key_name: %s", e)

            # shield_key_name: per-node path suffix, e.g. "planner/CONTEXTROUTER_PLANNER_MODEL_KEY"
            # fallback: model key itself, e.g. "openai/gpt-5-mini" — matches default/fallback storage
            path_suffix = shield_key_name or k
            shield_path = f"{tenant_id}/api_keys/{path_suffix}"

            # Try Shield first
            try:
                from contextrouter.service.shield_client import shield_get_secret

                secret_val = shield_get_secret(shield_path, tenant_id=tenant_id)
                if secret_val:
                    kwargs["api_key"] = secret_val
            except Exception as e:
                logger.debug("Shield lookup failed for %s: %s", shield_path, e)

            # Fallback: inline secrets (HMAC mode, same key structure)
            if "api_key" not in kwargs:
                try:
                    from contextrouter.service.mixins.registration import _project_secrets

                    if tenant_id in _project_secrets:
                        fallback_key = _project_secrets[tenant_id].get(path_suffix)
                        if fallback_key:
                            kwargs["api_key"] = fallback_key
                            logger.debug(
                                "Using inline API key for %s/%s (no-Shield fallback)",
                                tenant_id,
                                path_suffix,
                            )
                except Exception:
                    pass
        else:
            kwargs.pop("shield_key_name", None)

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
