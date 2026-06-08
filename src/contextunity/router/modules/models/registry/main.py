"""Main Model Registry -- singleton that loads provider config and exposes the unified model resolution API."""

from __future__ import annotations

from typing import Callable, Protocol, TypeGuard

from contextunity.core import get_contextunit_logger

from contextunity.router.core import RouterConfig, get_core_config
from contextunity.router.core.exceptions import RouterRegistryError

from ..base import BaseEmbeddings, BaseLLM
from .core import ModelKey, Registry
from .fallback import FallbackModel
from .types import BUILTIN_EMBEDDINGS, BUILTIN_LLMS, ModelSelectionStrategy

logger = get_contextunit_logger(__name__)


class _LLMProviderCtor(Protocol):
    def __call__(self, config: RouterConfig, /, **kwargs: object) -> BaseLLM: ...


class _EmbeddingsProviderCtor(Protocol):
    def __call__(self, config: RouterConfig, /, **kwargs: object) -> BaseEmbeddings: ...


def _is_llm_provider_ctor(cls: type[BaseLLM]) -> TypeGuard[_LLMProviderCtor]:
    return callable(cls)


def _is_embeddings_provider_ctor(cls: type[BaseEmbeddings]) -> TypeGuard[_EmbeddingsProviderCtor]:
    return callable(cls)


class ModelRegistry:
    """Registry for AI models."""

    def __init__(self) -> None:
        """Create empty LLM and embeddings registries with lazy builtin import maps."""
        # Lazy builtin import maps: keep startup fast and avoid importing optional deps.
        self._llms: Registry[BaseLLM] = Registry(
            name="llms", builtin_map=BUILTIN_LLMS, item_base=BaseLLM
        )
        self._embeddings: Registry[BaseEmbeddings] = Registry(
            name="embeddings",
            builtin_map=BUILTIN_EMBEDDINGS,
            item_base=BaseEmbeddings,
        )

    def register_llm(self, provider: str, name: str) -> Callable[[type[BaseLLM]], type[BaseLLM]]:
        """Decorator: register an LLM class under ``provider/name`` in the LLM registry."""
        key = ModelKey(provider=provider, name=name).as_str()

        def decorator(cls: type[BaseLLM]) -> type[BaseLLM]:
            """Register *cls* under the composed key and return it unchanged."""
            self._llms.register(key, cls)
            return cls

        return decorator

    def register_embeddings(
        self, provider: str, name: str
    ) -> Callable[[type[BaseEmbeddings]], type[BaseEmbeddings]]:
        """Decorator: register an embeddings class under ``provider/name`` in the embeddings registry."""
        key = ModelKey(provider=provider, name=name).as_str()

        def decorator(cls: type[BaseEmbeddings]) -> type[BaseEmbeddings]:
            """Register *cls* under the composed key and return it unchanged."""
            self._embeddings.register(key, cls)
            return cls

        return decorator

    def get_llm(self, key: str | None = None, *, config: RouterConfig | None = None) -> BaseLLM:
        """Resolve and instantiate an LLM by ``provider/model`` key.

        Falls back to ``config.models.default_llm`` when *key* is ``None``.

        Raises:
            ValueError: If the key is not in ``provider/name`` format.
        """
        cfg = config or get_core_config()
        k = key or cfg.models.default_llm
        if "/" not in k:
            raise RouterRegistryError(
                "LLM key must be 'provider/name' (e.g. 'vertex/gemini-2.5-flash-lite')"
            )
        cls = self._llms.get(k)
        _provider, name = k.split("/", 1)
        kwargs: dict[str, object] = {"model_name": name}
        if not _is_llm_provider_ctor(cls):
            raise RouterRegistryError(f"LLM registry entry '{k}' is not callable")
        return cls(cfg, **kwargs)

    def create_llm(
        self, key: str, *, config: RouterConfig | None = None, **kwargs: object
    ) -> BaseLLM:
        """Instantiate an LLM by explicit key, resolving API credentials via Shield or inline secrets.

        If ``tenant_id`` is passed in *kwargs*, the method attempts Shield-based
        secret lookup first. Inline Redis keys are looked up by ``project_id``
        when supplied, falling back to ``tenant_id`` for legacy registrations.

        Raises:
            ValueError: If the key is not in ``provider/name`` format.
        """
        cfg = config or get_core_config()
        k = key
        if "/" not in k:
            raise RouterRegistryError(
                "LLM key must be 'provider/name' (e.g. 'vertex/gemini-2.5-flash-lite')"
            )
        cls = self._llms.get(k)

        # intercept and process tenant_id/shield_key_name for api_key retrieval
        _tid = kwargs.pop("tenant_id", None)
        tenant_id: str | None = str(_tid) if _tid else None
        _pid = kwargs.pop("project_id", None)
        project_id: str | None = str(_pid) if _pid else None

        if tenant_id and "api_key" not in kwargs:
            _skn = kwargs.pop("shield_key_name", None)
            shield_key_name: str | None = str(_skn) if _skn else None

            # Auto-infer shield_key_name from ContextToken capabilities
            # SecureNode capability stripping leaves exactly one allowed path if a node has a model_secret_ref.
            if not shield_key_name:
                try:
                    from contextunity.router.core.context import get_current_access_token

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

            # shield_key_name: per-node path suffix, e.g. "planner/CU_ROUTER_PLANNER_MODEL_KEY"
            # fallback: model key itself, e.g. "openai/gpt-5-mini" — matches default/fallback storage
            path_suffix = shield_key_name or k
            shield_path = f"{tenant_id}/api_keys/{path_suffix}"

            # Shield-based secret lookup (fail-closed when enabled)
            shield_url = getattr(cfg, "router", None) and cfg.shield_url
            if shield_url:
                from contextunity.router.service.shield_client import shield_get_secret

                secret_val = shield_get_secret(shield_path, tenant_id=tenant_id)
                if secret_val:
                    kwargs["api_key"] = secret_val

            # Fallback 2: inline secrets (Redis scale-safe HMAC mode)
            if "api_key" not in kwargs:
                try:
                    from contextunity.core.discovery import get_project_key

                    secret_owner_id = project_id or tenant_id
                    key_data = get_project_key(secret_owner_id)
                    if key_data and "api_keys" in key_data:
                        fallback_key = key_data["api_keys"].get(path_suffix)
                        if fallback_key:
                            kwargs["api_key"] = fallback_key
                            logger.debug(
                                "Using inline API key for %s/%s (Redis fallback)",
                                secret_owner_id,
                                path_suffix,
                            )
                except Exception as e:
                    logger.debug("Failed inline key lookup for %s: %s", project_id or tenant_id, e)

        else:
            _ = kwargs.pop("shield_key_name", None)

        if "model_name" not in kwargs and "/" in k:
            _provider, name = k.split("/", 1)
            kwargs = dict(kwargs)
            kwargs["model_name"] = name
        if not _is_llm_provider_ctor(cls):
            raise RouterRegistryError(f"LLM registry entry '{k}' is not callable")
        return cls(cfg, **kwargs)

    def get_embeddings(
        self, key: str | None = None, *, config: RouterConfig | None = None
    ) -> BaseEmbeddings:
        """Resolve and instantiate an embeddings model.

        Falls back to ``config.models.default_embeddings`` when *key* is ``None``.
        """
        cfg = config or get_core_config()
        k = key or cfg.models.default_embeddings
        cls = self._embeddings.get(k)
        if not _is_embeddings_provider_ctor(cls):
            raise RouterRegistryError(f"Embeddings registry entry '{k}' is not callable")
        return cls(cfg)

    def create_embeddings(
        self, key: str, *, config: RouterConfig | None = None, **kwargs: object
    ) -> BaseEmbeddings:
        """Instantiate an embeddings model by explicit key with optional provider kwargs."""
        cfg = config or get_core_config()
        k = key.strip()
        cls = self._embeddings.get(k)
        if not _is_embeddings_provider_ctor(cls):
            raise RouterRegistryError(f"Embeddings registry entry '{k}' is not callable")
        return cls(cfg, **kwargs)

    def get_llm_with_fallback(
        self,
        key: str | None = None,
        *,
        fallback_keys: list[str] | None = None,
        strategy: ModelSelectionStrategy = "fallback",
        config: RouterConfig | None = None,
        budget_usd: float | None = None,
    ) -> BaseLLM:
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
        seen: set[str] = set()
        unique_keys: list[str] = []
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
            budget_usd=budget_usd,
        )


model_registry = ModelRegistry()
