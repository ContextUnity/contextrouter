"""Model registry for LLMs and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

from contextrouter.core import Config, get_core_config

from .base import BaseEmbeddings, BaseLLM

T = TypeVar("T")

# Built-in model mappings
BUILTIN_LLMS: dict[str, str] = {
    "vertex/gemini-2.5-flash-lite": "contextrouter.modules.models.llm.vertex.VertexLLM",
    "vertex/gemini-2.5-flash": "contextrouter.modules.models.llm.vertex.VertexLLM",
    "vertex/gemini-2.5-pro": "contextrouter.modules.models.llm.vertex.VertexLLM",
    "openai/gpt": "contextrouter.modules.models.llm.openai.OpenAILLM",
    "hf/transformers": "contextrouter.modules.models.llm.huggingface.HuggingFaceLLM",
}

BUILTIN_EMBEDDINGS: dict[str, str] = {
    "vertex/text-embedding": "contextrouter.modules.models.embeddings.vertex.VertexEmbeddings",
    "hf/sentence-transformers": "contextrouter.modules.models.embeddings.huggingface.HuggingFaceEmbeddings",
}


# Local Registry class for models
class Registry:
    """Minimal registry for model components."""

    def __init__(self, *, name: str, builtin_map: dict[str, str] | None = None) -> None:
        self._name = name
        self._items: dict[str, Any] = {}
        self._builtin_map: dict[str, str] = builtin_map or {}

    def get(self, key: str) -> Any:
        import importlib

        k = key.strip()
        if k not in self._items and k in self._builtin_map:
            # Lazy import
            raw = self._builtin_map[k]
            if ":" in raw:
                mod_name, attr = raw.split(":", 1)
            elif "." in raw:
                mod_name, attr = raw.rsplit(".", 1)
            else:
                mod_name = raw
                attr = raw
            mod = importlib.import_module(mod_name)
            self._items[k] = getattr(mod, attr)
        if k not in self._items:
            raise KeyError(f"{self._name}: unknown key '{k}'")
        return self._items[k]

    def register(self, key: str, value: Any, *, overwrite: bool = False) -> None:
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
        self._llms: Registry[type[BaseLLM]] = Registry(name="llms", builtin_map=BUILTIN_LLMS)
        self._embeddings: Registry[type[BaseEmbeddings]] = Registry(
            name="embeddings", builtin_map=BUILTIN_EMBEDDINGS
        )

    def _create_litellm(self, key: str, cfg: Config, **kwargs: Any) -> BaseLLM:
        # Special-case: LiteLLM is a routing layer, so we allow arbitrary model strings:
        # `litellm/<provider>/<model>` (e.g. litellm/openai/gpt-4o-mini).
        #
        # This stays deterministic and opt-in (only activates for keys starting with `litellm/`).
        from contextrouter.modules.models.llm.litellm import LiteLLMLLM

        # Strip only the leading "litellm/" prefix; do not parse provider/model further.
        model = key[len("litellm/") :].strip() if key.startswith("litellm/") else ""
        return LiteLLMLLM(cfg, model=model, **kwargs)

    def register_llm(self, provider: str, name: str) -> Callable[[type[BaseLLM]], type[BaseLLM]]:
        key = ModelKey(provider=provider, name=name).as_str()

        def decorator(cls: type[BaseLLM]) -> type[BaseLLM]:
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

    def get_llm(self, key: str | None = None, *, config: Config | None = None) -> BaseLLM:
        cfg = config or get_core_config()
        k = key or cfg.models.default_llm
        if k.strip().lower().startswith("litellm/"):
            return self._create_litellm(k, cfg)
        if "/" not in k:
            raise ValueError(
                "LLM model key must be explicit 'provider/name' (e.g. 'vertex/gemini-2.5-flash-lite')"
            )
        cls = self._llms.get(k)
        _provider, name = k.split("/", 1)
        kwargs: dict[str, Any] = {"model_name": name}
        ctor = cast(Callable[..., BaseLLM], cls)
        return ctor(cfg, **kwargs)

    def create_llm(self, key: str, *, config: Config | None = None, **kwargs: Any) -> BaseLLM:
        cfg = config or get_core_config()
        k = key
        if k.strip().lower().startswith("litellm/"):
            return self._create_litellm(k, cfg, **kwargs)
        if "/" not in k:
            raise ValueError(
                "LLM model key must be explicit 'provider/name' (e.g. 'vertex/gemini-2.5-flash-lite')"
            )
        cls = self._llms.get(k)
        if "model_name" not in kwargs and "/" in k:
            _provider, name = k.split("/", 1)
            kwargs = dict(kwargs)
            kwargs["model_name"] = name
        ctor = cast(Callable[..., BaseLLM], cls)
        return ctor(cfg, **kwargs)

    def get_embeddings(
        self, key: str | None = None, *, config: Config | None = None
    ) -> BaseEmbeddings:
        cfg = config or get_core_config()
        k = key or cfg.models.default_embeddings
        cls = self._embeddings.get(k)
        ctor = cast(Callable[..., BaseEmbeddings], cls)
        return ctor(cfg)


model_registry = ModelRegistry()

__all__ = ["ModelRegistry", "model_registry", "ModelKey"]
