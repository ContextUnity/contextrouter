"""Registry core -- provider instantiation, caching, and lookup by model identifier.

Provides the generic ``Registry`` and ``ModelKey`` classes used to instantiate
and retrieve models and embedding providers. Supports lazy importing from string paths
and wildcard mappings (e.g., matching ``openai/*`` dynamically).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeGuard, TypeVar

from contextunity.router.core.exceptions import RouterRegistryError

TProvider = TypeVar("TProvider")


def _is_provider_subclass(value: type, base: type[TProvider]) -> TypeGuard[type[TProvider]]:
    return issubclass(value, base)


class Registry(Generic[TProvider]):
    """Generic registry for lazy-loaded provider classes.

    Maintains a cache of registered items and a map of built-in paths for dynamic,
    on-demand importing. Supports wildcard registration lookup.
    """

    def __init__(
        self,
        *,
        name: str,
        builtin_map: dict[str, str] | None = None,
        item_base: type[TProvider] | None = None,
    ) -> None:
        """Initialize the Registry instance.

        Args:
            name: Readable identifier of the registry (used in error messages).
            builtin_map: Optional dictionary mapping keys to importable Python paths
                (e.g., {"openai/*": "contextunity.router.modules.models.llm.openai:OpenAI"}).
            item_base: Optional base class used to validate lazily imported providers.
        """
        self._name: str = name
        self._items: dict[str, type[TProvider]] = {}
        self._builtin_map: dict[str, str] = builtin_map or {}
        self._item_base: type[TProvider] | None = item_base

    def get(self, key: str) -> type[TProvider]:
        """Retrieve a registered item by its key, resolving and loading it lazily if needed.

        Looks up exact matches first, then falls back to wildcard matching (e.g.,
        matching "openai/gpt-4o" with "openai/*") inside both the cache and the
        built-in import map.

        Args:
            key: The lookup string key.

        Returns:
            The provider class registered for the key.

        Raises:
            KeyError: If the key is unknown and cannot be resolved or lazy-loaded.
        """
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
                module_globals: dict[str, object] = dict(vars(mod))
                if attr not in module_globals:
                    raise RouterRegistryError(
                        f"{self._name}: builtin '{raw}' missing attribute {attr!r}"
                    )
                loaded_obj: object = module_globals[attr]
                if not isinstance(loaded_obj, type):
                    raise RouterRegistryError(
                        f"{self._name}: builtin '{raw}' did not resolve to a class"
                    )
                loaded = loaded_obj
                if self._item_base is None:
                    raise RouterRegistryError(
                        f"{self._name}: lazy import for '{raw}' requires item_base"
                    )
                if not _is_provider_subclass(loaded, self._item_base):
                    base_name = getattr(self._item_base, "__name__", repr(self._item_base))
                    raise RouterRegistryError(
                        f"{self._name}: builtin '{raw}' is not a subclass of {base_name}"
                    )
                self._items[k] = loaded

        # If exact key is missing, retry wildcard from explicit registrations
        if k not in self._items and "/" in k:
            provider, _name = k.split("/", 1)
            wildcard = f"{provider}/*"
            if wildcard in self._items:
                return self._items[wildcard]

        if k not in self._items:
            raise KeyError(f"{self._name}: unknown key '{k}'")
        return self._items[k]

    def register(self, key: str, value: type[TProvider], *, overwrite: bool = False) -> None:
        """Explicitly register an item in the registry.

        Args:
            key: Unique lookup key.
            value: The provider class to register.
            overwrite: Whether to allow replacing an existing registered item.

        Raises:
            ValueError: If the key is empty.
            KeyError: If the key is already registered and overwrite is False.
        """
        k = key.strip()
        if not k:
            raise RouterRegistryError(f"{self._name}: registry key must be non-empty")
        if not overwrite and k in self._items:
            raise KeyError(f"{self._name}: '{k}' already registered")
        self._items[k] = value


@dataclass(frozen=True)
class ModelKey:
    """Parsed model provider and name identifiers.

    Enables safe structured passing of parsed provider keys throughout the LLM router.
    """

    provider: str
    name: str

    def as_str(self) -> str:
        """Format the key as a standard provider/name string.

        Returns:
            The combined string representation (e.g. "openai/gpt-4o").
        """
        return f"{self.provider}/{self.name}"
