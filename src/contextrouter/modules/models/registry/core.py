"""Minimal registry core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

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
    """Represents a provider/name combination for a model."""

    provider: str
    name: str

    def as_str(self) -> str:
        return f"{self.provider}/{self.name}"
