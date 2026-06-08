"""Pluggable capabilities for contextunity.router.

- connectors/: sources
- transformers/: pure-ish logic pipes
- providers/: sinks (storage/virtual/direct)
- protocols/: external communication contracts (AG-UI, A2UI, A2A, ...)
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import connectors, models, observability, providers, retrieval, transformers

__all__ = [
    "connectors",
    "providers",
    "transformers",
    "models",
    "retrieval",
    "observability",
]


def __getattr__(name: str) -> ModuleType:
    """Lazy submodule exports.

    Keep `import contextunity.router.modules` lightweight and avoid import-time side
    effects / optional dependency crashes.
    """
    if name in {"connectors", "providers", "transformers", "models", "retrieval", "observability"}:
        return importlib.import_module(f"contextunity.router.modules.{name}")
    raise AttributeError(name)
