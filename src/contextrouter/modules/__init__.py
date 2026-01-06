"""Pluggable capabilities for contextrouter.

- connectors/: sources
- transformers/: pure-ish logic pipes
- providers/: sinks (storage/virtual/direct)
- protocols/: external communication contracts (AG-UI, A2UI, A2A, ...)
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "connectors",
    "providers",
    "transformers",
    "models",
    "retrieval",
    "observability",
]


def __getattr__(name: str) -> Any:
    """Lazy submodule exports.

    Keep `import contextrouter.modules` lightweight and avoid import-time side
    effects / optional dependency crashes.
    """
    if name in {"connectors", "providers", "transformers", "models", "retrieval", "observability"}:
        return importlib.import_module(f"contextrouter.modules.{name}")
    raise AttributeError(name)
