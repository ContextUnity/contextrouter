"""contextrouter - shared LangGraph brain.

This package is the single "agent brain" used by:
- Web chat (AG-UI streaming via the API service)
- Telegram bot (webhook via the API service)

It contains the LangGraph graph and nodes, tool wrappers, and shared
brain-side utilities.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

__all__ = [
    "__version__",
    # Main entry points
    "stream_agent",
    "invoke_agent",
    # Telemetry
    "get_langfuse_callbacks",
    "trace_context",
    "langfuse_flush",
]

try:
    __version__ = importlib.metadata.version("contextrouter")
except Exception:  # noqa: BLE001
    # Fallback for editable/dev environments where metadata may be unavailable.
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    """Lazy exports to keep `import contextrouter` lightweight.

    This is important for CLI usage (`python -m contextrouter.cli`) where we want
    `--help` to work without importing the entire brain and its optional deps.
    """
    if name in {"invoke_agent", "stream_agent"}:
        mod = importlib.import_module("contextrouter.cortex")
        return getattr(mod, name)
    if name in {"get_langfuse_callbacks", "trace_context"}:
        mod = importlib.import_module("contextrouter.modules.observability")
        return getattr(mod, name)
    if name == "langfuse_flush":
        mod = importlib.import_module("contextrouter.modules.observability")
        return getattr(mod, "flush")
    raise AttributeError(name)
