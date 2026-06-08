"""contextunity.router - Distributed AI routing layer.

This package provides the central LangGraph-based workflow execution engine.
It manages dynamic configuration, state routing, and streaming execution decoupled
from any specific transport mechanism or user interface.
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextunity.router.cortex import (
        invoke_agent,
        invoke_dispatcher,
        stream_agent,
        stream_dispatcher,
    )
    from contextunity.router.cortex.services.dispatcher import get_dispatcher_service
    from contextunity.router.modules.observability import (
        flush as langfuse_flush,
    )
    from contextunity.router.modules.observability import (
        get_langfuse_callbacks,
        trace_context,
    )

__all__ = [
    "__version__",
    # Main entry points
    "stream_agent",
    "invoke_agent",
    # Dispatcher agent
    "invoke_dispatcher",
    "stream_dispatcher",
    "get_dispatcher_service",
    # Telemetry
    "get_langfuse_callbacks",
    "trace_context",
    "langfuse_flush",
]

try:
    __version__ = importlib.metadata.version("contextunity.router")
except Exception:  # noqa: BLE001
    # Fallback for editable/dev environments where metadata may be unavailable.
    __version__ = "0.0.0"


def __getattr__(name: str) -> object:
    """Lazy exports to keep `import contextunity.router` lightweight.

    This is important for CLI usage (`python -m contextunity.router.cli`) where we want
    `--help` to work without importing the entire brain and its optional deps.
    """
    if name == "invoke_agent":
        from contextunity.router.cortex import invoke_agent

        return invoke_agent
    if name == "stream_agent":
        from contextunity.router.cortex import stream_agent

        return stream_agent
    if name == "invoke_dispatcher":
        from contextunity.router.cortex import invoke_dispatcher

        return invoke_dispatcher
    if name == "stream_dispatcher":
        from contextunity.router.cortex import stream_dispatcher

        return stream_dispatcher
    if name == "get_dispatcher_service":
        from contextunity.router.cortex.services.dispatcher import get_dispatcher_service

        return get_dispatcher_service
    if name == "get_langfuse_callbacks":
        from contextunity.router.modules.observability import get_langfuse_callbacks

        return get_langfuse_callbacks
    if name == "trace_context":
        from contextunity.router.modules.observability import trace_context

        return trace_context
    if name == "langfuse_flush":
        from contextunity.router.modules.observability import flush

        return flush
    raise AttributeError(name)
