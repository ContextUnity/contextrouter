"""Runner entrypoints.

Runners are thin, host-facing adapters around graphs:
- Provide stable Python APIs (invoke/stream).
- Normalize LangGraph event streams into higher-level progress events where needed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # Chat / RAG retrieval runner
    "create_input_from_query",
    "invoke_agent",
    "invoke_agent_sync",
    "stream_agent",
    # Ingestion runner
    "invoke_ingestion",
    "stream_ingestion",
]


_EXPORTS: dict[str, str] = {
    # Chat
    "create_input_from_query": "contextrouter.cortex.runners.chat.create_input_from_query",
    "invoke_agent": "contextrouter.cortex.runners.chat.invoke_agent",
    "invoke_agent_sync": "contextrouter.cortex.runners.chat.invoke_agent_sync",
    "stream_agent": "contextrouter.cortex.runners.chat.stream_agent",
    # Ingestion
    "invoke_ingestion": "contextrouter.cortex.runners.ingestion.invoke_ingestion",
    "stream_ingestion": "contextrouter.cortex.runners.ingestion.stream_ingestion",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    path = _EXPORTS[name]
    mod_name, attr = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)
