"""Brain services for runtime knowledge access.

Import dispatcher symbols from ``contextunity.router.cortex.services.dispatcher``
directly — they are intentionally not re-exported here to keep this package
free of gRPC/service import cycles when loading ``redis_saver`` and ``graph``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import GraphService, get_graph_service


def __getattr__(name: str) -> object:
    """Lazily expose graph services without loading Brain for dispatcher imports."""
    if name in {"GraphService", "get_graph_service"}:
        from .graph import GraphService, get_graph_service

        return GraphService if name == "GraphService" else get_graph_service
    raise AttributeError(name)


__all__ = [
    "GraphService",
    "get_graph_service",
]
