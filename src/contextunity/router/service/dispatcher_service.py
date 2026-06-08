"""Dispatcher Agent gRPC Service.
Implements RouterService dispatcher methods with ContextUnit protocol
and SecurityScopes-based access control.
Registrations are persisted to Redis so that Router restarts
automatically restore all project tools and graphs.
The class is composed from mixins:
- ExecutionMixin:   ExecuteAgent, ExecuteDispatcher, StreamDispatcher
- RegistrationMixin: RegisterManifest, graph management
- PersistenceMixin:  Redis-backed registration persistence and recovery
"""

from __future__ import annotations

import threading

from contextunity.core import get_contextunit_logger, router_pb2_grpc

from contextunity.router.service.mixins import (
    ExecutionMixin,
    IntrospectionMixin,
    PersistenceMixin,
    RegistrationMixin,
    StreamMixin,
)
from contextunity.router.service.mixins.execution.types import (
    ProjectConfigMap,
    ProjectGraphMap,
    ProjectToolMap,
    RouterCallbackMap,
)

logger = get_contextunit_logger(__name__)


class DispatcherService(
    ExecutionMixin,
    IntrospectionMixin,
    RegistrationMixin,
    StreamMixin,
    PersistenceMixin,
    router_pb2_grpc.RouterServiceServicer,
):
    """Dispatcher Agent gRPC Service implementation.

    Composes execution, registration, and persistence concerns
    via mixin classes while inheriting from the gRPC servicer.
    """

    def __init__(self) -> None:
        """Initialize per-project state maps for tools, graphs, and streams."""
        # Track registered tools per project for cleanup
        self._project_tools: ProjectToolMap = {}
        self._project_graphs: ProjectGraphMap = {}
        self._project_configs: ProjectConfigMap = {}
        # Per-graph router_callbacks for ExecuteNode authorization
        # project_id → {graph_key → [node_names exposed for direct execution]}
        self._project_router_callbacks: RouterCallbackMap = {}
        # Per-registration one-time stream secrets (thread-safe)
        self._stream_secrets: dict[str, str] = {}
        self._stream_secrets_lock: threading.Lock = threading.Lock()


__all__ = ["DispatcherService"]
