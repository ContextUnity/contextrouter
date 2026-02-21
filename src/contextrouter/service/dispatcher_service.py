"""Dispatcher Agent gRPC Service.

Implements RouterService dispatcher methods with ContextUnit protocol
and SecurityScopes-based access control.

Registrations are persisted to Redis so that Router restarts
automatically restore all project tools and graphs.

The class is composed from mixins:
- ExecutionMixin:   ExecuteAgent, ExecuteDispatcher, StreamDispatcher
- RegistrationMixin: RegisterTools, DeregisterTools, graph management
- PersistenceMixin:  Redis-backed registration persistence and recovery
"""

from __future__ import annotations

import threading

from contextcore import get_context_unit_logger, router_pb2_grpc
from contextcore.security import get_security_guard

from contextrouter.service.mixins import (
    ExecutionMixin,
    PersistenceMixin,
    RegistrationMixin,
    StreamMixin,
)

logger = get_context_unit_logger(__name__)


class DispatcherService(
    ExecutionMixin,
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
        self._guard = get_security_guard()
        # Track registered tools per project for cleanup
        self._project_tools: dict[str, list[str]] = {}
        self._project_graphs: dict[str, str] = {}
        self._project_configs: dict[str, dict] = {}  # project_id → graph config
        # Per-registration one-time stream secrets (thread-safe)
        self._stream_secrets: dict[str, str] = {}
        self._stream_secrets_lock = threading.Lock()


__all__ = ["DispatcherService"]
