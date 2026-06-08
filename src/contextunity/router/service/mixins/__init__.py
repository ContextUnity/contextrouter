"""Dispatcher service mixins -- modular RPC handler groups (execution, registration, persistence, introspection)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextunity.router.service.mixins.execution import ExecutionMixin
    from contextunity.router.service.mixins.introspection import IntrospectionMixin
    from contextunity.router.service.mixins.persistence import PersistenceMixin
    from contextunity.router.service.mixins.registration import RegistrationMixin
    from contextunity.router.service.mixins.stream import StreamMixin


def __getattr__(name: str) -> object:
    """Lazily expose mixins so submodule imports cannot create payload cycles."""
    if name == "ExecutionMixin":
        from contextunity.router.service.mixins.execution import ExecutionMixin

        return ExecutionMixin
    if name == "IntrospectionMixin":
        from contextunity.router.service.mixins.introspection import IntrospectionMixin

        return IntrospectionMixin
    if name == "PersistenceMixin":
        from contextunity.router.service.mixins.persistence import PersistenceMixin

        return PersistenceMixin
    if name == "RegistrationMixin":
        from contextunity.router.service.mixins.registration import RegistrationMixin

        return RegistrationMixin
    if name == "StreamMixin":
        from contextunity.router.service.mixins.stream import StreamMixin

        return StreamMixin
    raise AttributeError(name)


__all__ = [
    "ExecutionMixin",
    "IntrospectionMixin",
    "PersistenceMixin",
    "RegistrationMixin",
    "StreamMixin",
]
