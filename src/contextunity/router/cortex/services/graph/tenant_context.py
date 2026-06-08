"""Request-scoped tenant id for Postgres-backed graph reads."""

from __future__ import annotations

import contextvars

_request_tenant_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "graph_request_tenant_id",
    default=None,
)


def set_request_tenant_id(tenant_id: str) -> None:
    """Bind the active tenant for the current async/thread context."""
    _ = _request_tenant_id.set(tenant_id)


def get_request_tenant_id() -> str | None:
    """Return the request-scoped tenant id when set."""
    return _request_tenant_id.get()


__all__ = ["get_request_tenant_id", "set_request_tenant_id"]
