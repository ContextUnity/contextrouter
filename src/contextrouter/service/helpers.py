"""Helper functions for Router gRPC service."""

from __future__ import annotations

import uuid
from typing import Any

from contextcore import ContextUnit, SecurityScopes, context_unit_pb2


def parse_unit(request) -> ContextUnit:
    """Parse protobuf request to ContextUnit."""
    return ContextUnit.from_protobuf(request)


def make_response(
    payload: dict[str, Any],
    trace_id: str | None = None,
    security: SecurityScopes | None = None,
    parent_unit: ContextUnit | None = None,
) -> bytes:
    """Create ContextUnit response protobuf.

    Args:
        payload: Response payload data
        trace_id: Trace identifier (inherited from parent_unit/request if None)
        security: Security scopes (inherited from request if None)
        parent_unit: Parent ContextUnit to inherit trace_id from

    Returns:
        Serialized protobuf bytes
    """
    if trace_id is None and parent_unit:
        trace_id = parent_unit.trace_id

    kwargs: dict[str, Any] = {
        "payload": payload,
        "trace_id": trace_id or uuid.uuid4(),
    }
    if security is not None:
        kwargs["security"] = security

    unit = ContextUnit(**kwargs)
    return unit.to_protobuf(context_unit_pb2)


def router_error_response_factory(request: Any, context: Any, error: Exception) -> bytes:
    """Factory to create standardized ContextUnit error responses for Router.

    Used with contextcore.exceptions.grpc_error_handler to prevent gRPC aborts
    and instead return standard ContextUnity protocol payloads to clients.
    """
    if isinstance(error, ValueError):
        error_type = "validation"
    elif isinstance(error, PermissionError):
        error_type = "permission_denied"
    else:
        # e.g., 'SecurityError', 'ConfigurationError', etc.
        error_type = type(error).__name__

    try:
        unit = parse_unit(request)
        trace_id = str(unit.trace_id)
        security = unit.security
    except Exception:
        # Fallback if request is malformed
        trace_id = str(uuid.uuid4())
        security = None

    return make_response(
        payload={"error": str(error) or repr(error), "error_type": error_type},
        trace_id=trace_id,
        security=security,
    )


__all__ = ["parse_unit", "make_response", "router_error_response_factory"]
