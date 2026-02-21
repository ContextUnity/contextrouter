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
    provenance: list[str] | None = None,
    security: SecurityScopes | None = None,
    parent_unit: ContextUnit | None = None,
) -> bytes:
    """Create ContextUnit response protobuf.

    Args:
        payload: Response payload data
        trace_id: Trace identifier (inherited from parent_unit/request if None)
        provenance: Provenance labels to append (extended from parent if given)
        security: Security scopes (inherited from request if None)
        parent_unit: Parent ContextUnit to inherit trace_id and provenance from

    Returns:
        Serialized protobuf bytes
    """
    if trace_id is None and parent_unit:
        trace_id = parent_unit.trace_id

    if provenance is None:
        if parent_unit:
            provenance = list(parent_unit.provenance) + ["router:response"]
        else:
            provenance = ["router:response"]
    elif parent_unit:
        provenance = list(parent_unit.provenance) + provenance

    unit = ContextUnit(
        payload=payload,
        trace_id=trace_id or uuid.uuid4(),
        provenance=provenance,
        security=security,
    )
    return unit.to_protobuf(context_unit_pb2)


def check_security_scopes(
    unit: ContextUnit,
    required_read: list[str] | None = None,
    required_write: list[str] | None = None,
) -> bool:
    """Check if ContextUnit has required security scopes.

    Args:
        unit: ContextUnit to check
        required_read: Required read scopes (e.g., ["dispatcher:execute"])
        required_write: Required write scopes

    Returns:
        True if all required scopes are present
    """
    if not unit.security:
        return False

    if required_read:
        unit_read = set(unit.security.read or [])
        required_read_set = set(required_read)
        if not required_read_set.issubset(unit_read):
            return False

    if required_write:
        unit_write = set(unit.security.write or [])
        required_write_set = set(required_write)
        if not required_write_set.issubset(unit_write):
            return False

    return True


__all__ = ["parse_unit", "make_response", "check_security_scopes"]
