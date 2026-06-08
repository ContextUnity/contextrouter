"""Security validation for dispatcher gRPC service."""

from __future__ import annotations

from contextunity.core import ContextToken, ContextUnit, get_contextunit_logger
from contextunity.core.exceptions import SecurityError
from contextunity.core.permissions import Permissions
from contextunity.core.permissions.access import has_introspection_access
from contextunity.core.types import is_object_dict, is_object_list, is_object_tuple

logger = get_contextunit_logger(__name__)


def sanitize_for_struct(obj: object) -> object:
    """Recursively coerce values into protobuf ``Struct``-safe types.

    Handles nested dicts, lists, Pydantic models (``model_dump``),
    and legacy ``dict()`` objects. Unsupported types are stringified.

    Args:
        obj: Arbitrary nested structure.

    Returns:
        Same structure with all values safe for ``google.protobuf.Struct``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_object_dict(obj):
        sanitized: dict[str, object] = {}
        for key, value in obj.items():
            sanitized[key] = sanitize_for_struct(value)
        return sanitized
    if is_object_list(obj):
        return [sanitize_for_struct(value) for value in obj]
    if is_object_tuple(obj):
        return [sanitize_for_struct(value) for value in obj]
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class ModelDumpable(Protocol):
        """Structural type for Pydantic v2+ models with ``model_dump``."""

        def model_dump(self) -> dict[str, object]:
            """Serialize model to a plain dict.

            Returns:
                Dict of field name to value.
            """
            ...

    @runtime_checkable
    class Dictable(Protocol):
        """Structural type for objects with a legacy ``dict()`` method."""

        def dict(self) -> dict[str, object]:
            """Serialize to a plain dict (Pydantic v1 style).

            Returns:
                Dict of field name to value.
            """
            ...

    if isinstance(obj, ModelDumpable):
        return sanitize_for_struct(obj.model_dump())
    if isinstance(obj, Dictable):
        return sanitize_for_struct(obj.dict())
    return str(obj)


def validate_dispatcher_access(
    unit: ContextUnit,
    context: object,
    *,
    permission: str = Permissions.ROUTER_EXECUTE,
    rpc_name: str = "ExecuteDispatcher",
) -> "ContextToken":
    """Validate access token, permissions, and unit-level capability.

    Enforces fail-closed: missing or expired tokens always reject.
    Uses the canonical ``authorize()`` engine for permission checks.

    Args:
        unit: Incoming ContextUnit with optional security scopes.
        context: gRPC request context (unused directly, kept for API contract).
        permission: Required permission string.
        rpc_name: RPC method name for audit logging.

    Returns:
        Validated ``ContextToken``.

    Raises:
        SecurityError: If authentication or authorization fails.
    """
    _ = context
    from contextunity.core.authz.context import get_auth_context
    from contextunity.core.authz.engine import authorize

    auth_ctx = get_auth_context()
    token = auth_ctx.token if auth_ctx else None

    # Security is always enforced (fail-closed)
    if token is None:
        raise SecurityError("Missing ContextToken — authentication required")
    if token.is_expired():
        raise SecurityError("ContextToken expired — please obtain a new token")

    # Use canonical authorize() engine
    decision = authorize(
        auth_ctx if auth_ctx is not None else token,
        permission=permission,
        service="router",
        rpc_name=rpc_name,
    )
    if decision.denied:
        raise SecurityError(decision.reason)

    # If caller provided unit scopes, enforce capability checks against unit.
    if unit.security and (unit.security.read or unit.security.write):
        try:
            from contextunity.core import TokenBuilder

            tb = TokenBuilder()
            tb.verify_unit_access(token, unit, operation="read")
        except PermissionError as e:
            raise SecurityError(str(e)) from e

    return token


def validate_introspection_access(
    unit: ContextUnit,
    context: object,
    *,
    project_id: str | None = None,
) -> ContextToken:
    """Validate RPC auth and project-scoped introspection rights.

    ``allowed_tenants`` on the token does **not** gate which ``project_id``
    may be introspected — use :func:`~contextunity.core.permissions.access.has_introspection_access`.
    """
    token = validate_dispatcher_access(
        unit,
        context,
        permission=Permissions.ROUTER_INTROSPECT,
        rpc_name="IntrospectRegistrations",
    )
    if project_id is not None and not has_introspection_access(token.permissions, project_id):
        raise SecurityError(f"Introspection denied for project '{project_id}'")
    return token


__all__ = [
    "sanitize_for_struct",
    "validate_dispatcher_access",
    "validate_introspection_access",
]
