"""Security validation for dispatcher gRPC service."""

from __future__ import annotations

from contextcore import ContextUnit, get_context_unit_logger
from contextcore.exceptions import SecurityError

logger = get_context_unit_logger(__name__)


def sanitize_for_struct(obj: object) -> object:
    """Recursively convert values to protobuf Struct-safe types.

    Struct only supports: None, bool, int, float, str, list, dict.
    Everything else is converted to str.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_struct(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_struct(v) for v in obj]
    # Pydantic models, LangChain messages, UUIDs, etc.
    if hasattr(obj, "model_dump"):
        return sanitize_for_struct(obj.model_dump())
    if hasattr(obj, "dict"):
        return sanitize_for_struct(obj.dict())
    return str(obj)


def validate_dispatcher_access(
    unit: ContextUnit,
    context,
) -> object | None:
    """Validate dispatcher access token, scopes, and unit access.

    Prefers ``VerifiedAuthContext`` from interceptor (already cryptographically
    verified). Falls back to legacy extract for backward compatibility.

    Raises:
        SecurityError: If token is missing, expired, or lacks required permission.
            Maps to gRPC PERMISSION_DENIED via grpc_error_handler.
    """
    from contextcore.authz.context import get_auth_context
    from contextcore.authz.engine import authorize

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
        permission="router:execute",
        service="router",
        rpc_name="ExecuteDispatcher",
    )
    if decision.denied:
        raise SecurityError(decision.reason)

    # If caller provided unit scopes, enforce capability checks against unit.
    if unit.security and (unit.security.read or unit.security.write):
        try:
            from contextcore import TokenBuilder

            tb = TokenBuilder()
            tb.verify_unit_access(token, unit, operation="read")
        except PermissionError as e:
            raise SecurityError(str(e)) from e

    return token


__all__ = ["sanitize_for_struct", "validate_dispatcher_access"]
