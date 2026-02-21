"""Security validation for dispatcher gRPC service."""

from __future__ import annotations

from contextcore import ContextUnit, extract_token_from_grpc_metadata, get_context_unit_logger
from contextcore.exceptions import SecurityError

from contextrouter.core import AccessManager, get_core_config

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

    Raises:
        SecurityError: If token is missing, expired, or lacks required permission.
            Maps to gRPC PERMISSION_DENIED via grpc_error_handler.
    """
    token = extract_token_from_grpc_metadata(context)
    config = get_core_config()

    if not config.security.enabled:
        env = config.security.environment
        if env in ("production", "prod", "staging"):
            logger.critical(
                "🚨 SECURITY DISABLED IN %s — ALL PERMISSION CHECKS BYPASSED. "
                "Set SECURITY_ENABLED=true in your environment immediately. "
                "Env var: CONTEXTROUTER_SECURITY_ENABLED=true",
                env.upper(),
            )
        else:
            logger.warning(
                "⚠️ Security disabled (SECURITY_ENABLED=false). "
                "Acceptable for local development only. "
                "Set CONTEXTROUTER_SECURITY_ENABLED=true for staging/production.",
            )
        return token

    # Fail-closed: SecurityError → gRPC PERMISSION_DENIED (not INTERNAL)
    if token is None:
        raise SecurityError("Missing ContextToken — authentication required")
    if token.is_expired():
        raise SecurityError("ContextToken expired — please obtain a new token")

    access = AccessManager.from_core_config()
    try:
        access.verify_read(token, permission="dispatcher:execute")
    except PermissionError as e:
        raise SecurityError(str(e)) from e

    # If caller provided unit scopes, enforce capability checks against unit.
    if unit.security and (unit.security.read or unit.security.write):
        try:
            from contextcore import TokenBuilder

            tb = TokenBuilder(
                enabled=config.security.enabled,
                private_key_path=config.security.private_key_path,
            )
            tb.verify_unit_access(token, unit, operation="read")
        except PermissionError as e:
            raise SecurityError(str(e)) from e

    return token


__all__ = ["sanitize_for_struct", "validate_dispatcher_access"]
