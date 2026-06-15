"""Shield client — stateless secret operations via gRPC.

Router NEVER stores secrets in memory.  Every operation calls Shield
directly.  Even if Router's process memory is dumped, no secrets are
exposed.

Usage:
    from contextunity.router.service.shield_client import shield_put_secret, shield_verify_secret

    # Store a secret:
    shield_put_secret("acme/stream/auth_token", token_value)

    # Verify (fetch + compare + discard):
    ok = shield_verify_secret("acme/stream/auth_token", candidate, tenant_id="acme")
"""

from __future__ import annotations

import secrets

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk.types import GrpcMetadata
from contextunity.core.types import ContextUnitPayload

logger = get_contextunit_logger(__name__)

# Default Shield endpoint — overridden by core config or env
_DEFAULT_SHIELD_URL = "localhost:50054"


def get_shield_url() -> str:
    """Get Shield URL from Router config (resolved at startup).

    Returns empty string when Shield is not configured — callers
    must check before attempting gRPC calls.
    """
    try:
        from contextunity.router.core import get_core_config

        config = get_core_config()
        return config.shield_url or ""
    except Exception:  # graceful-degrade: Shield unavailable, continue without
        return ""


_get_shield_url = get_shield_url


def _shield_stub(shield_url: str | None = None):
    """Create a Shield gRPC stub.  Caller MUST close the channel."""
    from contextunity.core import shield_pb2_grpc
    from contextunity.core.grpc_utils import create_channel_sync

    from contextunity.router.core import get_core_config as get_router_config

    url = shield_url or _get_shield_url()
    channel = create_channel_sync(url, config=get_router_config())
    stub = shield_pb2_grpc.ShieldServiceStub(channel)
    return stub, channel


def shield_metadata(*, tenant_id: str | None = None) -> GrpcMetadata:
    """Create gRPC metadata for Router → Shield calls (SPOT, fail-closed).

    Propagates the verified caller token from gRPC auth context, or the
    attenuated graph token from ``secure_node``. Never mints a local service token.

    Args:
        tenant_id: Reserved for future tenant-scoped metadata hints (unused).

    Raises:
        SecurityError: When no caller token is available in the current context.
    """
    _ = tenant_id
    from contextunity.core.authz.context import get_auth_context
    from contextunity.core.exceptions import SecurityError
    from contextunity.core.signing import get_signing_backend
    from contextunity.core.token_utils import serialize_token

    from contextunity.router.core.context import get_current_access_token

    auth_ctx = get_auth_context()
    if auth_ctx and auth_ctx.token_string:
        return (("authorization", f"Bearer {auth_ctx.token_string}"),)

    graph_token = get_current_access_token()
    if graph_token is not None:
        from contextunity.core.token_utils.serialization import LocalSigningBackend

        backend = get_signing_backend(project_id="router")
        if not isinstance(backend, LocalSigningBackend):
            raise SecurityError(
                "Router→Shield graph token requires a local signing backend; "
                "cannot serialize ContextToken without sign()."
            )
        token_str = serialize_token(graph_token, backend=backend)
        return (("authorization", f"Bearer {token_str}"),)

    raise SecurityError(
        "Router→Shield requires an active ContextToken; "
        "background service-token minting is forbidden."
    )


_shield_metadata = shield_metadata


def shield_put_secret(
    path: str,
    value: str,
    *,
    shield_url: str | None = None,
    tenant_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> ContextUnitPayload:
    """Store a secret in Shield via gRPC ``PutSecret``. Raises ``PlatformServiceError`` on remote failure."""
    from contextunity.core import ContextUnit, contextunit_pb2
    from contextunity.core.sdk.models import SecurityScopes

    stub, channel = _shield_stub(shield_url)

    unit = ContextUnit(
        payload={
            "path": path,
            "value": value,
            "created_by": "contextunity.router:shield_client",
            "tags": tags or {"type": "stream_auth_token"},
            "tenant_id": tenant_id or "default",
        },
        provenance=["contextunity.router:shield_client:put_secret"],
        security=SecurityScopes(write=["secrets:write"]),
    )
    pb = unit.to_protobuf(contextunit_pb2)

    metadata = _shield_metadata(tenant_id=tenant_id)

    try:
        response = stub.PutSecret(pb, metadata=tuple(metadata))
        from contextunity.core.sdk.payload import wire_payload_from_field

        result: ContextUnitPayload = wire_payload_from_field(response.payload)
        if not result:
            from contextunity.core.exceptions import PlatformServiceError

            raise PlatformServiceError("Shield PutSecret returned invalid payload")
        if result.get("error"):
            from contextunity.core.exceptions import PlatformServiceError

            raise PlatformServiceError(
                f"Shield PutSecret failed: path={path}, error={result.get('message', result['error'])}"
            )
        logger.info("Shield: stored secret at path=%s", path)
        return result
    finally:
        channel.close()


def shield_get_secret(
    path: str,
    *,
    shield_url: str | None = None,
    tenant_id: str | None = None,
) -> str | None:
    """Fetch a secret value from Shield.  Returns value or None.

    Returns None immediately if Shield is not configured (no URL).
    Logs a warning only when Shield IS configured but the RPC fails.
    """
    url = shield_url or _get_shield_url()
    if not url:
        return None

    from contextunity.core import ContextUnit, contextunit_pb2
    from contextunity.core.sdk.models import SecurityScopes

    stub, channel = _shield_stub(url)

    unit = ContextUnit(
        payload={"path": path, "tenant_id": tenant_id or "default"},
        provenance=["contextunity.router:shield_client:get_secret"],
        security=SecurityScopes(read=["secrets:read"]),
    )
    pb = unit.to_protobuf(contextunit_pb2)

    metadata = _shield_metadata(tenant_id=tenant_id)

    from contextunity.core.exceptions import PlatformServiceError, SecurityError

    try:
        response = stub.GetSecret(pb, metadata=tuple(metadata))
        from contextunity.core.sdk.payload import wire_payload_from_field

        result: ContextUnitPayload = wire_payload_from_field(response.payload)
        if not result:
            raise PlatformServiceError("Shield GetSecret returned invalid payload")
        if result.get("error"):
            raise PlatformServiceError(
                f"Shield GetSecret failed: path={path}, error={result.get('error')}"
            )
        value = result.get("value")
        return value if isinstance(value, str) else None
    except PlatformServiceError:
        raise  # re-raise our own error above
    except Exception as e:  # graceful-degrade: Shield unavailable, continue without
        # Fail-closed: Shield is explicitly enabled but unreachable
        raise SecurityError(f"Shield secret retrieval failed (fail-closed): path={path}") from e
    finally:
        channel.close()


def shield_verify_secret(
    path: str,
    candidate: str,
    *,
    shield_url: str | None = None,
    tenant_id: str | None = None,
) -> bool:
    """Verify a candidate secret against Shield — stateless, constant-time.

    Fetches the stored value from Shield, compares using ``secrets.compare_digest``
    (constant-time to prevent timing attacks), then DISCARDS.  Router never
    stores the secret — even during verification it exists only as a local
    variable that goes out of scope immediately.

    Returns:
        True if match, False otherwise.
    """
    stored = shield_get_secret(path, shield_url=shield_url, tenant_id=tenant_id)
    if stored is None:
        logger.warning(
            "Shield verify: no stored secret at path=%s",
            path,
        )
        return False

    # Constant-time comparison (prevents timing-based side-channel)
    match = secrets.compare_digest(stored.encode(), candidate.encode())

    if not match:
        logger.warning(
            "Shield verify: secret mismatch for path=%s",
            path,
        )

    # `stored` goes out of scope here — never stored as attribute/field
    return match


def generate_stream_secret() -> str:
    """Generate a cryptographically secure stream authentication token.

    Uses ``secrets.token_urlsafe(32)`` — 256 bits of entropy.
    """
    return secrets.token_urlsafe(32)


__all__ = [
    "shield_metadata",
    "shield_put_secret",
    "shield_get_secret",
    "shield_verify_secret",
    "generate_stream_secret",
]
