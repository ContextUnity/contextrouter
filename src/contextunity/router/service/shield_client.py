"""Shield client — stateless secret operations via gRPC.

Router NEVER stores secrets in memory.  Every operation calls Shield
directly.  Even if Router's process memory is dumped, no secrets are
exposed.

Usage:
    from contextunity.router.service.shield_client import shield_put_secret, shield_verify_secret

    # Store a secret:
    shield_put_secret("acme/stream/auth_token", token_value, tenant_id="acme")

    # Verify (fetch + compare + discard):
    ok = shield_verify_secret("acme/stream/auth_token", candidate, tenant_id="acme")
"""

from __future__ import annotations

import secrets
from typing import Any

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)

# Default Shield endpoint — overridden by core config or env
_DEFAULT_SHIELD_URL = "localhost:50054"


def _get_shield_url() -> str:
    """Get Shield URL from core config (resolved at startup) or fallback to default."""
    try:
        from contextunity.router.core import get_core_config

        config = get_core_config()
        return config.router.shield_grpc_host or _DEFAULT_SHIELD_URL
    except Exception:
        return _DEFAULT_SHIELD_URL


def _shield_stub(shield_url: str | None = None):
    """Create a Shield gRPC stub.  Caller MUST close the channel."""
    from contextunity.core import shield_pb2_grpc
    from contextunity.core.grpc_utils import create_channel_sync

    url = shield_url or _get_shield_url()
    channel = create_channel_sync(url)
    stub = shield_pb2_grpc.ShieldServiceStub(channel)
    return stub, channel


def _shield_metadata():
    """Create gRPC metadata with service token for Shield authentication.

    Single source of truth for all Router → Shield gRPC calls.
    Propagates caller token if available (SPOT pattern).
    """
    from contextunity.core.authz.context import get_auth_context

    # Ensure caller token gets propagated to Shield
    ctx = get_auth_context()
    if ctx and ctx.token_string:
        return [("authorization", f"Bearer {ctx.token_string}")]

    # Fallback to local minting if background task
    from contextunity.core.signing import get_signing_backend
    from contextunity.core.token_utils import create_grpc_metadata_with_token
    from contextunity.core.tokens import mint_service_token

    token = mint_service_token(
        "router-shield-client",
        permissions=(
            "shield:check",
            "shield:scan",
            "shield:put_secret",
            "shield:get_secret",
            "shield:secrets:read",
        ),
    )
    backend = get_signing_backend(project_id="router")
    return create_grpc_metadata_with_token(token, backend=backend)


def shield_put_secret(
    path: str,
    value: str,
    *,
    tenant_id: str = "",
    shield_url: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Store a secret in Shield.  Returns response dict or raises."""
    from contextunity.core import ContextUnit, contextunit_pb2
    from contextunity.core.sdk.models import SecurityScopes
    from google.protobuf.json_format import MessageToDict

    stub, channel = _shield_stub(shield_url)

    unit = ContextUnit(
        payload={
            "path": path,
            "value": value,
            "tenant_id": tenant_id,
            "created_by": "contextunity.router:shield_client",
            "tags": tags or {"type": "stream_auth_token"},
        },
        provenance=["contextunity.router:shield_client:put_secret"],
        security=SecurityScopes(write=["secrets:write"]),
    )
    pb = unit.to_protobuf(contextunit_pb2)

    metadata = _shield_metadata()

    try:
        response = stub.PutSecret(pb, metadata=metadata)
        result = MessageToDict(response.payload)
        if result.get("error"):
            raise RuntimeError(f"Shield PutSecret error: {result.get('message', result['error'])}")
        logger.info("Shield: stored secret at path=%s tenant=%s", path, tenant_id)
        return result
    finally:
        channel.close()


def shield_get_secret(
    path: str,
    *,
    tenant_id: str = "",
    shield_url: str | None = None,
) -> str | None:
    """Fetch a secret value from Shield.  Returns value or None."""
    from contextunity.core import ContextUnit, contextunit_pb2
    from contextunity.core.sdk.models import SecurityScopes
    from google.protobuf.json_format import MessageToDict

    stub, channel = _shield_stub(shield_url)

    unit = ContextUnit(
        payload={"path": path, "tenant_id": tenant_id},
        provenance=["contextunity.router:shield_client:get_secret"],
        security=SecurityScopes(read=["secrets:read"]),
    )
    pb = unit.to_protobuf(contextunit_pb2)

    metadata = _shield_metadata()

    try:
        response = stub.GetSecret(pb, metadata=metadata)
        result = MessageToDict(response.payload)
        if result.get("error"):
            logger.warning("Shield GetSecret failed: path=%s error=%s", path, result.get("error"))
            return None
        return result.get("value", None)
    except Exception as e:
        logger.warning("Shield unavailable for GetSecret: path=%s error=%s", path, e)
        return None
    finally:
        channel.close()


def shield_verify_secret(
    path: str,
    candidate: str,
    *,
    tenant_id: str = "",
    shield_url: str | None = None,
) -> bool:
    """Verify a candidate secret against Shield — stateless, constant-time.

    Fetches the stored value from Shield, compares using ``secrets.compare_digest``
    (constant-time to prevent timing attacks), then DISCARDS.  Router never
    stores the secret — even during verification it exists only as a local
    variable that goes out of scope immediately.

    Returns:
        True if match, False otherwise.
    """
    stored = shield_get_secret(path, tenant_id=tenant_id, shield_url=shield_url)
    if stored is None:
        logger.warning(
            "Shield verify: no stored secret at path=%s tenant=%s",
            path,
            tenant_id,
        )
        return False

    # Constant-time comparison (prevents timing-based side-channel)
    match = secrets.compare_digest(stored.encode(), candidate.encode())

    if not match:
        logger.warning(
            "Shield verify: secret mismatch for path=%s tenant=%s",
            path,
            tenant_id,
        )

    # `stored` goes out of scope here — never stored as attribute/field
    return match


def generate_stream_secret() -> str:
    """Generate a cryptographically secure stream authentication token.

    Uses ``secrets.token_urlsafe(32)`` — 256 bits of entropy.
    """
    return secrets.token_urlsafe(32)


__all__ = [
    "shield_put_secret",
    "shield_get_secret",
    "shield_verify_secret",
    "generate_stream_secret",
]
