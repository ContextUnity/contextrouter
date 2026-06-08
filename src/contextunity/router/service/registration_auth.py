"""Registration token verification helpers."""

from __future__ import annotations

from contextunity.core import contextunit_pb2
from contextunity.core.authz.context import VerifiedAuthContext
from contextunity.core.exceptions import SecurityError
from contextunity.core.signing import VerifierBackend
from grpc.aio import ServicerContext

ContextUnit = contextunit_pb2.ContextUnit


def extract_registration_token_string(
    context: ServicerContext[ContextUnit, ContextUnit],
) -> str:
    """Extract bearer token from gRPC metadata for registration RPCs."""
    metadata: dict[str, str] = {}
    for key, value in context.invocation_metadata() or ():
        metadata[str(key)] = value.decode() if isinstance(value, bytes) else str(value)

    auth_header = metadata.get("authorization", "").strip()
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return ""


async def build_registration_verifier(
    *,
    token_str: str,
    project_id: str,
) -> VerifierBackend:
    """Build a verifier backend for registration/auth bootstrap."""
    parts = token_str.rsplit(".", 2)
    if len(parts) != 3:
        raise SecurityError("Registration token must use composite kid wire format")

    kid = parts[0]
    if ":" not in kid:
        raise SecurityError(
            "Registration token must use composite kid format '<project>:<key-version>'"
        )

    caller_project_id, key_version = kid.split(":", 1)
    if caller_project_id != project_id:
        raise SecurityError(
            f"Registration token project mismatch: token='{caller_project_id}', target='{project_id}'"
        )

    from contextunity.core.discovery import get_project_key

    key_data = get_project_key(project_id) or {}
    stored_secret = key_data.get("project_secret")

    if "session" in key_version:
        public_key_b64 = key_data.get("public_key_b64")
        if not public_key_b64:
            from contextunity.router.service.shield_client import get_shield_url

            shield_url = get_shield_url()
            if not shield_url:
                raise SecurityError(
                    f"No public key for project '{project_id}' and Shield is not configured."
                )

            from contextunity.core.token_utils import fetch_project_public_key_async

            from contextunity.router.core import get_core_config as get_router_config

            public_key_b64, returned_kid = await fetch_project_public_key_async(
                project_id,
                kid,
                shield_url,
                provenance="router:register_manifest:fetch_public_key",
                config=get_router_config(),
            )

            from contextunity.core.discovery import update_project_public_key

            _ = update_project_public_key(project_id, public_key_b64, returned_kid)

        try:
            from contextunity.core.ed25519 import Ed25519Backend
        except ImportError as exc:
            raise SecurityError("Ed25519 backend unavailable — missing dependency.") from exc

        return Ed25519Backend(public_key_b64=public_key_b64, kid=kid)

    if stored_secret:
        from contextunity.core.signing import HmacBackend

        return HmacBackend(project_id, stored_secret)

    raise SecurityError(
        f"No project-scoped HMAC secret available for project '{project_id}'. "
        + "Use a Shield session token or register project key material before HMAC bootstrap."
    )


async def get_verified_registration_auth_context(
    context: ServicerContext[ContextUnit, ContextUnit],
    *,
    project_id: str,
) -> VerifiedAuthContext:
    """Verify registration token and return canonical auth context."""
    from contextunity.core.token_utils import verify_token_string

    token_str = extract_registration_token_string(context)
    if not token_str:
        raise SecurityError("Missing registration token in gRPC metadata")

    backend = await build_registration_verifier(
        token_str=token_str,
        project_id=project_id,
    )
    token = verify_token_string(token_str, backend)
    if token is None:
        raise SecurityError("Registration token cryptographic verification failed")
    if token.is_expired():
        raise SecurityError(f"Registration token expired for project '{project_id}'.")

    return VerifiedAuthContext.from_token(token, token_str, project_id=project_id)
