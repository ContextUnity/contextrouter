"""Inline Shield firewall check for Router execution mixins.

Calls Shield's Scan RPC via gRPC to check user input for prompt
injection / jailbreak before LLM invocation.

Shield is an optional Enterprise gRPC service. If CONTEXTSHIELD_GRPC_URL
is not configured, the check is skipped (open-source mode).
"""

from __future__ import annotations

from dataclasses import dataclass

from contextcore.logging import get_context_unit_logger

logger = get_context_unit_logger(__name__)

_shield_url: str | None = None
_shield_url_resolved = False


def _get_shield_url() -> str:
    """Lazily resolve Shield gRPC URL from config (cached)."""
    global _shield_url, _shield_url_resolved
    if _shield_url_resolved:
        return _shield_url or ""
    _shield_url_resolved = True

    from contextcore.config import get_core_config

    url = get_core_config().shield_url
    # Default placeholder "localhost:50054" means not explicitly configured
    if url and url != "localhost:50054":
        _shield_url = url
        logger.info("Shield firewall enabled for inline checks: %s", url)
    else:
        _shield_url = None
        logger.debug("Shield not configured — inline content checks disabled")
    return _shield_url or ""


@dataclass
class ShieldCheckResult:
    """Minimal result from inline Shield check."""

    blocked: bool = False
    reason: str = ""


_shield_channel = None


def _get_shield_channel(url: str):
    """Get or create the global async gRPC channel for Shield."""
    global _shield_channel
    if _shield_channel is None:
        from contextcore.grpc_utils import create_channel_async

        _shield_channel = create_channel_async(url)
    return _shield_channel


async def check_user_input(
    user_input: str,
    *,
    request_id: str = "",
    tenant: str = "",
) -> ShieldCheckResult:
    """Check user input through Shield Scan RPC (if configured).

    Makes a gRPC call to Shield's Scan endpoint. If Shield is not
    configured (no CONTEXTSHIELD_GRPC_URL), returns allowed immediately.
    Reuses a single gRPC channel to avoid connection overhead.

    Args:
        user_input: The user's message to check.
        request_id: Request ID for logging.
        tenant: Tenant ID for logging.

    Returns:
        ShieldCheckResult with blocked/reason.
    """
    shield_url = _get_shield_url()
    if not shield_url:
        return ShieldCheckResult()

    try:
        from contextcore import context_unit_pb2, shield_pb2_grpc
        from contextcore.sdk import ContextUnit

        from contextrouter.service.shield_client import _shield_metadata

        unit = ContextUnit(
            payload={"text": user_input},
            provenance=[f"router:shield_check:{tenant}"],
        )
        req = unit.to_protobuf(context_unit_pb2)

        channel = _get_shield_channel(shield_url)
        stub = shield_pb2_grpc.ShieldServiceStub(channel)

        # Shield firewall scan relies on SPOT caller token metadata
        resp = await stub.Scan(req, timeout=5.0, metadata=_shield_metadata())

        resp_unit = ContextUnit.from_protobuf(resp)
        payload = resp_unit.payload or {}

        blocked = payload.get("blocked", False)
        reason = payload.get("reason", "")

        if blocked:
            logger.warning(
                "Shield BLOCKED | req=%s tenant=%s reason=%s",
                request_id[:12] if request_id else "?",
                tenant,
                reason,
            )

        return ShieldCheckResult(blocked=bool(blocked), reason=str(reason))

    except Exception as e:
        logger.warning(
            "Shield Scan RPC failed (allowing request): %s | req=%s",
            e,
            request_id[:12] if request_id else "?",
        )
        # Fail-open: if Shield is down, don't block the user
        return ShieldCheckResult()
