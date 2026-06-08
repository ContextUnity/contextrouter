"""Inline Shield firewall check for Router execution mixins.

Calls Shield's Scan RPC via gRPC to check user input for prompt
injection / jailbreak before LLM invocation.

Shield is an optional Enterprise gRPC service. If CU_SHIELD_GRPC_URL
is not configured, the check is skipped (open-source mode).
"""

from __future__ import annotations

from dataclasses import dataclass

from contextunity.core.logging import get_contextunit_logger

logger = get_contextunit_logger(__name__)

_shield_url: str | None = None
_shield_url_resolved = False
_DEFAULT_LOCAL_SHIELD_URL = "localhost:50054"


def _get_shield_url() -> str:
    """Lazily resolve Shield gRPC URL from Router config (cached).

    Uses the Router-specific config where ``_resolve_service_endpoints``
    has already set ``shield_grpc_host`` to empty when Shield is not
    reachable.  The core config always has a default (localhost:50054)
    and must NOT be used here.
    """
    global _shield_url, _shield_url_resolved
    if _shield_url_resolved:
        return _shield_url or ""
    _shield_url_resolved = True

    try:
        from contextunity.router.core import get_core_config as get_router_config

        url = (get_router_config().shield_url or "").strip()
    except Exception:  # graceful-degrade: Shield check failure is non-blocking
        url = ""

    if url:
        _shield_url = url
        logger.info("Shield firewall enabled for inline checks: %s", url)
    else:
        _shield_url = None
        logger.debug("Shield not configured — inline content checks disabled")
    return _shield_url or ""


@dataclass
class ShieldCheckResult:
    """Minimal result from inline Shield check.

    Attributes:
        blocked: Whether the input was blocked.
        reason: Reason for blocking (if blocked).
        mode: How the check was resolved:
            - "shield" — Shield gRPC service was called.
            - "passthrough" — Shield not configured, input allowed without check.
    """

    blocked: bool = False
    reason: str = ""
    mode: str = "passthrough"


_shield_channel = None


def _get_shield_channel(url: str):
    """Get or create the global async gRPC channel for Shield."""
    global _shield_channel
    if _shield_channel is None:
        from contextunity.core.grpc_utils import create_channel

        from contextunity.router.core import get_core_config as get_router_config

        _shield_channel = create_channel(url, config=get_router_config())
    return _shield_channel


async def check_user_input(
    user_input: str,
    *,
    request_id: str = "",
    tenant: str = "",
) -> ShieldCheckResult:
    """Check user input through Shield Scan RPC (if configured).

    Makes a gRPC call to Shield's Scan endpoint. If Shield is not
    configured (no CU_SHIELD_GRPC_URL), returns allowed immediately.
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
        from contextunity.core import contextunit_pb2, shield_pb2_grpc
        from contextunity.core.sdk import ContextUnit

        from contextunity.router.service.shield_client import shield_metadata

        unit = ContextUnit(
            payload={"text": user_input},
            provenance=[f"router:shield_check:{tenant}"],
        )
        req = unit.to_protobuf(contextunit_pb2)

        channel = _get_shield_channel(shield_url)
        stub = shield_pb2_grpc.ShieldServiceStub(channel)

        # Shield firewall scan relies on SPOT caller token metadata
        resp = await stub.Scan(req, timeout=5.0, metadata=shield_metadata())

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

        return ShieldCheckResult(blocked=bool(blocked), reason=str(reason), mode="shield")

    except Exception as e:  # graceful-degrade: Shield check failure is non-blocking
        # Fail-closed: Shield is explicitly enabled but unreachable.
        # Silently bypassing an enabled firewall is a security breach.
        req_id = request_id[:12] if request_id else "?"
        logger.error(
            "Shield Scan RPC failed (BLOCKING request): %s | req=%s",
            e,
            req_id,
        )
        return ShieldCheckResult(blocked=True, reason="Shield unavailable", mode="shield")
