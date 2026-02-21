"""ContextShield security tools for Dispatcher Agent.

These tools expose ContextShield's AI Firewall, Policy Engine, and
compliance checking as LangChain tools for the dispatcher agent.

Dual-mode operation:
  1. **Local** — when contextshield package is installed, call directly
  2. **RPC** — when CONTEXTSHIELD_GRPC_HOST is set, use gRPC stub

When CONTEXTSHIELD is installed, tools are auto-registered on import.
If not installed, the module is silently skipped.

Tools:
    shield_scan       — Scan prompt for injection/jailbreak/PII
    check_policy      — Check if an action is allowed by policy engine
    check_compliance  — Run compliance audit on security posture
    audit_event       — Log a security audit event
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.tools import tool

from contextrouter.modules.tools import register_tool

logger = logging.getLogger(__name__)

# ── RPC client (lazy init) ────────────────────────────────────────

_grpc_stub: Any = None


def _get_grpc_stub():
    """Get or create the gRPC ShieldService stub."""
    global _grpc_stub
    if _grpc_stub is not None:
        return _grpc_stub

    from contextrouter.core import get_core_config

    host = get_core_config().router.contextshield_grpc_host
    if not host:
        return None

    from contextcore import shield_pb2_grpc
    from contextcore.grpc_utils import create_channel_sync

    channel = create_channel_sync(host)
    _grpc_stub = shield_pb2_grpc.ShieldServiceStub(channel)
    logger.info("Connected to ContextShield gRPC at %s", host)
    return _grpc_stub


def _grpc_call(rpc_name: str, payload: dict) -> dict:
    """Make a gRPC call to ShieldService and return response payload."""
    from contextcore import ContextUnit, context_unit_pb2

    stub = _get_grpc_stub()
    if stub is None:
        raise RuntimeError("CONTEXTSHIELD_GRPC_HOST not configured")

    unit = ContextUnit(
        payload=payload,
        provenance=[f"router:security_tool:{rpc_name.lower()}"],
    )
    req = unit.to_protobuf(context_unit_pb2)

    rpc = getattr(stub, rpc_name)
    resp = rpc(req)

    resp_unit = ContextUnit.from_protobuf(resp)
    return resp_unit.payload or {}


def _use_rpc() -> bool:
    """Whether to use gRPC mode (remote Shield service)."""
    from contextrouter.core import get_core_config

    return bool(get_core_config().router.contextshield_grpc_host)


# ============================================================================
# Tool: shield_scan
# ============================================================================


@tool
async def shield_scan(
    text: str, context: str = "", validators: list[str] | None = None
) -> dict[str, Any]:
    """Scan text for prompt injection, jailbreak attempts, and PII leaks.

    Use this tool BEFORE sending any user input to an external LLM.
    It detects:
    - Prompt injection attacks
    - Jailbreak attempts
    - PII in text (names, emails, phones, medical data)
    - RAG context poisoning

    Args:
        text: Text to scan (user message, prompt, or RAG context).
        context: Additional context (e.g. RAG-retrieved chunks).
        validators: Optional list of additional validators: "pii", "rag_context".

    Returns:
        Dictionary with:
        - allowed: True if no threats detected
        - blocked: True if text should be blocked
        - threats: List of detected threats with details
        - risk_score: 0.0 (safe) to 1.0 (critical)
        - severity: Severity level of the worst threat
        - latency_ms: Processing time
    """
    if _use_rpc():
        return _grpc_call(
            "Scan",
            {
                "text": text,
                "context": context,
                "validators": validators or [],
            },
        )

    # Local mode — direct package call
    from contextshield import Shield

    shield = Shield()

    # Add specific validators if requested
    if validators:
        if "pii" in validators:
            from contextshield import PIIValidator

            shield._validators.append(PIIValidator())
        if "rag_context" in validators:
            from contextshield import RAGContextValidator

            shield._validators.append(RAGContextValidator())

    result = shield.check(user_input=text, context=context)

    return {
        "allowed": result.allowed,
        "blocked": result.blocked,
        "threats": [
            {
                "validator": f.validator,
                "reason": f.reason,
                "severity": f.severity.name,
            }
            for f in result.flags
        ],
        "risk_score": result.severity.value / 3.0,  # Normalize to 0-1 range
        "severity": result.severity.name,
        "latency_ms": result.latency_ms,
    }


# ============================================================================
# Tool: check_policy
# ============================================================================


@tool
async def check_policy(
    action: str,
    resource: str,
    tenant_id: str = "default",
    permissions: list[str] | None = None,
    token_id: str | None = None,
) -> dict[str, Any]:
    """Check if an action is allowed by the ContextShield policy engine.

    Use this to verify authorization before performing sensitive operations.
    The policy engine evaluates declarative rules including:
    - Permission-based access control (uses ContextToken internally)
    - Tenant isolation
    - Time-based restrictions
    - Operation-level guards

    Args:
        action: The action to check (e.g., "read", "write", "execute", "delete").
        resource: The resource being accessed (e.g., "patient_records", "brain_data").
        tenant_id: Tenant identifier for multi-tenant isolation.
        permissions: List of permissions the caller has.
        token_id: Optional token ID for audit trail.

    Returns:
        Dictionary with:
        - allowed: True if action is permitted
        - reason: Explanation of the decision
        - matched_policy: Name of the policy that matched (if any)
        - evaluation_ms: How long evaluation took
    """
    from contextrouter.cortex.runtime_context import get_current_access_token

    active_token = get_current_access_token()
    if not active_token:
        return {
            "allowed": False,
            "reason": "No active access token found in runtime context.",
            "matched_policy": "",
            "evaluation_ms": 0.0,
        }

    # Override tenant_id with authoritative value if available
    tenant_id = active_token.allowed_tenants[0] if active_token.allowed_tenants else "default"
    # Override permissions to what the token actually has
    permissions = list(active_token.permissions)
    token_id = active_token.token_id

    if _use_rpc():
        return _grpc_call(
            "EvaluatePolicy",
            {
                "token_id": token_id or "",
                "permissions": permissions or [],
                "tenant_id": tenant_id,
                "operation": action,
                "resource": resource,
            },
        )

    # Local mode — construct ContextToken and evaluate
    from contextcore.tokens import ContextToken
    from contextshield import PermissionCondition, Policy, PolicyEngine

    token = ContextToken(
        token_id=token_id or f"tool-{uuid.uuid4().hex[:8]}",
        permissions=tuple(permissions or []),
        allowed_tenants=(tenant_id,) if tenant_id != "default" else (),
    )

    # Default policies — in production these would come from config
    engine = PolicyEngine(
        policies=[
            Policy(
                name="allow-matching-permissions",
                effect="allow",
                conditions=(PermissionCondition(permission=f"{resource}:{action}"),),
            ),
            Policy(
                name="allow-admin",
                effect="allow",
                conditions=(PermissionCondition(permission="admin:*"),),
                priority=10,
            ),
        ],
        default_effect="deny",
    )

    result = engine.evaluate(
        token=token,
        context={"operation": action, "resource": resource, "tenant_id": tenant_id},
    )

    return {
        "allowed": result.allowed,
        "reason": result.reason,
        "matched_policy": result.matched_policy or "",
        "evaluation_ms": result.evaluation_ms,
    }


# ============================================================================
# Tool: check_compliance
# ============================================================================


@tool
async def check_compliance(
    standards: list[str] | None = None,
) -> dict[str, Any]:
    """Run a compliance audit on the current security posture.

    Checks:
    - Whether signing backends are properly configured
    - Token encryption status
    - Policy engine health
    - Audit trail integrity
    - TLS configuration

    Args:
        standards: List of standards to check ("soc2", "gdpr", "hipaa", "pci_dss").
                   Default: all standards.

    Returns:
        Dictionary with:
        - compliant: True if all checks pass
        - score: 0-100 compliance score
        - findings: List of compliance findings
        - summary: Human-readable summary
    """
    if _use_rpc():
        return _grpc_call(
            "CheckCompliance",
            {
                "standards": standards or [],
            },
        )

    # Local mode
    from contextshield import ComplianceChecker

    checker = ComplianceChecker(standards=standards)
    report = checker.check()

    return {
        "compliant": report.passed,
        "score": report.overall_score,
        "findings": [
            {
                "check_id": f.check_id,
                "standard": f.standard.value if hasattr(f.standard, "value") else str(f.standard),
                "description": f.description,
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
            }
            for f in report.findings
        ],
        "summary": report.summary(),
    }


# ============================================================================
# Tool: audit_event
# ============================================================================


@tool
async def audit_event(
    event_type: str,
    description: str,
    tenant_id: str = "default",
    actor: str = "",
    metadata: dict[str, str] | None = None,
) -> str:
    """Log a security audit event to the ContextShield audit trail.

    Use to record security-relevant events such as:
    - Access control decisions
    - PII access attempts
    - Policy violations
    - Authentication events

    Args:
        event_type: Type of event. Predefined types include:
                   "shield.check", "shield.block", "token.mint",
                   "token.verify", "policy.evaluate", "policy.deny",
                   "pii.mask", "pii.unmask", "pii.leak_detected".
        description: Human-readable description of the event.
        tenant_id: Tenant the event relates to.
        actor: Who performed the action.
        metadata: Optional key-value metadata for the event.

    Returns:
        Confirmation message with event details.
    """
    from contextrouter.cortex.runtime_context import get_current_access_token

    active_token = get_current_access_token()
    if not active_token:
        return "Error: No active access token found in runtime context. Cannot record audit event."

    # Override tenant_id with authoritative value if available
    tenant_id = active_token.allowed_tenants[0] if active_token.allowed_tenants else "default"

    if _use_rpc():
        _grpc_call(
            "RecordAudit",
            {
                "event_type": event_type,
                "description": description,
                "tenant_id": tenant_id,
                "actor": actor,
                "metadata": metadata or {},
            },
        )
        return f"Audit event recorded via gRPC: {event_type}"

    # Local mode
    from contextshield import AuditEvent, AuditEventType, AuditTrail

    trail = AuditTrail()

    # Map string to enum if possible
    try:
        etype = AuditEventType(event_type)
    except ValueError:
        etype = AuditEventType.SHIELD_CHECK  # Fallback

    event = AuditEvent(
        event_type=etype,
        actor=actor,
        tenant=tenant_id,
        details={"description": description, **(metadata or {})},
    )
    trail.record(event)

    return f"Audit event recorded: {event_type} (tenant: {tenant_id})"


# ============================================================================
# Auto-register tools on import
# ============================================================================

_SECURITY_TOOLS = [
    shield_scan,
    check_policy,
    check_compliance,
    audit_event,
]

for _t in _SECURITY_TOOLS:
    register_tool(_t)

logger.info("Registered %d ContextShield security tools", len(_SECURITY_TOOLS))

__all__ = [
    "shield_scan",
    "check_policy",
    "check_compliance",
    "audit_event",
]
