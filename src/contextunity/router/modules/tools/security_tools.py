"""contextunity.shield security tools for Dispatcher Agent.

These tools expose contextunity.shield's AI Firewall, Policy Engine, and
compliance checking as LangChain tools for the dispatcher agent.

Dual-mode operation:
  1. **Local** — when contextunity.shield package is installed, call directly
  2. **RPC** — when CU_SHIELD_GRPC_HOST is set, use gRPC stub

When contextunity.shield is installed, tools are auto-registered on import.
If not installed, the module is silently skipped.

Tools:
    shield_scan       — Scan prompt for injection/jailbreak/PII
    check_policy      — Check if an action is allowed by policy engine
    check_compliance  — Run compliance audit on security posture
    audit_event       — Log a security audit event
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list

from contextunity.router.langchain_boundaries import tool
from contextunity.router.modules.tools import register_tool
from contextunity.router.modules.tools.schemas import SecurityResult

logger = get_contextunit_logger(__name__)

# ── RPC client (lazy init) ────────────────────────────────────────

_grpc_stub: object | None = None


def _get_grpc_stub() -> object | None:
    """Get or create the gRPC ShieldService stub."""
    global _grpc_stub
    if _grpc_stub is not None:
        return _grpc_stub

    from contextunity.router.core import get_core_config

    config = get_core_config()
    host = config.shield_url
    if not host:
        return None

    from contextunity.core import shield_pb2_grpc
    from contextunity.core.grpc_utils import create_channel_sync

    channel = create_channel_sync(host, config=config)
    _grpc_stub = shield_pb2_grpc.ShieldServiceStub(channel)
    logger.info("Connected to contextunity.shield gRPC at %s", host)
    return _grpc_stub


def _grpc_call(rpc_name: str, payload: dict[str, object]) -> dict[str, object]:
    """Make a gRPC call to ShieldService and return response payload."""
    from contextunity.core import ContextUnit, contextunit_pb2

    from contextunity.router.service.shield_client import shield_metadata

    stub = _get_grpc_stub()
    if stub is None:
        from contextunity.core.exceptions import ConfigurationError

        raise ConfigurationError("CU_SHIELD_GRPC_HOST not configured")

    unit = ContextUnit(
        payload=payload,
        provenance=[f"router:security_tool:{rpc_name.lower()}"],
    )
    req = unit.to_protobuf(contextunit_pb2)

    metadata = shield_metadata()

    rpc_method_obj: object = getattr(stub, rpc_name, None)
    if not callable(rpc_method_obj):
        from contextunity.core.exceptions import ConfigurationError

        raise ConfigurationError(f"Shield RPC '{rpc_name}' is not callable")
    rpc_method: Callable[..., object] = rpc_method_obj
    resp_obj: object = rpc_method(req, metadata=metadata)

    if not isinstance(resp_obj, contextunit_pb2.ContextUnit):
        from contextunity.core.exceptions import ConfigurationError

        raise ConfigurationError(f"Shield RPC '{rpc_name}' returned unexpected type")

    resp_unit = ContextUnit.from_protobuf(resp_obj)
    return resp_unit.payload or {}


def _use_rpc() -> bool:
    """Whether to use gRPC mode (remote Shield service)."""
    return _get_grpc_stub() is not None


def _get_bool(d: dict[str, object], key: str, default: bool = False) -> bool:
    val = d.get(key)
    return bool(val) if isinstance(val, bool) else default


def _get_float(d: dict[str, object], key: str, default: float = 0.0) -> float:
    val = d.get(key)
    return float(val) if isinstance(val, (int, float, str)) else default


def _get_str(d: dict[str, object], key: str, default: str = "") -> str:
    val = d.get(key)
    return str(val) if val is not None else default


def _get_list_of_dicts(d: dict[str, object], key: str) -> list[dict[str, str]]:
    raw = d.get(key)
    if not is_object_list(raw):
        return []
    res: list[dict[str, str]] = []
    for item in raw:
        if not is_object_dict(item):
            continue
        row: dict[str, str] = {}
        for field_key, field_val in item.items():
            row[field_key] = str(field_val)
        res.append(row)
    return res


# ============================================================================
# Tool: shield_scan
# ============================================================================


@tool
async def shield_scan(
    text: str, context: str = "", validators: list[str] | None = None
) -> SecurityResult:
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
        resp = _grpc_call(
            "Scan",
            {
                "text": text,
                "context": context,
                "validators": validators or [],
            },
        )
        return {
            "success": True,
            "allowed": _get_bool(resp, "allowed"),
            "blocked": _get_bool(resp, "blocked"),
            "threats": _get_list_of_dicts(resp, "threats"),
            "risk_score": _get_float(resp, "risk_score"),
            "severity": _get_str(resp, "severity"),
            "latency_ms": _get_float(resp, "latency_ms"),
        }

    # Local mode — direct package call
    from contextunity.shield import Shield

    shield = Shield()

    # Add specific validators if requested
    if validators:
        validators_list: list[object] = getattr(shield, "_validators", [])
        if "pii" in validators:
            from contextunity.shield import PIIValidator

            validators_list.append(PIIValidator())
        if "rag_context" in validators:
            from contextunity.shield import RAGContextValidator

            validators_list.append(RAGContextValidator())

    result = shield.check(user_input=text, context=context)

    return {
        "success": True,
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
) -> SecurityResult:
    """Check if an action is allowed by the contextunity.shield policy engine.

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
    tenant_id = tenant_id or "default"

    if _use_rpc():
        resp = _grpc_call(
            "EvaluatePolicy",
            {
                "token_id": token_id or "",
                "permissions": permissions or [],
                "tenant_id": tenant_id,
                "operation": action,
                "resource": resource,
            },
        )
        return {
            "success": True,
            "allowed": _get_bool(resp, "allowed"),
            "reason": _get_str(resp, "reason"),
            "matched_policy": _get_str(resp, "matched_policy"),
            "evaluation_ms": _get_float(resp, "evaluation_ms"),
        }

    # Local mode — construct ContextToken and evaluate
    from contextunity.core.tokens import ContextToken
    from contextunity.shield import PermissionCondition, Policy, PolicyEngine

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
        "success": True,
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
) -> SecurityResult:
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
        resp = _grpc_call(
            "CheckCompliance",
            {
                "standards": standards or [],
            },
        )
        return {
            "success": True,
            "compliant": _get_bool(resp, "compliant"),
            "score": _get_float(resp, "score"),
            "findings": _get_list_of_dicts(resp, "findings"),
            "summary": _get_str(resp, "summary"),
        }

    # Local mode
    from contextunity.shield import ComplianceChecker

    checker = ComplianceChecker(standards=standards)
    report = checker.check()

    return {
        "success": True,
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
    """Log a security audit event to the contextunity.shield audit trail.

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
    tenant_id = tenant_id or "default"

    if _use_rpc():
        _ = _grpc_call(
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
    from contextunity.shield import AuditEvent, AuditEventType, AuditTrail

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

logger.info("Registered %d contextunity.shield security tools", len(_SECURITY_TOOLS))

__all__ = [
    "shield_scan",
    "check_policy",
    "check_compliance",
    "audit_event",
]
