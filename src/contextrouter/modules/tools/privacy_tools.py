"""ContextZero privacy tools for Dispatcher Agent.

These tools expose ContextZero's privacy proxy functionality
as LangChain tools that the dispatcher agent can use autonomously.

Dual-mode operation:
  1. **Local** — when contextzero package is installed, call directly
  2. **RPC** — when CONTEXTZERO_GRPC_HOST is set, use gRPC stub

When CONTEXTZERO is installed, tools are auto-registered on import.
If not installed, the module is silently skipped (see discover_all_tools).

Tools:
    anonymize_text    — Mask PII before external LLM calls
    deanonymize_text  — Restore PII in LLM responses
    check_pii         — Scan text for PII without modifying it
    apply_persona     — Inject persona system prompt
    destroy_privacy_session — Wipe encryption keys from RAM
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.tools import tool

from contextrouter.modules.tools import register_tool

logger = logging.getLogger(__name__)

# ── Lazy-initialized connections ──────────────────────────────────

_proxy_service: Any = None
_grpc_stub: Any = None


def _get_grpc_stub():
    """Get or create the gRPC ZeroService stub."""
    global _grpc_stub
    if _grpc_stub is not None:
        return _grpc_stub

    from contextrouter.core import get_core_config

    host = get_core_config().router.contextzero_grpc_host
    if not host:
        return None

    from contextcore import zero_pb2_grpc
    from contextcore.grpc_utils import create_channel_sync

    channel = create_channel_sync(host)
    _grpc_stub = zero_pb2_grpc.ZeroServiceStub(channel)
    logger.info("Connected to ContextZero gRPC at %s", host)
    return _grpc_stub


def _grpc_call(rpc_name: str, payload: dict) -> dict:
    """Make a gRPC call to ZeroService and return response payload.

    Raises:
        RuntimeError: If Zero returns an error payload.
    """
    from contextcore import ContextUnit, context_unit_pb2
    from contextcore.token_utils import create_grpc_metadata_with_token
    from contextcore.tokens import mint_service_token

    stub = _get_grpc_stub()
    if stub is None:
        raise RuntimeError("CONTEXTZERO_GRPC_HOST not configured")

    unit = ContextUnit(
        payload=payload,
        provenance=[f"router:privacy_tool:{rpc_name.lower()}"],
    )
    req = unit.to_protobuf(context_unit_pb2)

    # Service-level token with Zero permissions
    token = mint_service_token(
        "router-zero-service",
        permissions=("zero:anonymize", "zero:deanonymize"),
    )
    metadata = create_grpc_metadata_with_token(token)

    rpc = getattr(stub, rpc_name)
    resp = rpc(req, metadata=metadata)

    resp_unit = ContextUnit.from_protobuf(resp)
    result = resp_unit.payload or {}

    # Check for error responses — Zero wraps errors in ContextUnit payloads
    if "error" in result:
        error_msg = result.get("error", "Unknown Zero error")
        logger.error("Zero %s failed: %s", rpc_name, error_msg)
        raise RuntimeError(f"Zero {rpc_name}: {error_msg}")

    return result


def _use_rpc() -> bool:
    """Whether to use gRPC mode (remote Zero service)."""
    from contextrouter.core import get_core_config

    return bool(get_core_config().router.contextzero_grpc_host)


def _get_proxy_service() -> Any:
    """Get or create the singleton ProxyService (local mode).

    Uses ephemeral AES-256 encryption by default.
    Configuration is loaded from CONTEXTZERO_ environment variables.
    """
    global _proxy_service
    if _proxy_service is not None:
        return _proxy_service

    from contextzero import ProxyService
    from contextzero.config import get_config
    from contextzero.masking import MaskingConfig

    config = get_config()
    _proxy_service = ProxyService.create(
        masking_config=MaskingConfig(),
        config=config,
    )
    logger.info("Initialized ContextZero ProxyService for Router tools")
    return _proxy_service


# ============================================================================
# Tool: anonymize_text
# ============================================================================


@tool
async def anonymize_text(
    text: str,
    session_id: str,
    persona_name: str | None = None,
) -> dict[str, Any]:
    """Mask PII (names, phones, emails, medical data) before sending to external LLM.

    ALWAYS use this when the user's message contains personal data that should
    not be sent to third-party LLM providers. Returns anonymized text where
    PII is replaced by safe tokens (e.g., "Іваненко" → "NM_7f3a").

    The session_id must be kept for deanonymize_text to restore the values.

    Args:
        text: Raw text that may contain PII.
        session_id: Session identifier for consistent token mapping. Use
                    the current session_id from DispatcherState.
        persona_name: Optional persona to inject (neutral, professional,
                     creative, medical_ua).

    Returns:
        Dictionary with:
        - anonymized_text: PII-free text safe to send to LLM
        - entities_masked: Number of PII entities found
        - entity_types: Types of PII detected
        - session_id: Session used (keep for deanonymization)
    """
    if _use_rpc():
        return _grpc_call(
            "Anonymize",
            {
                "prompt": text,
                "session_id": session_id,
                "persona_name": persona_name or "",
            },
        )

    # Local mode
    from contextzero import ProxyRequest

    service = _get_proxy_service()
    resp = service.anonymize(
        ProxyRequest(
            prompt=text,
            session_id=session_id,
            persona_name=persona_name or "",
        )
    )

    return {
        "anonymized_text": resp.anonymized_prompt,
        "entities_masked": resp.entities_masked,
        "entity_types": resp.entity_types,
        "session_id": resp.session_id,
        "persona_injected": resp.persona_injected,
    }


# ============================================================================
# Tool: deanonymize_text
# ============================================================================


@tool
async def deanonymize_text(text: str, session_id: str) -> str:
    """Restore PII tokens in LLM response back to real values.

    Use AFTER receiving a response from an external LLM to replace
    anonymization tokens (e.g., "NM_7f3a") with original values
    (e.g., "Іваненко").

    The session_id MUST be the same one used in anonymize_text.

    Args:
        text: LLM response containing anonymization tokens.
        session_id: Same session_id used during anonymize_text.

    Returns:
        Text with PII restored to original values.
    """
    if _use_rpc():
        result = _grpc_call(
            "Deanonymize",
            {
                "text": text,
                "session_id": session_id,
            },
        )
        return result.get("restored_text", text)

    # Local mode
    from contextzero import DeanonymizeRequest

    service = _get_proxy_service()
    return service.deanonymize(DeanonymizeRequest(text=text, session_id=session_id))


# ============================================================================
# Tool: check_pii
# ============================================================================


@tool
async def check_pii(text: str) -> dict[str, Any]:
    """Check if text contains PII without modifying it.

    Use for audit and compliance — scan text for personal data
    (names, phones, emails, medical records) and report findings
    without making any changes.

    Args:
        text: Text to scan for PII.

    Returns:
        Dictionary with:
        - contains_pii: True if PII was detected
        - entities_found: Number of PII entities
        - entity_types: Types of PII detected
    """
    if _use_rpc():
        return _grpc_call("ScanPII", {"text": text})

    # Local mode
    from contextzero import ProxyRequest

    service = _get_proxy_service()
    scan_session = f"scan-{uuid.uuid4().hex[:8]}"
    resp = service.anonymize(ProxyRequest(prompt=text, session_id=scan_session))
    service.destroy_session(scan_session)

    return {
        "contains_pii": resp.entities_masked > 0,
        "entities_found": resp.entities_masked,
        "entity_types": resp.entity_types,
    }


# ============================================================================
# Tool: apply_persona
# ============================================================================


@tool
async def apply_persona(prompt: str, persona_name: str, session_id: str = "") -> str:
    """Apply a persona template to a prompt by injecting system instructions.

    Available personas:
    - neutral: Helpful, accurate, concise assistant
    - professional: Formal advisor with evidence-based reasoning
    - creative: Creative writing assistant
    - medical_ua: Медичний аналітик (Ukrainian medical, no diagnoses)

    Use when the response tone/style needs to match a specific persona.

    Args:
        prompt: The user's prompt to enhance.
        persona_name: Name of the persona template to apply.
        session_id: Session for persona tracking (optional).

    Returns:
        Enhanced prompt with persona system instructions prepended.
    """
    # Persona injection always uses local (no RPC needed — stateless)
    from contextzero import PersonaEngine

    engine = PersonaEngine()
    persona = engine.create_persona(persona_name, session_id=session_id or "tool-session")
    return persona.inject_into_prompt(prompt)


# ============================================================================
# Tool: destroy_privacy_session
# ============================================================================


@tool
async def destroy_privacy_session(session_id: str) -> str:
    """Destroy a privacy session and wipe all encryption keys from RAM.

    Use when a session is complete and you want to ensure NO decryption
    is possible after this point. This is a security best practice.

    Args:
        session_id: Session to destroy.

    Returns:
        Confirmation message.
    """
    if _use_rpc():
        _grpc_call("DestroySession", {"session_id": session_id})
        return f"Session '{session_id}' destroyed via gRPC."

    # Local mode
    service = _get_proxy_service()
    service.destroy_session(session_id)
    return f"Session '{session_id}' destroyed. All encryption keys wiped from RAM."


# ============================================================================
# Auto-register tools on import
# ============================================================================

_PRIVACY_TOOLS = [
    anonymize_text,
    deanonymize_text,
    check_pii,
    apply_persona,
    destroy_privacy_session,
]

for _t in _PRIVACY_TOOLS:
    register_tool(_t)

logger.info("Registered %d ContextZero privacy tools", len(_PRIVACY_TOOLS))

__all__ = [
    "anonymize_text",
    "deanonymize_text",
    "check_pii",
    "apply_persona",
    "destroy_privacy_session",
]
