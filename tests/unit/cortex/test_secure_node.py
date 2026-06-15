"""Behavioral tests for secure_node.make_secure_node.

Tests cover:
  - SecurityError on missing token (fail-closed)
  - Token injection into state (__token__ key)
  - Capability stripping (attenuated permissions)
  - Pass-through mode (infrastructure nodes)
  - PII masking scope resolution
  - Provenance recording (node + shield)
  - Sync/async node parity
  - Service scope injection
  - Execute tools scope injection
"""

import pytest
from contextunity.core.exceptions import SecurityError, TamperDetectedError
from contextunity.core.tokens import ContextToken, TokenBuilder

from contextunity.router.core.context import (
    get_accumulated_provenance,
    get_current_access_token,
    init_provenance_accumulator,
    reset_current_access_token,
    reset_provenance_accumulator,
    set_current_access_token,
)
from contextunity.router.cortex.secure_node import make_secure_node

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def root_token():
    """Broad-permission root token for testing capability stripping."""
    return TokenBuilder().mint_root(
        user_ctx={"user_id": "test-user"},
        permissions=[
            "router:execute",
            "shield:secrets:read",
            "shield:secrets:read:default/api_keys/planner/model_secret_ref",
            "brain:search",
            "tool:medical_sql",
            "tool:medical_sql:execute",
            "privacy:anonymize",
            "privacy:deanonymize",
            "privacy:check_pii",
        ],
        ttl_s=3600,
        allowed_tenants=["default"],
    )


@pytest.fixture
def _set_token(root_token):
    """Context manager that sets and cleans up token + provenance."""
    token_ref = set_current_access_token(root_token)
    accum_ref = init_provenance_accumulator()
    yield root_token
    reset_provenance_accumulator(accum_ref)
    reset_current_access_token(token_ref)


@pytest.fixture(autouse=True)
def _tenant_scope_default(monkeypatch):
    """Secure-node unit tests focus on capability stripping, not tenant resolution."""
    monkeypatch.setattr(
        "contextunity.router.cortex.secure_node.resolve_node_effective_tenants",
        lambda *args, **kwargs: ("default",),
    )


def _sync_node(state: dict, config: dict) -> dict:
    """Captures the injected __token__ for assertion."""
    return {"result": "ok", "__captured_token__": state.get("__token__")}


async def _async_node(state: dict, config: dict) -> dict:
    return {"result": "async_ok", "__captured_token__": state.get("__token__")}


# ── 1. Fail-Closed: SecurityError Without Token ──────────────────────────


@pytest.mark.asyncio
async def test_no_token_raises_security_error():
    """Node execution without an active ContextToken raises SecurityError."""
    secure = make_secure_node("blocked_node", _sync_node, requires_llm=False)
    # No token set — fail-closed
    with pytest.raises(SecurityError):
        await secure({"state": "init"}, {})


# ── 2. Token Injection Into State ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_injected_into_state(_set_token):
    """Wrapped node receives an attenuated __token__ in its state dict."""
    secure = make_secure_node("injector", _sync_node, requires_llm=False)

    result = await secure({"state": "init"}, {})

    captured = result.get("__captured_token__")
    assert captured is not None, "Node must receive __token__ in state"
    assert isinstance(captured, ContextToken)


# ── 3. Capability Stripping ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capability_stripping_narrows_permissions(_set_token):
    """Attenuated token has ONLY the scopes the node needs, not the full root set."""
    secure = make_secure_node(
        "narrow_node",
        _sync_node,
        requires_llm=False,
        service_scopes=["brain:search"],
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert captured.has_permission("brain:search"), "Granted scope must be present"
    # Root had router:execute but the node didn't request it — must be stripped
    assert not captured.has_permission("router:execute"), (
        "Permissions not requested by node must be stripped"
    )


@pytest.mark.asyncio
async def test_llm_node_gets_shield_scopes(_set_token):
    """Node with requires_llm=True and model_secret_ref gets shield:secrets:read scopes."""
    secure = make_secure_node(
        "planner",
        _sync_node,
        {"model_secret_ref": "openai-key"},
        requires_llm=True,
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert captured.has_permission("shield:secrets:read")


# ── 4. Pass-Through Mode ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pass_through_preserves_full_token(_set_token, root_token):
    """pass_through_token=True skips attenuation — full root token passes through."""
    secure = make_secure_node(
        "infra_node",
        _sync_node,
        {"pii_masking": False},
        requires_llm=False,
        pass_through_token=True,
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    # Must be the original root token, not attenuated
    assert captured.token_id == root_token.token_id
    assert captured.has_permission("router:execute"), "Pass-through must preserve all permissions"
    assert captured.has_permission("brain:search"), "Pass-through must preserve all permissions"


# ── 5. PII Masking Scope Resolution ──────────────────────────────────────


@pytest.mark.asyncio
async def test_pii_masking_adds_zero_scopes(_set_token):
    """Node with pii_masking=True gets privacy:anonymize, privacy:deanonymize, privacy:check_pii."""
    from unittest.mock import patch

    secure = make_secure_node(
        "pii_node",
        _sync_node,
        {"pii_masking": True},
        requires_llm=False,
    )

    # Mock PII session — we only need to verify token scopes, not run real pipeline
    with (
        patch("contextunity.router.cortex.secure_node.dispatch_custom_event"),
        patch("contextunity.router.cortex.utils.pii.PiiSession._create_anonymizer"),
    ):
        result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert captured.has_permission("privacy:anonymize")
    assert captured.has_permission("privacy:deanonymize")
    assert captured.has_permission("privacy:check_pii")


@pytest.mark.asyncio
async def test_no_pii_no_privacy_scopes(_set_token):
    """Node with explicit pii_masking=False does NOT get privacy:* scopes."""
    secure = make_secure_node(
        "clean_node",
        _sync_node,
        {"pii_masking": False},
        requires_llm=False,
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert not captured.has_permission("privacy:anonymize")


# ── 6. Provenance Recording ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provenance_records_node_name(_set_token):
    """Provenance accumulator records 'node:{name}' for every execution."""
    secure = make_secure_node("prov_node", _sync_node, requires_llm=False)

    await secure({"state": "init"}, {})

    history = get_accumulated_provenance()
    assert "node:prov_node" in history


@pytest.mark.asyncio
async def test_provenance_records_shield_when_enabled(_set_token):
    """Shield secrets paths are recorded when shield is enabled in project config."""
    secure = make_secure_node(
        "planner",
        _sync_node,
        {"model_secret_ref": "openai-key"},
        requires_llm=True,
    )

    state = {
        "state": "init",
        "metadata": {"project_config": {"services": {"shield": {"enabled": True}}}},
    }
    await secure(state, {})

    history = get_accumulated_provenance()
    assert "node:planner" in history
    shield_entries = [h for h in history if h.startswith("shield:secrets:read:")]
    assert len(shield_entries) >= 1, "Shield secret fetch must be in provenance"


@pytest.mark.asyncio
async def test_provenance_no_shield_when_disabled(_set_token):
    """Shield secrets paths are NOT recorded when shield is disabled."""
    secure = make_secure_node(
        "planner",
        _sync_node,
        {"model_secret_ref": "openai-key"},
        requires_llm=True,
    )

    state = {
        "state": "init",
        "metadata": {"project_config": {"services": {"shield": {"enabled": False}}}},
    }
    await secure(state, {})

    history = get_accumulated_provenance()
    shield_entries = [h for h in history if h.startswith("shield:secrets:read:")]
    assert len(shield_entries) == 0, "Shield provenance must not appear when disabled"


# ── 7. Sync/Async Parity ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_node_returns_result(_set_token):
    """Synchronous node function wrapped in secure_node returns correct result."""
    secure = make_secure_node("sync_test", _sync_node, requires_llm=False)
    result = await secure({"state": "init"}, {})
    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_async_node_returns_result(_set_token):
    """Async node function wrapped in secure_node returns correct result."""
    secure = make_secure_node("async_test", _async_node, requires_llm=False)
    result = await secure({"state": "init"}, {})
    assert result["result"] == "async_ok"


# ── 8. Execute Tools Scope Injection ──────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_tools_grants_tool_scopes(_set_token):
    """execute_tools=['medical_sql'] adds tool:medical_sql:execute + bare tool:medical_sql."""
    secure = make_secure_node(
        "tool_node",
        _sync_node,
        requires_llm=False,
        execute_tools=["medical_sql"],
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert captured.has_permission("tool:medical_sql:execute")
    assert captured.has_permission("tool:medical_sql")


# ── 9. Service Scopes Injection ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_service_scopes_added_to_token(_set_token):
    """service_scopes are added to the attenuated token."""
    secure = make_secure_node(
        "svc_node",
        _sync_node,
        requires_llm=False,
        service_scopes=["brain:search"],
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert captured.has_permission("brain:search")


# ── 10. Token Restored After Execution ────────────────────────────────────


@pytest.mark.asyncio
async def test_token_restored_after_execution(_set_token, root_token):
    """After node execution, the original root token is restored in context."""
    secure = make_secure_node("restore_test", _sync_node, requires_llm=False)

    await secure({"state": "init"}, {})

    # After execution, current token must be the root again
    current = get_current_access_token()
    assert current is not None
    assert current.token_id == root_token.token_id


@pytest.mark.asyncio
async def test_token_restored_even_on_error(_set_token, root_token):
    """Token is restored even when the inner node raises an exception."""

    def _failing_node(state, config):
        raise RuntimeError("boom")

    secure = make_secure_node("crash_test", _failing_node, requires_llm=False)

    with pytest.raises(RuntimeError):
        await secure({"state": "init"}, {})

    current = get_current_access_token()
    assert current is not None
    assert current.token_id == root_token.token_id


# ── 11. Default Parameter Mutation Killers ────────────────────────────────


@pytest.mark.asyncio
async def test_default_requires_llm_grants_shield_scopes(_set_token):
    """Default requires_llm=True (no explicit kwarg) grants shield:secrets:read when secret_ref is provided.

    Kills mutant: requires_llm: bool = True → False.
    """
    secure = make_secure_node(
        "default_llm_node",
        _sync_node,
        {"model_secret_ref": "some-key"},
        # NOTE: requires_llm NOT passed — uses default True
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    assert captured.has_permission("shield:secrets:read"), (
        "Default requires_llm=True must grant shield scopes when model_secret_ref is set"
    )


@pytest.mark.asyncio
async def test_default_pass_through_false_does_attenuation(_set_token, root_token):
    """Default pass_through_token=False (no explicit kwarg) does capability stripping.

    Kills mutant: pass_through_token: bool = False → True.
    """
    secure = make_secure_node(
        "stripped_node",
        _sync_node,
        requires_llm=False,
        service_scopes=["brain:search"],
        # NOTE: pass_through_token NOT passed — uses default False
    )

    result = await secure({"state": "init"}, {})

    captured: ContextToken = result["__captured_token__"]
    # Attenuated token should have ONLY brain:search, not the full root set
    assert captured.has_permission("brain:search")
    assert not captured.has_permission("router:execute"), (
        "Default pass_through=False must strip permissions not requested by node"
    )


# ── 12. Provenance Flag Inversion ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_provenance_privacy_pii_applied(_set_token):
    """privacy:pii_applied provenance recorded when PII masking is active.

    PII masking is always-on by default in Router privacy.
    """
    from unittest.mock import patch

    secure = make_secure_node(
        "pii_prov_node",
        _sync_node,
        {"pii_masking": True},
        requires_llm=False,
    )

    state = {
        "state": "init",
        "metadata": {"project_config": {}},
    }

    with (
        patch("contextunity.router.cortex.secure_node.dispatch_custom_event"),
        patch("contextunity.router.cortex.utils.pii.PiiSession._create_anonymizer"),
    ):
        await secure(state, {})

    history = get_accumulated_provenance()
    assert "privacy:pii_applied" in history, (
        "PII provenance must always be recorded when pii_masking=True"
    )


@pytest.mark.asyncio
async def test_provenance_no_pii_when_explicitly_disabled(_set_token):
    """privacy:pii_applied NOT recorded when pii_masking explicitly False."""
    secure = make_secure_node(
        "no_pii_prov",
        _sync_node,
        {"pii_masking": False},
        requires_llm=False,
    )

    state = {
        "state": "init",
        "metadata": {"project_config": {}},
    }

    await secure(state, {})

    history = get_accumulated_provenance()
    assert "privacy:pii_applied" not in history, (
        "PII provenance must NOT appear when pii_masking is explicitly disabled"
    )


# ── 13. Prompt Integrity Key Material ─────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_integrity_without_project_secret_fails_closed(_set_token, monkeypatch):
    """Signed prompt must not be verified with a global Router fallback secret."""
    monkeypatch.setattr(
        "contextunity.core.discovery.get_project_key",
        lambda project_id: {},
    )

    secure = make_secure_node("planner", _sync_node, requires_llm=False)
    state = {
        "state": "init",
        "metadata": {
            "project_config": {
                "project_id": "test-project",
                "allowed_tenants": ["default"],
                "nodes": [
                    {
                        "name": "planner",
                        "prompt_signature": "test-project:hmac-001.payload.signature",
                    }
                ],
                "graph": {
                    "default": {
                        "config": {"planner_prompt": "Use verified prompt only."},
                        "nodes": [],
                    }
                },
            }
        },
    }

    with pytest.raises(TamperDetectedError, match="project-scoped HMAC key material"):
        await secure(state, {})


@pytest.mark.asyncio
async def test_prompt_integrity_stripped_signature_fails_closed(_set_token, monkeypatch):
    """A node that ships an LLM prompt but no signature is treated as tampering.

    Regression for WS-9: an attacker must not be able to bypass prompt-integrity
    verification simply by removing ``prompt_signature`` from the manifest while
    keeping (or injecting) the prompt text.
    """
    # A project key IS available — proving the failure is about the *missing
    # signature*, not missing key material.
    monkeypatch.setattr(
        "contextunity.core.discovery.get_project_key",
        lambda project_id: {"project_secret": "s3cr3t"},
    )

    secure = make_secure_node("planner", _sync_node, requires_llm=False)
    state = {
        "state": "init",
        "metadata": {
            "project_config": {
                "project_id": "test-project",
                "allowed_tenants": ["default"],
                # Node carries NO prompt_signature …
                "nodes": [{"name": "planner"}],
                "graph": {
                    "default": {
                        # … but a prompt is present (possibly injected).
                        "config": {"planner_prompt": "Injected unsigned prompt."},
                        "nodes": [],
                    }
                },
            }
        },
    }

    with pytest.raises(TamperDetectedError, match="no integrity signature"):
        await secure(state, {})


@pytest.mark.asyncio
async def test_prompt_integrity_no_prompt_is_skipped(_set_token, monkeypatch):
    """A node without any LLM prompt has nothing to verify and must not raise."""
    monkeypatch.setattr(
        "contextunity.core.discovery.get_project_key",
        lambda project_id: {},
    )

    secure = make_secure_node("planner", _sync_node, requires_llm=False)
    state = {
        "state": "init",
        "metadata": {
            "project_config": {
                "project_id": "test-project",
                "allowed_tenants": ["default"],
                "nodes": [{"name": "planner"}],
                "graph": {"default": {"config": {}, "nodes": []}},
            }
        },
    }

    # No prompt → no signature required → no TamperDetectedError.
    await secure(state, {})
