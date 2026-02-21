import pytest
from contextcore import ContextToken

from contextrouter.cortex.runtime_context import (
    reset_current_access_token,
    set_current_access_token,
)
from contextrouter.modules.tools.security_tools import check_policy


@pytest.mark.asyncio
async def test_tool_tenant_isolation_no_token():
    """Test that tool fails closed when no active access token is present."""
    # Ensure no token is set in runtime context
    token_ref = set_current_access_token(None)
    try:
        # PII checking, security, etc should fail closed when no token present
        # the LLM might try to provide tenant_id="admin"
        result = await check_policy.ainvoke(
            {"action": "read", "resource": "brain", "tenant_id": "admin"}
        )

        # Must fail closed instead of evaluating with tenant_id="admin"
        assert isinstance(result, dict)
        assert result.get("allowed") is False
        assert "No active access token" in result.get("reason", "")
    finally:
        reset_current_access_token(token_ref)


@pytest.mark.asyncio
async def test_tool_tenant_isolation_overrides_llm():
    """Test that tool overwrites LLM-provided tenant_id with authoritative token."""
    # Create fake token for non-admin tenant
    token = ContextToken(
        token_id="user-123",
        permissions=("brain:read",),
        allowed_tenants=("limited_tenant",),
    )

    token_ref = set_current_access_token(token)
    try:
        # LLM maliciously tries to check policy for "target_tenant"
        # for a completely unrelated resource (needs secret:write or admin:*)
        result = await check_policy.ainvoke(
            {
                "action": "write",
                "resource": "secret",
                "tenant_id": "target_tenant",
                "permissions": ["admin:all"],  # Malicious permissions spoofing
            }
        )

        # check_policy in local mode evaluates rules
        # Let's just make sure it didn't crash and was evaluated
        assert isinstance(result, dict)

        # The key assertion is that the local evaluation in check_policy ran
        # using the real token / tenant, so it didn't crash but returned standard response.
        # Check logs/result or the fact that it didn't permit based on "admin:all"
        assert (
            result.get("allowed") is False
        )  # limited_tenant with "brain:read" doesn't have "admin:*"

    finally:
        reset_current_access_token(token_ref)
