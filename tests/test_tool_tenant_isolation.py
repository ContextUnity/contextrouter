import pytest
from contextcore import ContextToken

from contextrouter.cortex.runtime_context import (
    reset_current_access_token,
    set_current_access_token,
)
from contextrouter.modules.tools.secure import SecureTool
from contextrouter.modules.tools.security_tools import check_policy


@pytest.mark.asyncio
async def test_tool_tenant_isolation_no_token():
    """Test that tool fails closed when no active access token is present."""
    # Ensure no token is set in runtime context
    token_ref = set_current_access_token(None)
    secure_check_policy = SecureTool.wrap(check_policy)
    try:
        # PII checking, security, etc should fail closed when no token present
        # the LLM might try to provide tenant_id="admin"
        try:
            await secure_check_policy.ainvoke(
                {"action": "read", "resource": "brain", "tenant_id": "admin"}
            )
            assert False, "Expected PermissionError from SecureTool wrapper"
        except PermissionError as e:
            assert "No access token" in str(e)
    finally:
        reset_current_access_token(token_ref)


@pytest.mark.asyncio
async def test_tool_tenant_isolation_overrides_llm():
    """Test that tool overwrites LLM-provided tenant_id with authoritative token."""
    # Create fake token for non-admin tenant
    token = ContextToken(
        token_id="user-123",
        permissions=("brain:read", "tool:check_policy"),
        allowed_tenants=("limited_tenant",),
    )

    token_ref = set_current_access_token(token)
    try:
        secure_check_policy = SecureTool.wrap(check_policy)
        # LLM maliciously tries to check policy for "target_tenant"
        # for a completely unrelated resource (needs secret:write or admin:*)
        result = await secure_check_policy.ainvoke(
            {
                "action": "write",
                "resource": "secret",
                "tenant_id": "target_tenant",
                "permissions": ["admin:all"],  # Malicious permissions spoofing
            }
        )

        # The key assertion is that the local evaluation in check_policy ran
        # using the REAL token / tenant. The real token doesn't have secret:write or admin:*,
        # so it should be denied. If the spoofing worked, it would be allowed, and test would fail.
        assert isinstance(result, dict)
        assert result.get("allowed") is False

    finally:
        reset_current_access_token(token_ref)
