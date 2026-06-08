"""Tests for runtime tenant narrowing in secure_node and SecureTool.

Note: A naive test that subclasses ``SecureTool`` and calls ``tool._run()`` directly
does **not** exercise SecureTool's attenuation path — Python MRO bypasses
``SecureTool._run`` and goes straight to the subclass override. The contract
under test only fires when the tool is invoked via the public ``invoke()``
path (i.e. as a real ``SecureTool`` instance wrapping a raw ``BaseTool``).
"""

from __future__ import annotations

import pytest
from contextunity.core.tokens import ContextToken, TokenBuilder
from langchain_core.tools import BaseTool

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.modules.tools.secure import SecureTool


class _InnerEcho(BaseTool):
    name: str = "inner_echo"
    description: str = "echo"

    def _run(self, **kwargs: object) -> str:
        from contextunity.router.core.context import get_current_access_token

        token = get_current_access_token()
        assert token is not None
        return ",".join(token.allowed_tenants)


@pytest.fixture
def parent_token() -> ContextToken:
    return ContextToken(
        token_id="t1",
        permissions=("tool:inner_echo:execute", "tool:inner_echo"),
        allowed_tenants=("nszu", "nszu-staging"),
    )


def test_secure_tool_narrows_token_to_bound_tenants(parent_token: ContextToken) -> None:
    inner = _InnerEcho()
    tool = SecureTool.wrap(
        inner,
        permission="tool:inner_echo",
        allowed_tenants=("nszu-staging",),
    )
    token_ref = set_current_access_token(parent_token)
    try:
        result = tool.invoke({})
    finally:
        reset_current_access_token(token_ref)
    assert result == "nszu-staging"


def test_secure_tool_intersects_bound_with_already_narrowed_context(
    parent_token: ContextToken,
) -> None:
    narrowed = TokenBuilder().attenuate(
        parent_token,
        allowed_tenants=("nszu-staging",),
        agent_id="node:agent",
    )
    inner = _InnerEcho()
    tool = SecureTool.wrap(
        inner,
        permission="tool:inner_echo",
        allowed_tenants=("nszu", "nszu-staging"),
    )
    token_ref = set_current_access_token(narrowed)
    try:
        result = tool.invoke({})
    finally:
        reset_current_access_token(token_ref)
    assert result == "nszu-staging"
