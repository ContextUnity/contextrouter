"""Behavioral tests for dispatcher_agent.nodes.security — security_guard_node.

Tests cover:
  - No tool calls → empty return
  - Unknown tool → blocked with tool_not_found
  - Denied tool → blocked with tool_denied
  - Token permission denied → blocked with permission_denied
  - State-level whitelist enforcement (with * wildcard)
  - Allowed tool passes through
  - Multiple tool calls with mixed blocking
  - Security flags accumulation
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from contextunity.core.tokens import ContextToken

from contextunity.router.cortex.dispatcher_agent.nodes.security import security_guard_node

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _make_message(tool_calls: list[dict] | None = None) -> SimpleNamespace:
    msg = SimpleNamespace()
    msg.tool_calls = tool_calls or []
    return msg


def _make_state(
    tool_calls: list[dict] | None = None,
    *,
    token: ContextToken | None = None,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    security_flags: list | None = None,
) -> dict:
    state = {
        "messages": [_make_message(tool_calls)],
        "allowed_tools": ["*"] if allowed_tools is None else allowed_tools,
        "denied_tools": denied_tools or [],
    }
    if token is not None:
        state["access_token"] = token
    if security_flags is not None:
        state["security_flags"] = security_flags
    return state


AVAILABLE_TOOLS = [_make_tool("search"), _make_tool("calculator"), _make_tool("sql_query")]


# ── Tests ─────────────────────────────────────────────────────────────────


class TestSecurityGuardNode:
    """Behavioral tests for security_guard_node."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty(self):
        """Message without tool_calls returns empty dict."""
        state = _make_state(tool_calls=[])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_unknown_tool_is_blocked(self):
        """Tool not in available_tools is blocked with tool_not_found."""
        state = _make_state([{"name": "nonexistent_tool", "id": "call-1"}])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "not available" in result["messages"][0].content
        flags = result.get("security_flags", [])
        assert any(f["event"] == "tool_not_found" for f in flags)

    @pytest.mark.asyncio
    async def test_denied_tool_is_blocked(self):
        """Tool in denied_tools list is blocked with tool_denied."""
        state = _make_state(
            [{"name": "search", "id": "call-1"}],
            denied_tools=["search"],
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert "messages" in result
        assert "denied" in result["messages"][0].content.lower()
        flags = result.get("security_flags", [])
        assert any(f["event"] == "tool_denied" for f in flags)

    @pytest.mark.asyncio
    async def test_token_permission_denied(self):
        """Tool denied by token permissions is blocked."""
        token = ContextToken(
            token_id="test-guard",
            user_id="user-1",
            permissions=("tool:calculator",),  # No tool:search
        )
        state = _make_state(
            [{"name": "search", "id": "call-1"}],
            token=token,
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert "messages" in result
        flags = result.get("security_flags", [])
        assert any(f["event"] == "permission_denied" for f in flags)

    @pytest.mark.asyncio
    async def test_token_grants_access(self):
        """Tool allowed by token permissions passes through."""
        token = ContextToken(
            token_id="test-guard",
            user_id="user-1",
            permissions=("tool:search", "tool:search:execute"),
        )
        state = _make_state(
            [{"name": "search", "id": "call-1"}],
            token=token,
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        # No blocked messages — tool was allowed
        assert "messages" not in result or not result.get("messages")

    @pytest.mark.asyncio
    async def test_whitelist_blocks_unlisted_tool(self):
        """State-level allowed_tools blocks tools not in list (no token)."""
        state = _make_state(
            [{"name": "search", "id": "call-1"}],
            allowed_tools=["calculator"],
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert "messages" in result
        flags = result.get("security_flags", [])
        assert any(f["event"] == "permission_denied" for f in flags)

    @pytest.mark.asyncio
    async def test_whitelist_wildcard_allows_all(self):
        """allowed_tools=['*'] allows all tools when token grants access."""
        token = ContextToken(
            token_id="test-guard",
            user_id="user-1",
            permissions=("tool:search", "tool:search:execute"),
        )
        state = _make_state(
            [{"name": "search", "id": "call-1"}],
            allowed_tools=["*"],
            token=token,
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert "messages" not in result or not result.get("messages")

    @pytest.mark.asyncio
    async def test_multiple_calls_partial_blocking(self):
        """Mixed tool calls: one allowed, one denied."""
        token = ContextToken(
            token_id="test-guard",
            user_id="user-1",
            permissions=("tool:search", "tool:search:execute"),
        )
        state = _make_state(
            [
                {"name": "search", "id": "call-1"},
                {"name": "nonexistent", "id": "call-2"},
            ],
            token=token,
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        # Only nonexistent is blocked
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "nonexistent" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_security_flags_accumulate(self):
        """Existing security_flags are preserved and new ones appended."""
        existing_flag = {"event": "previous_event", "tool": "old"}
        state = _make_state(
            [{"name": "unknown_tool", "id": "call-1"}],
            security_flags=[existing_flag],
        )
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        flags = result.get("security_flags", [])
        assert len(flags) >= 2
        assert existing_flag in flags

    @pytest.mark.asyncio
    async def test_tool_call_without_name_blocked(self):
        """Tool call with no 'name' or 'tool' key is blocked fail-closed."""
        state = _make_state([{"id": "call-no-name"}])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})
        assert "messages" in result
        assert any("unknown" in msg.content for msg in result["messages"])

    @pytest.mark.asyncio
    async def test_tool_call_uses_tool_key_fallback(self):
        """Tool call with 'tool' key (no 'name') is resolved correctly."""
        state = _make_state([{"tool": "nonexistent_tool", "id": "call-fb"}])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert "messages" in result
        flags = result.get("security_flags", [])
        assert any(f["event"] == "tool_not_found" for f in flags)

    @pytest.mark.asyncio
    async def test_blocked_call_has_tool_call_id(self):
        """Blocked tool calls carry the original tool_call_id for ToolMessage routing."""
        state = _make_state([{"name": "unknown_tool", "id": "call-42"}])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        assert result["messages"][0].tool_call_id == "call-42"

    @pytest.mark.asyncio
    async def test_security_flags_have_tool_key(self):
        """Security flags include 'tool' key with the tool name."""
        state = _make_state([{"name": "phantom", "id": "call-x"}])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        flags = result.get("security_flags", [])
        assert len(flags) >= 1
        assert flags[-1]["tool"] == "phantom"

    @pytest.mark.asyncio
    async def test_no_token_blocks_known_tool(self):
        """Known tool without token is blocked (token required for execution)."""
        state = _make_state([{"name": "search", "id": "call-1"}])
        with patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=AVAILABLE_TOOLS,
        ):
            result = await security_guard_node(state, {})

        blocked = result.get("messages", [])
        assert len(blocked) >= 1
        flags = result.get("security_flags", [])
        assert any(
            f.get("event") == "permission_denied" and f.get("tool") == "search" for f in flags
        )
