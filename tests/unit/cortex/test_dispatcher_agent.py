"""Unit tests for dispatcher_agent graph modules.

Tests routing decisions, state shape, and builder construction
after the monolith→modular refactoring.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# DispatcherState — shape & defaults
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


class TestShouldExecuteTools:
    """Tests for should_execute_tools routing function."""

    def _make_state(self, last_message) -> dict:
        return {"messages": [last_message], "iteration": 1}

    def test_returns_execute_when_tool_calls_present(self):
        from contextunity.router.cortex.dispatcher_agent.routing import should_execute_tools

        msg = SimpleNamespace(
            tool_calls=[{"name": "search", "id": "1"}], content="Using search tool"
        )
        state = self._make_state(msg)

        assert should_execute_tools(state) == "execute"

    def test_returns_blocked_on_security_violation(self):
        from contextunity.router.cortex.dispatcher_agent.routing import should_execute_tools

        msg = SimpleNamespace(
            tool_calls=[], content="Security Violation: Access to tool 'dangerous' is denied."
        )
        state = self._make_state(msg)

        assert should_execute_tools(state) == "blocked"

    def test_returns_blocked_on_tool_error(self):
        from contextunity.router.cortex.dispatcher_agent.routing import should_execute_tools

        msg = SimpleNamespace(
            tool_calls=[], content="Error: Tool 'missing_tool' is not available in the system."
        )
        state = self._make_state(msg)

        assert should_execute_tools(state) == "blocked"

    def test_returns_end_when_no_tool_calls(self):
        from contextunity.router.cortex.dispatcher_agent.routing import should_execute_tools

        msg = SimpleNamespace(tool_calls=[], content="Here is your answer.")
        state = self._make_state(msg)

        assert should_execute_tools(state) == "end"


class TestShouldContinue:
    """Tests for should_continue routing function."""

    def _make_state(self, last_message, iteration=1, max_iterations=10) -> dict:
        return {
            "messages": [last_message],
            "iteration": iteration,
            "max_iterations": max_iterations,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Builder — structure tests (no LLM/tool calls)
# ---------------------------------------------------------------------------


class TestBuilderStructure:
    """Test that build_dispatcher_graph returns a valid StateGraph."""

    @patch("contextunity.router.modules.tools.discover_all_tools")
    @patch("contextunity.router.cortex.secure_node.make_secure_node")
    def test_agent_node_does_not_grant_all_tool_scopes(self, mock_secure, mock_tools):
        """Built-in dispatcher agent must not attenuate to every discovered tool."""
        from langchain_core.tools import StructuredTool

        from contextunity.router.cortex.dispatcher_agent.builder import build_dispatcher_graph

        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="secret_tool",
            description="A test tool",
        )
        mock_tools.return_value = [tool]
        mock_secure.side_effect = lambda name, fn, spec=None, **kwargs: fn

        build_dispatcher_graph()

        agent_call = next(call for call in mock_secure.call_args_list if call.args[0] == "agent")
        assert agent_call.kwargs.get("execute_tools") in (None, [])

    @patch("contextunity.router.modules.tools.discover_all_tools")
    def test_build_returns_state_graph(self, mock_tools):
        from langgraph.graph import StateGraph

        from contextunity.router.cortex.dispatcher_agent.builder import (
            build_dispatcher_graph,
        )

        mock_tools.return_value = []
        graph = build_dispatcher_graph()
        assert isinstance(graph, StateGraph)

    @patch("contextunity.router.modules.tools.discover_all_tools")
    def test_build_graph_has_required_nodes(self, mock_tools):
        from contextunity.router.cortex.dispatcher_agent.builder import (
            build_dispatcher_graph,
        )

        mock_tools.return_value = []
        graph = build_dispatcher_graph()
        node_names = set(graph.nodes.keys())
        assert "agent" in node_names
        assert "security" in node_names
        assert "reflect" in node_names

    @patch("contextunity.router.modules.tools.discover_all_tools")
    def test_build_graph_with_tools_has_tools_node(self, mock_tools):
        from langchain_core.tools import StructuredTool

        from contextunity.router.cortex.dispatcher_agent.builder import (
            build_dispatcher_graph,
        )

        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="test_tool",
            description="A test tool",
        )
        mock_tools.return_value = [tool]

        graph = build_dispatcher_graph()
        assert "tools" in graph.nodes

    @patch("contextunity.router.modules.tools.discover_all_tools")
    def test_compile_returns_runnable(self, mock_tools):
        from contextunity.router.cortex.dispatcher_agent.builder import (
            compile_dispatcher_graph,
        )

        mock_tools.return_value = []
        compiled = compile_dispatcher_graph()
        assert hasattr(compiled, "ainvoke") or hasattr(compiled, "invoke")


class TestValidatedDispatcherStreamEvent:
    def test_partial_node_update_without_hitl_approved(self):
        """Security node partial updates omit hitl_approved; stream must still forward."""
        from contextunity.router.cortex.dispatcher_agent.types import (
            validated_dispatcher_stream_event,
        )

        event = {
            "security": {
                "messages": [],
                "security_flags": [{"event": "permission_denied", "tool": "search"}],
            }
        }
        result = validated_dispatcher_stream_event(event)
        assert result == event

    def test_rejects_wrong_hitl_approved_type(self):
        from contextunity.core.exceptions import ConfigurationError

        from contextunity.router.cortex.dispatcher_agent.types import (
            validated_dispatcher_stream_event,
        )

        with pytest.raises(ConfigurationError, match="hitl_approved"):
            validated_dispatcher_stream_event(
                {"hitl_approved": "yes", "messages": [], "security_flags": []}
            )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
