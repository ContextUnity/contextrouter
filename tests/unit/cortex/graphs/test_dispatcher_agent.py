"""Unit tests for dispatcher_agent graph modules.

Tests routing decisions, state shape, and builder construction
after the monolith→modular refactoring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# DispatcherState — shape & defaults
# ---------------------------------------------------------------------------


class TestDispatcherState:
    """Verify DispatcherState TypedDict has all required keys."""

    def test_state_has_required_keys(self):
        from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState

        annotations = DispatcherState.__annotations__
        required = {
            "messages",
            "tenant_id",
            "session_id",
            "platform",
            "metadata",
            "iteration",
            "max_iterations",
            "allowed_tools",
            "denied_tools",
            "access_token",
            "trace_id",
            "_start_ts",
            "security_flags",
            "hitl_approved",
            "error_detected",
            "healing_triggered",
        }
        assert required.issubset(annotations.keys()), (
            f"Missing keys: {required - annotations.keys()}"
        )

    def test_messages_uses_add_messages_reducer(self):
        """Ensure messages field uses langgraph add_messages annotation."""
        from typing import get_type_hints

        from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState

        hints = get_type_hints(DispatcherState, include_extras=True)
        messages_hint = hints["messages"]
        # Annotated type should contain add_messages metadata
        assert hasattr(messages_hint, "__metadata__"), "messages should be Annotated"


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


class TestShouldExecuteTools:
    """Tests for should_execute_tools routing function."""

    def _make_state(self, last_message) -> dict:
        return {"messages": [last_message], "iteration": 1}

    def test_returns_execute_when_tool_calls_present(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_execute_tools

        msg = MagicMock()
        msg.tool_calls = [{"name": "search", "id": "1"}]
        msg.content = "Using search tool"
        state = self._make_state(msg)

        assert should_execute_tools(state) == "execute"

    def test_returns_blocked_on_security_violation(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_execute_tools

        msg = MagicMock()
        msg.tool_calls = []
        msg.content = "Security Violation: Access to tool 'dangerous' is denied."
        state = self._make_state(msg)

        assert should_execute_tools(state) == "blocked"

    def test_returns_blocked_on_tool_error(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_execute_tools

        msg = MagicMock()
        msg.tool_calls = []
        msg.content = "Error: Tool 'missing_tool' is not available in the system."
        state = self._make_state(msg)

        assert should_execute_tools(state) == "blocked"

    def test_returns_end_when_no_tool_calls(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_execute_tools

        msg = MagicMock()
        msg.tool_calls = []
        msg.content = "Here is your answer."
        state = self._make_state(msg)

        assert should_execute_tools(state) == "end"

    def test_returns_end_for_plain_message(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_execute_tools

        msg = MagicMock(spec=[])  # no tool_calls attr
        msg.content = "Plain text response"
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

    def test_returns_tools_when_tool_calls(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_continue

        msg = MagicMock()
        msg.tool_calls = [{"name": "brain_search", "id": "1"}]
        msg.content = ""
        state = self._make_state(msg)

        assert should_continue(state) == "tools"

    def test_returns_end_when_max_iterations_reached(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_continue

        msg = MagicMock()
        msg.tool_calls = [{"name": "search", "id": "1"}]
        state = self._make_state(msg, iteration=10, max_iterations=10)

        assert should_continue(state) == "end"

    def test_returns_end_when_no_tool_calls(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import should_continue

        msg = MagicMock()
        msg.tool_calls = []
        msg.content = "Here is the final answer."
        state = self._make_state(msg)

        assert should_continue(state) == "end"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    """Verify system prompt is well-formed."""

    def test_system_prompt_not_empty(self):
        from contextrouter.cortex.graphs.dispatcher_agent.prompts import SYSTEM_PROMPT

        assert len(SYSTEM_PROMPT) > 100, "System prompt should be substantial"

    def test_system_prompt_has_tool_section(self):
        from contextrouter.cortex.graphs.dispatcher_agent.prompts import SYSTEM_PROMPT

        assert "Available Tools" in SYSTEM_PROMPT

    def test_system_prompt_has_security_section(self):
        from contextrouter.cortex.graphs.dispatcher_agent.prompts import SYSTEM_PROMPT

        assert "Security" in SYSTEM_PROMPT or "Governance" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Builder — structure tests (no LLM/tool calls)
# ---------------------------------------------------------------------------


class TestBuilderStructure:
    """Test that build_dispatcher_graph returns a valid StateGraph."""

    @patch("contextrouter.cortex.graphs.dispatcher_agent.builder.discover_all_tools")
    def test_build_returns_state_graph(self, mock_tools):
        from langgraph.graph import StateGraph

        from contextrouter.cortex.graphs.dispatcher_agent.builder import build_dispatcher_graph

        mock_tools.return_value = []
        graph = build_dispatcher_graph()
        assert isinstance(graph, StateGraph)

    @patch("contextrouter.cortex.graphs.dispatcher_agent.builder.discover_all_tools")
    def test_build_graph_has_required_nodes(self, mock_tools):
        from contextrouter.cortex.graphs.dispatcher_agent.builder import build_dispatcher_graph

        mock_tools.return_value = []
        graph = build_dispatcher_graph()
        node_names = set(graph.nodes.keys())
        assert "agent" in node_names
        assert "security" in node_names
        assert "reflect" in node_names

    @patch("contextrouter.cortex.graphs.dispatcher_agent.builder.discover_all_tools")
    def test_build_graph_with_tools_has_tools_node(self, mock_tools):
        from langchain_core.tools import StructuredTool

        from contextrouter.cortex.graphs.dispatcher_agent.builder import build_dispatcher_graph

        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="test_tool",
            description="A test tool",
        )
        mock_tools.return_value = [tool]

        graph = build_dispatcher_graph()
        assert "tools" in graph.nodes

    @patch("contextrouter.cortex.graphs.dispatcher_agent.builder.discover_all_tools")
    def test_compile_returns_runnable(self, mock_tools):
        from contextrouter.cortex.graphs.dispatcher_agent.builder import (
            compile_dispatcher_graph,
        )

        mock_tools.return_value = []
        compiled = compile_dispatcher_graph()
        assert hasattr(compiled, "ainvoke") or hasattr(compiled, "invoke")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestExports:
    """Ensure all __all__ exports are properly defined."""

    def test_nodes_init_exports(self):
        from contextrouter.cortex.graphs.dispatcher_agent import nodes

        assert hasattr(nodes, "agent_node")
        assert hasattr(nodes, "security_guard_node")
        assert hasattr(nodes, "reflect_dispatcher")

    def test_state_exports(self):
        from contextrouter.cortex.graphs.dispatcher_agent.state import __all__

        assert "DispatcherState" in __all__

    def test_routing_exports(self):
        from contextrouter.cortex.graphs.dispatcher_agent.routing import __all__

        assert "should_execute_tools" in __all__
        assert "should_continue" in __all__

    def test_prompts_exports(self):
        from contextrouter.cortex.graphs.dispatcher_agent.prompts import __all__

        assert "SYSTEM_PROMPT" in __all__

    def test_builder_exports(self):
        from contextrouter.cortex.graphs.dispatcher_agent.builder import __all__

        assert "build_dispatcher_graph" in __all__
        assert "compile_dispatcher_graph" in __all__
