"""Test _trace node injection and prompt resolver (Phase 4.C + 4.D).

RED phase: Tests define contracts for:
- _trace platform-bound tool node auto-injection before __end__
- Prompt resolver with ctx_ prefix convention
"""

import pytest
from contextunity.core.exceptions import ConfigurationError

# ── 4.12-4.13: _trace Node Injection ──────────────────────────────


class TestTraceInjection:
    """_trace platform-bound tool node auto-injected before __end__ in all compiled graphs."""

    def test_trace_injected_before_end(self):
        """4.12: inject_trace_node rewires last→__end__ to last→_trace→__end__."""
        from contextunity.router.cortex.compiler.trace_injection import (
            inject_trace_node,
        )

        nodes = [
            {"name": "step_a", "type": "llm"},
            {"name": "step_b", "type": "llm"},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "step_a"},
            {"from_node": "step_a", "to_node": "step_b"},
            {"from_node": "step_b", "to_node": "__end__"},
        ]
        new_nodes, new_edges = inject_trace_node(nodes, edges, {})

        # _trace node added
        node_names = {n["name"] for n in new_nodes}
        assert "_trace" in node_names

        # _trace is platform-bound tool node calling brain_upsert
        trace_node = next(n for n in new_nodes if n["name"] == "_trace")
        assert trace_node["type"] == "tool"
        assert trace_node["tool_binding"] == "brain_upsert"

    def test_trace_edge_rewiring(self):
        """Last node → __end__ becomes last → _trace → __end__."""
        from contextunity.router.cortex.compiler.trace_injection import (
            inject_trace_node,
        )

        nodes = [{"name": "step_a", "type": "llm"}]
        edges = [
            {"from_node": "__start__", "to_node": "step_a"},
            {"from_node": "step_a", "to_node": "__end__"},
        ]
        new_nodes, new_edges = inject_trace_node(nodes, edges, {})

        # step_a → _trace edge exists
        assert any(
            e.get("from_node") == "step_a" and e.get("to_node") == "_trace" for e in new_edges
        )
        # _trace → __end__ edge exists
        assert any(
            e.get("from_node") == "_trace" and e.get("to_node") == "__end__" for e in new_edges
        )
        # original step_a → __end__ removed
        assert not any(
            e.get("from_node") == "step_a" and e.get("to_node") == "__end__" for e in new_edges
        )

    def test_trace_not_duplicated(self):
        """Calling inject_trace_node twice doesn't add duplicate _trace."""
        from contextunity.router.cortex.compiler.trace_injection import (
            inject_trace_node,
        )

        nodes = [{"name": "step_a", "type": "llm"}]
        edges = [
            {"from_node": "__start__", "to_node": "step_a"},
            {"from_node": "step_a", "to_node": "__end__"},
        ]
        n1, e1 = inject_trace_node(nodes, edges, {})
        n2, e2 = inject_trace_node(n1, e1, {})

        trace_count = sum(1 for n in n2 if n["name"] == "_trace")
        assert trace_count == 1

    def test_trace_with_conditional_end_edges(self):
        """Conditional edges to __end__ should also get _trace injection."""
        from contextunity.router.cortex.compiler.trace_injection import (
            inject_trace_node,
        )

        nodes = [
            {"name": "router_node", "type": "llm"},
            {"name": "path_a", "type": "llm"},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "router_node"},
            {
                "from_node": "router_node",
                "condition_key": "route",
                "condition_map": {"go": "path_a", "done": "__end__"},
            },
            {"from_node": "path_a", "to_node": "__end__"},
        ]
        new_nodes, new_edges = inject_trace_node(nodes, edges, {})
        node_names = {n["name"] for n in new_nodes}
        assert "_trace" in node_names


# ── 4.18-4.20: Prompt Resolver ────────────────────────────────────


class TestPromptResolver:
    """resolve_prompt_ref() loads prompt text and applies ctx_ parameters."""

    def test_resolve_known_prompt_returns_string(self):
        """4.18: resolve_prompt_ref('rag_intent') → non-empty prompt string."""
        from contextunity.router.cortex.compiler.prompt_resolver import (
            resolve_prompt_ref,
        )

        result = resolve_prompt_ref("rag_intent")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resolve_missing_prompt_raises(self):
        """4.20a: Missing prompt_ref → ConfigurationError."""
        from contextunity.router.cortex.compiler.prompt_resolver import (
            resolve_prompt_ref,
        )

        with pytest.raises(ConfigurationError, match="not_a_real_prompt"):
            resolve_prompt_ref("not_a_real_prompt")

    def test_resolve_with_ctx_variables(self):
        """4.19: ctx_ variables are substituted in prompt text."""
        from contextunity.router.cortex.compiler.prompt_resolver import (
            register_prompt,
            resolve_prompt_ref,
        )

        register_prompt(
            "test_parametric",
            "Hello {ctx_user_name}, your query is: {ctx_query}",
        )
        result = resolve_prompt_ref(
            "test_parametric",
            ctx_vars={"ctx_user_name": "Alice", "ctx_query": "weather"},
        )
        assert "Alice" in result
        assert "weather" in result

    def test_resolve_rejects_non_ctx_variables(self):
        """4.20b: Variables without ctx_ prefix → ConfigurationError."""
        from contextunity.router.cortex.compiler.prompt_resolver import (
            register_prompt,
            resolve_prompt_ref,
        )

        register_prompt("test_unsafe_var", "Secret: {password}")
        with pytest.raises(ConfigurationError, match="password"):
            resolve_prompt_ref("test_unsafe_var")

    def test_resolve_missing_ctx_variable_raises(self):
        """Required ctx_ variable not provided → ConfigurationError."""
        from contextunity.router.cortex.compiler.prompt_resolver import (
            register_prompt,
            resolve_prompt_ref,
        )

        register_prompt("test_missing_ctx", "Hello {ctx_name}")
        with pytest.raises(ConfigurationError, match="ctx_name"):
            resolve_prompt_ref("test_missing_ctx", ctx_vars={})
