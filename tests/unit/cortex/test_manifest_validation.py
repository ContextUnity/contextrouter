"""Tests for manifest security validation.

Zero-mock tests for the compile-time graph security gate:
- Reserved node names rejected
- Node count limits
- Node name format validation
- Edge integrity — dangling references
- Cycle detection with max_retries guard
- Service dependency validation
- Tool binding format validation
"""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import SecurityError

from contextunity.router.core.exceptions import RouterGraphBuilderError
from contextunity.router.cortex.compiler.validation import (
    MAX_NODES_DEFAULT,
    RESERVED_NODE_NAMES,
    validate_manifest_security,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _node(name: str, **kwargs) -> dict:
    return {"name": name, "type": "llm", **kwargs}


def _tool_node(name: str, tool_binding: str, **kwargs) -> dict:
    return {"name": name, "type": "tool", "tool_binding": tool_binding, **kwargs}


def _agent_node(name: str, tools: list[str] | None = None, **kwargs) -> dict:
    return {"name": name, "type": "agent", "tools": tools or [], **kwargs}


def _edge(from_node: str, to_node: str) -> dict:
    return {"from_node": from_node, "to_node": to_node}


def _cond_edge(from_node: str, condition_map: dict[str, str]) -> dict:
    return {"from_node": from_node, "condition_map": condition_map}


def _manifest(**kwargs) -> dict:
    return kwargs


# ═══════════════════════════════════════════════════════════════════
# Node count limits
# ═══════════════════════════════════════════════════════════════════


class TestNodeCountLimit:
    """Graph bomb prevention via max_nodes."""

    def test_within_limit_passes(self):
        nodes = [_node(f"node_{i}") for i in range(5)]
        validate_manifest_security(nodes, [], _manifest())

    def test_exactly_at_limit_passes(self):
        nodes = [_node(f"node_{i}") for i in range(MAX_NODES_DEFAULT)]
        validate_manifest_security(nodes, [], _manifest())

    def test_exceeds_limit_raises(self):
        nodes = [_node(f"node_{i}") for i in range(MAX_NODES_DEFAULT + 1)]
        with pytest.raises(RouterGraphBuilderError, match="exceeds max_nodes"):
            validate_manifest_security(nodes, [], _manifest())

    def test_custom_limit(self):
        nodes = [_node(f"node_{i}") for i in range(3)]
        with pytest.raises(RouterGraphBuilderError, match="exceeds max_nodes"):
            validate_manifest_security(nodes, [], _manifest(), max_nodes=2)


# ═══════════════════════════════════════════════════════════════════
# Reserved node names
# ═══════════════════════════════════════════════════════════════════


class TestReservedNames:
    """Reserved names prevent provenance forgery."""

    @pytest.mark.parametrize("name", sorted(RESERVED_NODE_NAMES)[:6])
    def test_reserved_name_rejected(self, name):
        with pytest.raises(SecurityError, match="reserved"):
            validate_manifest_security([_node(name)], [], _manifest())

    def test_reserved_name_case_insensitive(self):
        with pytest.raises(SecurityError, match="reserved"):
            validate_manifest_security([_node("System")], [], _manifest())

    def test_non_reserved_passes(self):
        validate_manifest_security([_node("my_agent")], [], _manifest())


# ═══════════════════════════════════════════════════════════════════
# Node name format
# ═══════════════════════════════════════════════════════════════════


class TestNodeNameFormat:
    """Node names must be lowercase alphanumeric + underscores."""

    def test_valid_name(self):
        validate_manifest_security([_node("my_agent_v2")], [], _manifest())

    def test_leading_underscore_valid(self):
        validate_manifest_security([_node("_private")], [], _manifest())

    def test_uppercase_rejected(self):
        with pytest.raises(RouterGraphBuilderError, match="Invalid node name"):
            validate_manifest_security([_node("MyAgent")], [], _manifest())

    def test_special_chars_rejected(self):
        with pytest.raises(RouterGraphBuilderError, match="Invalid node name"):
            validate_manifest_security([_node("my-agent")], [], _manifest())

    def test_empty_name_rejected(self):
        with pytest.raises(RouterGraphBuilderError, match="missing required 'name'"):
            validate_manifest_security([{"type": "llm"}], [], _manifest())

    def test_too_long_name_rejected(self):
        with pytest.raises(RouterGraphBuilderError, match="Invalid node name"):
            validate_manifest_security([_node("a" * 65)], [], _manifest())

    def test_max_length_accepted(self):
        validate_manifest_security([_node("a" * 64)], [], _manifest())

    def test_duplicate_name_rejected(self):
        with pytest.raises(RouterGraphBuilderError, match="Duplicate node name"):
            validate_manifest_security([_node("dup"), _node("dup")], [], _manifest())


# ═══════════════════════════════════════════════════════════════════
# Node type validation
# ═══════════════════════════════════════════════════════════════════


class TestNodeType:
    """Node types limited to llm, agent, tool."""

    def test_valid_types(self):
        nodes = [
            _node("a", type="llm"),
            _agent_node("b"),
            _tool_node("c", "platform:brain_search"),
        ]
        manifest = _manifest(services={"brain": {"enabled": True}})
        validate_manifest_security(nodes, [], manifest)

    def test_invalid_type_rejected(self):
        with pytest.raises(RouterGraphBuilderError, match="Unsupported node type"):
            validate_manifest_security([{"name": "bad", "type": "workflow"}], [], _manifest())


# ═══════════════════════════════════════════════════════════════════
class TestPromptRefResolution:
    """LLM/agent prompt_ref declarations must be resolved before graph compilation."""

    def test_llm_prompt_ref_without_resolved_prompt_rejected(self):
        nodes = [_node("planner", prompt_ref="src/chat/prompts.py::PLANNER_PROMPT")]
        with pytest.raises(RouterGraphBuilderError, match="no resolved 'planner_prompt'"):
            validate_manifest_security(nodes, [], _manifest(config={}))

    def test_agent_prompt_ref_without_resolved_prompt_rejected(self):
        nodes = [_agent_node("agent_node", prompt_ref="src/chat/prompts.py::AGENT_PROMPT")]
        with pytest.raises(RouterGraphBuilderError, match="no resolved 'agent_node_prompt'"):
            validate_manifest_security(nodes, [], _manifest(config={}))

    def test_prompt_ref_with_resolved_prompt_passes(self):
        nodes = [_node("planner", prompt_ref="src/chat/prompts.py::PLANNER_PROMPT")]
        validate_manifest_security(
            nodes,
            [],
            _manifest(config={"planner_prompt": "Respond with JSON."}),
        )


# model_secret_ref — no path traversal
# ═══════════════════════════════════════════════════════════════════


class TestSecretRefValidation:
    """model_secret_ref must be alphanumeric with underscores/hyphens."""

    def test_valid_ref(self):
        validate_manifest_security([_node("a", model_secret_ref="my-key_v2")], [], _manifest())

    def test_path_traversal_rejected(self):
        with pytest.raises(SecurityError, match="Invalid model_secret_ref"):
            validate_manifest_security(
                [_node("a", model_secret_ref="../../etc/passwd")], [], _manifest()
            )

    def test_slash_rejected(self):
        with pytest.raises(SecurityError, match="Path separators"):
            validate_manifest_security(
                [_node("a", model_secret_ref="keys/openai")], [], _manifest()
            )


# ═══════════════════════════════════════════════════════════════════
# Edge integrity — dangling references
# ═══════════════════════════════════════════════════════════════════


class TestEdgeIntegrity:
    """Every edge from/to must reference existing node or __start__/__end__."""

    def test_valid_edges(self):
        nodes = [_node("a"), _node("b")]
        edges = [_edge("__start__", "a"), _edge("a", "b"), _edge("b", "__end__")]
        validate_manifest_security(nodes, edges, _manifest())

    def test_dangling_from_rejected(self):
        nodes = [_node("a")]
        edges = [_edge("nonexistent", "a")]
        with pytest.raises(RouterGraphBuilderError, match="non-existent source"):
            validate_manifest_security(nodes, edges, _manifest())

    def test_dangling_to_rejected(self):
        nodes = [_node("a")]
        edges = [_edge("a", "nonexistent")]
        with pytest.raises(RouterGraphBuilderError, match="non-existent target"):
            validate_manifest_security(nodes, edges, _manifest())

    def test_conditional_edge_dangling_target(self):
        nodes = [_node("a")]
        edges = [_cond_edge("a", {"yes": "nonexistent", "no": "__end__"})]
        with pytest.raises(RouterGraphBuilderError, match="non-existent node"):
            validate_manifest_security(nodes, edges, _manifest())

    def test_conditional_edge_valid(self):
        nodes = [_node("a"), _node("b")]
        edges = [_cond_edge("a", {"yes": "b", "no": "__end__"})]
        validate_manifest_security(nodes, edges, _manifest())


# ═══════════════════════════════════════════════════════════════════
# Cycle detection
# ═══════════════════════════════════════════════════════════════════


class TestCycleDetection:
    """Back-edges require config.max_retries."""

    def test_acyclic_graph_passes(self):
        nodes = [_node("a"), _node("b"), _node("c")]
        edges = [_edge("a", "b"), _edge("b", "c")]
        validate_manifest_security(nodes, edges, _manifest())

    def test_cycle_without_max_retries_rejected(self):
        nodes = [_node("a"), _node("b")]
        edges = [_edge("a", "b"), _edge("b", "a")]
        with pytest.raises(RouterGraphBuilderError, match="cycle"):
            validate_manifest_security(nodes, edges, _manifest())

    def test_cycle_with_max_retries_passes(self):
        nodes = [_node("a"), _node("b")]
        edges = [_edge("a", "b"), _edge("b", "a")]
        validate_manifest_security(nodes, edges, _manifest(max_retries=3))

    def test_self_loop_detected(self):
        nodes = [_node("a")]
        edges = [_edge("a", "a")]
        with pytest.raises(RouterGraphBuilderError, match="cycle"):
            validate_manifest_security(nodes, edges, _manifest())

    def test_conditional_cycle_detected(self):
        nodes = [_node("a"), _node("b")]
        edges = [_edge("a", "b"), _cond_edge("b", {"retry": "a", "done": "__end__"})]
        with pytest.raises(RouterGraphBuilderError, match="cycle"):
            validate_manifest_security(nodes, edges, _manifest())


# ═══════════════════════════════════════════════════════════════════
# Service dependency validation
# ═══════════════════════════════════════════════════════════════════


class TestServiceDependency:
    """Platform tools require their service enabled in manifest."""

    def test_brain_tool_requires_brain_service(self):
        nodes = [_tool_node("a", "platform:brain_search")]
        with pytest.raises(RouterGraphBuilderError, match="service 'brain'"):
            validate_manifest_security(nodes, [], _manifest())

    def test_brain_tool_with_service_enabled(self):
        nodes = [_tool_node("a", "platform:brain_search")]
        manifest = _manifest(services={"brain": {"enabled": True}})
        validate_manifest_security(nodes, [], manifest)

    def test_truthy_non_bool_enabled_rejected(self):
        nodes = [_tool_node("a", "platform:brain_search")]
        manifest = _manifest(services={"brain": {"enabled": "true"}})
        with pytest.raises(RouterGraphBuilderError, match="service 'brain'"):
            validate_manifest_security(nodes, [], manifest)

    def test_shield_tool_requires_shield_service(self):
        nodes = [_tool_node("a", "platform:shield_check")]
        with pytest.raises(RouterGraphBuilderError, match="service 'shield'"):
            validate_manifest_security(nodes, [], _manifest())

    def test_self_hosted_tools_skip_service_check(self):
        """router_* and language_* tools run inside Router — no service needed."""
        nodes = [
            _tool_node("a", "router_sql"),
            _tool_node("b", "language_detect"),
        ]
        validate_manifest_security(nodes, [], _manifest())

    def test_unknown_platform_prefix_rejected(self):
        nodes = [_tool_node("a", "platform:unknown_tool")]
        with pytest.raises(RouterGraphBuilderError, match="Unknown platform tool prefix"):
            validate_manifest_security(nodes, [], _manifest())


# ═══════════════════════════════════════════════════════════════════
# Tool binding format
# ═══════════════════════════════════════════════════════════════════


class TestToolBinding:
    """Tool nodes must declare valid tool_binding."""

    def test_empty_binding_rejected(self):
        nodes = [{"name": "a", "type": "tool", "tool_binding": ""}]
        with pytest.raises(RouterGraphBuilderError, match="invalid tool_binding"):
            validate_manifest_security(nodes, [], _manifest())

    def test_missing_binding_rejected(self):
        nodes = [{"name": "a", "type": "tool"}]
        with pytest.raises(RouterGraphBuilderError, match="invalid tool_binding"):
            validate_manifest_security(nodes, [], _manifest())


# ═══════════════════════════════════════════════════════════════════
# Agent tool references
# ═══════════════════════════════════════════════════════════════════


class TestAgentToolRefs:
    """Agent nodes validate tool reference format."""

    def test_valid_tool_ref(self):
        nodes = [_agent_node("a", tools=["platform:brain_search"])]
        manifest = _manifest(services={"brain": {"enabled": True}})
        validate_manifest_security(nodes, [], manifest)

    def test_non_string_tool_ref_rejected(self):
        nodes = [{"name": "a", "type": "agent", "tools": [123]}]
        with pytest.raises(RouterGraphBuilderError, match="non-string tool reference"):
            validate_manifest_security(nodes, [], _manifest())

    def test_non_list_tools_rejected(self):
        nodes = [{"name": "a", "type": "agent", "tools": {"platform:brain_search": True}}]
        with pytest.raises(RouterGraphBuilderError, match="invalid tools list"):
            validate_manifest_security(nodes, [], _manifest())

    def test_invalid_tool_ref_format_rejected(self):
        nodes = [_agent_node("a", tools=["not a valid ref!!!"])]
        with pytest.raises(RouterGraphBuilderError, match="invalid"):
            validate_manifest_security(nodes, [], _manifest())
