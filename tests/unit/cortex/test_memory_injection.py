"""Tests for Brain Phase C — compiler memory integration.

Tests: pipeline.memory auto-injection, retrieval depth tiers,
experience_lookup on LLM nodes.
Memory chain: _memory_load → first_node ... last_node → _memory_save
PII scanning is internal to brain_memory_write (via Zero), not a graph node.
"""

import pytest
from contextunity.core.exceptions import ConfigurationError


class TestMemoryAutoInjection:
    """pipeline.memory: true auto-injects memory_load and memory_save nodes."""

    def test_memory_true_injects_two_nodes(self):
        """memory: true → 2 extra nodes (_memory_load, _memory_save)."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [
            {"name": "generator", "type": "llm", "config": {}},
            {"name": "reflector", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "generator"},
            {"from_node": "generator", "to_node": "reflector"},
            {"from_node": "reflector", "to_node": "__end__"},
        ]
        graph_config = {"pipeline": {"memory": True}, "config": {}}

        new_nodes, new_edges = inject_memory_nodes(nodes, edges, graph_config)

        node_names = [n["name"] for n in new_nodes]
        assert "_memory_load" in node_names
        assert "_memory_save" in node_names
        assert len(new_nodes) == 4  # 2 original + 2 injected

    def test_memory_true_rewires_start_edge(self):
        """__start__ → first_node becomes __start__ → _memory_load → first_node."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "generator", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "generator"},
            {"from_node": "generator", "to_node": "__end__"},
        ]
        graph_config = {"pipeline": {"memory": True}, "config": {}}

        _, new_edges = inject_memory_nodes(nodes, edges, graph_config)

        start_edges = [e for e in new_edges if e.get("from_node") == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0]["to_node"] == "_memory_load"

        load_edges = [e for e in new_edges if e.get("from_node") == "_memory_load"]
        assert len(load_edges) == 1
        assert load_edges[0]["to_node"] == "generator"

    def test_memory_true_rewires_end_edge(self):
        """last_node → __end__ becomes last_node → _memory_save → __end__."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "generator", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "generator"},
            {"from_node": "generator", "to_node": "__end__"},
        ]
        graph_config = {"pipeline": {"memory": True}, "config": {}}

        _, new_edges = inject_memory_nodes(nodes, edges, graph_config)

        gen_edges = [e for e in new_edges if e.get("from_node") == "generator"]
        assert len(gen_edges) == 1
        assert gen_edges[0]["to_node"] == "_memory_save"

        save_edges = [e for e in new_edges if e.get("from_node") == "_memory_save"]
        assert len(save_edges) == 1
        assert save_edges[0]["to_node"] == "__end__"

    def test_memory_false_no_injection(self):
        """memory: false → no changes."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "generator", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "generator"},
            {"from_node": "generator", "to_node": "__end__"},
        ]
        graph_config = {"pipeline": {"memory": False}, "config": {}}

        new_nodes, new_edges = inject_memory_nodes(nodes, edges, graph_config)

        assert len(new_nodes) == 1
        assert new_nodes == nodes

    def test_memory_absent_no_injection(self):
        """No pipeline key → no changes."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "gen", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "gen"},
            {"from_node": "gen", "to_node": "__end__"},
        ]
        graph_config = {"config": {}}

        new_nodes, new_edges = inject_memory_nodes(nodes, edges, graph_config)
        assert len(new_nodes) == 1

    def test_injected_nodes_are_platform_type(self):
        """Injected memory nodes must be tool type with brain bindings."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "gen", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "gen"},
            {"from_node": "gen", "to_node": "__end__"},
        ]
        graph_config = {"pipeline": {"memory": True}, "config": {}}

        new_nodes, _ = inject_memory_nodes(nodes, edges, graph_config)

        load_node = next(n for n in new_nodes if n["name"] == "_memory_load")
        save_node = next(n for n in new_nodes if n["name"] == "_memory_save")

        assert load_node["type"] == "tool"
        assert load_node["tool_binding"] == "brain_memory_read"
        assert save_node["type"] == "tool"
        assert save_node["tool_binding"] == "brain_memory_write"

    def test_memory_save_has_pii_scan_flag(self):
        """_memory_save config includes pii_scan: true for executor-level PII via Zero."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "gen", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "gen"},
            {"from_node": "gen", "to_node": "__end__"},
        ]
        graph_config = {"pipeline": {"memory": True}, "config": {}}

        new_nodes, _ = inject_memory_nodes(nodes, edges, graph_config)

        save_node = next(n for n in new_nodes if n["name"] == "_memory_save")
        assert save_node["config"]["tool_config"]["pii_scan"] is True

    def test_memory_depth_propagated_to_load_node(self):
        """pipeline.memory_depth propagated to _memory_load config."""
        from contextunity.router.cortex.compiler.memory_injection import (
            inject_memory_nodes,
        )

        nodes = [{"name": "gen", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "gen"},
            {"from_node": "gen", "to_node": "__end__"},
        ]
        graph_config = {
            "pipeline": {"memory": True, "memory_depth": "deep"},
            "config": {},
        }

        new_nodes, _ = inject_memory_nodes(nodes, edges, graph_config)

        load_node = next(n for n in new_nodes if n["name"] == "_memory_load")
        assert load_node["config"]["tool_config"]["search_depth"] == "deep"


class TestRetrievalDepthTiers:
    """Retrieval depth tier configuration for brain_memory_read."""

    def test_shallow_config(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            get_retrieval_params,
        )

        params = get_retrieval_params("shallow")
        assert params["include_facts"] is True
        assert params["include_episodes"] is False
        assert params["facts_limit"] == 10

    def test_standard_config(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            get_retrieval_params,
        )

        params = get_retrieval_params("standard")
        assert params["include_facts"] is True
        assert params["include_episodes"] is True
        assert params["episodes_limit"] == 5

    def test_deep_config(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            get_retrieval_params,
        )

        params = get_retrieval_params("deep")
        assert params["include_experiences"] is True

    def test_default_is_standard(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            get_retrieval_params,
        )

        params = get_retrieval_params(None)
        assert params["include_episodes"] is True
        assert params["include_experiences"] is False

    def test_invalid_depth_raises_configuration_error(self):
        """Invalid depth tier → ConfigurationError, not silent fallback."""
        from contextunity.router.cortex.compiler.memory_injection import (
            get_retrieval_params,
        )

        with pytest.raises(ConfigurationError, match="Invalid memory_depth"):
            get_retrieval_params("../../malicious")

    def test_unknown_depth_raises_configuration_error(self):
        """Unknown depth tier name → ConfigurationError."""
        from contextunity.router.cortex.compiler.memory_injection import (
            get_retrieval_params,
        )

        with pytest.raises(ConfigurationError, match="Invalid memory_depth"):
            get_retrieval_params("ultra_deep")


class TestExperienceLookupConfig:
    """experience_lookup: true on LLM node config."""

    def test_build_experience_lookup_config(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            build_experience_lookup_config,
        )

        config = {
            "experience_lookup": True,
            "experience_min_q": 0.8,
            "experience_limit": 5,
        }
        result = build_experience_lookup_config(config)
        assert result["enabled"] is True
        assert result["min_q"] == 0.8
        assert result["limit"] == 5

    def test_experience_lookup_defaults(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            build_experience_lookup_config,
        )

        config = {"experience_lookup": True}
        result = build_experience_lookup_config(config)
        assert result["min_q"] == 0.7
        assert result["limit"] == 3

    def test_experience_lookup_false(self):
        from contextunity.router.cortex.compiler.memory_injection import (
            build_experience_lookup_config,
        )

        result = build_experience_lookup_config({"experience_lookup": False})
        assert result["enabled"] is False

    def test_experience_min_q_out_of_bounds_raises(self):
        """min_q > 1.0 → Pydantic validation error."""
        from pydantic import ValidationError

        from contextunity.router.cortex.compiler.memory_injection import (
            build_experience_lookup_config,
        )

        with pytest.raises(ValidationError):
            build_experience_lookup_config({"experience_lookup": True, "experience_min_q": 1.5})

    def test_experience_limit_out_of_bounds_raises(self):
        """limit > 100 → Pydantic validation error."""
        from pydantic import ValidationError

        from contextunity.router.cortex.compiler.memory_injection import (
            build_experience_lookup_config,
        )

        with pytest.raises(ValidationError):
            build_experience_lookup_config({"experience_lookup": True, "experience_limit": 999999})

    def test_experience_limit_zero_raises(self):
        """limit = 0 → Pydantic validation error (must be >= 1)."""
        from pydantic import ValidationError

        from contextunity.router.cortex.compiler.memory_injection import (
            build_experience_lookup_config,
        )

        with pytest.raises(ValidationError):
            build_experience_lookup_config({"experience_lookup": True, "experience_limit": 0})
