"""Test edge integrity and service dependency validation for Graph Compiler.

RED phase: Tests define the contracts for edge validation, service dependency
checking, and federated binding cross-referencing in _validate_manifest_security().
"""

import pytest

from contextunity.router.core.exceptions import RouterGraphBuilderError


class TestEdgeIntegrityValidation:
    """Every from:/to: must reference an existing node name or __start__/__end__."""

    def _build(self, nodes, edges, config=None):
        from contextunity.router.cortex.compiler.builder import build_local_graph

        return build_local_graph(nodes, edges, config or {})

    def test_edge_to_nonexistent_node_rejected(self):
        nodes = [{"name": "classifier", "type": "llm", "model": "test/model"}]
        edges = [
            {"from_node": "__start__", "to_node": "classifier"},
            {"from_node": "classifier", "to_node": "phantom_node"},  # does not exist
        ]
        with pytest.raises(RouterGraphBuilderError, match="phantom_node"):
            self._build(nodes, edges)

    def test_edge_from_nonexistent_node_rejected(self):
        nodes = [{"name": "classifier", "type": "llm", "model": "test/model"}]
        edges = [
            {"from_node": "phantom_node", "to_node": "classifier"},  # does not exist
        ]
        with pytest.raises(RouterGraphBuilderError, match="phantom_node"):
            self._build(nodes, edges)

    def test_valid_edges_with_start_end_pass(self):
        nodes = [
            {"name": "step_a", "type": "llm", "model": "test/model"},
            {"name": "step_b", "type": "llm", "model": "test/model"},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "step_a"},
            {"from_node": "step_a", "to_node": "step_b"},
            {"from_node": "step_b", "to_node": "__end__"},
        ]
        # Should compile without error
        graph = self._build(nodes, edges)
        assert graph is not None

    def test_conditional_edge_with_nonexistent_target_rejected(self):
        nodes = [{"name": "router_node", "type": "llm", "model": "test/model"}]
        edges = [
            {"from_node": "__start__", "to_node": "router_node"},
            {
                "from_node": "router_node",
                "condition_key": "route",
                "condition_map": {
                    "path_a": "phantom_a",  # does not exist
                    "default": "__end__",
                },
            },
        ]
        with pytest.raises(RouterGraphBuilderError, match="phantom_a"):
            self._build(nodes, edges)

    def test_conditional_edge_with_all_valid_targets_pass(self):
        nodes = [
            {"name": "router_node", "type": "llm", "model": "test/model"},
            {"name": "path_a_node", "type": "llm", "model": "test/model"},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "router_node"},
            {
                "from_node": "router_node",
                "condition_key": "route",
                "condition_map": {
                    "go_a": "path_a_node",
                    "done": "__end__",
                },
            },
            {"from_node": "path_a_node", "to_node": "__end__"},
        ]
        graph = self._build(nodes, edges)
        assert graph is not None


class TestServiceDependencyValidation:
    """Platform nodes require their target service enabled in config."""

    def _build(self, nodes, edges, config=None):
        from contextunity.router.cortex.compiler.builder import build_local_graph

        return build_local_graph(nodes, edges, config or {})

    def test_brain_tool_without_brain_enabled_rejected(self):
        nodes = [
            {
                "name": "search_kb",
                "type": "tool",
                "tool_binding": "brain_search",
            }
        ]
        edges = [
            {"from_node": "__start__", "to_node": "search_kb"},
            {"from_node": "search_kb", "to_node": "__end__"},
        ]
        # No services config → brain not enabled → rejected
        with pytest.raises(RouterGraphBuilderError, match="brain"):
            self._build(nodes, edges, config={})

    def test_brain_tool_with_brain_enabled_passes(self):
        nodes = [
            {
                "name": "search_kb",
                "type": "tool",
                "tool_binding": "brain_search",
            }
        ]
        edges = [
            {"from_node": "__start__", "to_node": "search_kb"},
            {"from_node": "search_kb", "to_node": "__end__"},
        ]
        config = {"services": {"brain": {"enabled": True}}}
        # Should compile
        graph = self._build(nodes, edges, config=config)
        assert graph is not None


class TestToolBindingNamespace:
    """tool_binding routes by namespace; bare means platform."""

    def _build(self, nodes, edges, config=None):
        from contextunity.router.cortex.compiler.builder import build_local_graph

        return build_local_graph(nodes, edges, config or {})

    def test_bare_unknown_binding_is_platform_and_rejected_by_prefix(self):
        nodes = [
            {
                "name": "fetch_data",
                "type": "tool",
                "tool_binding": "export_products",
            }
        ]
        edges = [
            {"from_node": "__start__", "to_node": "fetch_data"},
            {"from_node": "fetch_data", "to_node": "__end__"},
        ]
        with pytest.raises(RouterGraphBuilderError, match="Unknown platform tool prefix"):
            self._build(nodes, edges)

    def test_platform_node_type_rejected(self):
        nodes = [
            {
                "name": "extract",
                "type": "platform",
                "tool_binding": "router_extract_query",
            }
        ]
        edges = [
            {"from_node": "__start__", "to_node": "extract"},
            {"from_node": "extract", "to_node": "__end__"},
        ]
        with pytest.raises(RouterGraphBuilderError, match="Unsupported node type 'platform'"):
            self._build(nodes, edges)

    def test_bare_platform_binding_passes_for_router_tool(self):
        nodes = [
            {
                "name": "extract",
                "type": "tool",
                "tool_binding": "router_extract_query",
            }
        ]
        edges = [
            {"from_node": "__start__", "to_node": "extract"},
            {"from_node": "extract", "to_node": "__end__"},
        ]
        graph = self._build(nodes, edges)
        assert graph is not None

    def test_tool_binding_with_namespace_passes(self):
        nodes = [
            {
                "name": "fetch_data",
                "type": "tool",
                "tool_binding": "federated:export_products",
            }
        ]
        edges = [
            {"from_node": "__start__", "to_node": "fetch_data"},
            {"from_node": "fetch_data", "to_node": "__end__"},
        ]
        graph = self._build(nodes, edges)
        assert graph is not None


# ── build_graph (top-level orchestrator) ──────────────────────────


class TestBuildGraphOrchestrator:
    """build_graph resolves template names and rejects legacy aliases."""

    def _config(self):
        from types import SimpleNamespace

        return SimpleNamespace(router=SimpleNamespace(override_path="", graph=""))

    def test_resolves_rlm_bulk_matcher_template(self):
        from unittest.mock import patch

        from contextunity.router.cortex.builder import build_graph

        with (
            patch(
                "contextunity.router.cortex.builder.get_core_config", return_value=self._config()
            ),
            patch("contextunity.router.cortex.builder.graph_registry.has", return_value=False),
            patch(
                "contextunity.router.cortex.compiler.builder.build_from_template",
                return_value="compiled-graph",
            ) as mock_build,
        ):
            graph = build_graph("rlm_bulk_matcher")
        assert graph == "compiled-graph"
        mock_build.assert_called_once_with("rlm_bulk_matcher")

    def test_rejects_legacy_matcher_alias(self):
        from unittest.mock import patch

        from contextunity.router.cortex.builder import build_graph

        with (
            patch(
                "contextunity.router.cortex.builder.get_core_config", return_value=self._config()
            ),
            patch("contextunity.router.cortex.builder.graph_registry.has", return_value=False),
        ):
            with pytest.raises(RouterGraphBuilderError, match="matcher"):
                build_graph("matcher")

    def test_rejects_removed_commerce_builtin(self):
        from unittest.mock import patch

        from contextunity.router.cortex.builder import build_graph

        with (
            patch(
                "contextunity.router.cortex.builder.get_core_config", return_value=self._config()
            ),
            patch("contextunity.router.cortex.builder.graph_registry.has", return_value=False),
        ):
            with pytest.raises(RouterGraphBuilderError, match="commerce"):
                build_graph("commerce")
