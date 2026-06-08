"""Test registration mixin template routing (Phase 4.E).

RED phase: Tests define contracts for template-based graph registration
via _register_graph() dispatching to build_from_template().

Security invariants tested:
- Atomic hot-reload: compile-first-then-swap
- project:{id}:{name} namespacing enforced
- ConfigurationError from template loader propagated
- Unknown template → ConfigurationError (not bare ValueError)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from contextunity.core.exceptions import ConfigurationError


class _FakeGraphConfig:
    """Minimal stub matching GraphConfig model contract."""

    def __init__(
        self,
        name: str,
        template: str | None = None,
        builtin: str | None = None,
        config: dict | None = None,
        nodes: list | None = None,
        edges: list | None = None,
        overrides: dict | None = None,
    ):
        self.name = name
        self.template = template
        self.builtin = builtin
        self.config = config or {}
        self.nodes = nodes
        self.edges = edges
        self.overrides = overrides or {}


class TestRegistrationTemplateRouting:
    """_register_graph() routes template manifests through build_from_template."""

    def _make_mixin(self):
        """Create a minimal registration mixin instance."""
        from contextunity.router.service.mixins.registration import (
            RegistrationMixin,
        )

        mixin = RegistrationMixin.__new__(RegistrationMixin)
        mixin._project_graphs = {}
        mixin._project_configs = {}
        return mixin

    @patch("contextunity.router.core.registry.graph_registry")
    def test_yaml_template_routes_to_build_from_template(self, mock_registry):
        """4.21: template='yaml:retrieval_augmented' routes through build_from_template."""
        mixin = self._make_mixin()

        # yaml: prefix triggers template loader path
        gc = _FakeGraphConfig(
            name="my_rag",
            template="yaml:retrieval_augmented",
            config={"services": {"brain": {"enabled": True}}},
        )
        result = mixin._register_graph("test_project", gc)

        assert result == "project:test_project:my_rag"
        # Graph registered under namespaced key
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[0][0] == "project:test_project:my_rag"

    @patch("contextunity.router.core.registry.graph_registry")
    def test_yaml_template_with_overrides(self, mock_registry):
        """4.21: yaml template + overrides passes through to build_from_template."""
        mixin = self._make_mixin()

        gc = _FakeGraphConfig(
            name="my_rag",
            template="yaml:retrieval_augmented",
            config={"services": {"brain": {"enabled": True}}},
            overrides={"detect_intent": {"model": "anthropic/claude-sonnet-4-20250514"}},
        )
        result = mixin._register_graph("test_project", gc)
        assert result == "project:test_project:my_rag"

    @patch("contextunity.router.core.registry.graph_registry")
    def test_yaml_nonexistent_template_raises_configuration_error(self, mock_registry):
        """Missing yaml template → ConfigurationError, registry untouched."""
        mixin = self._make_mixin()

        gc = _FakeGraphConfig(
            name="bad_graph",
            template="yaml:nonexistent_template",
            config={},
        )
        with pytest.raises(ConfigurationError, match="nonexistent_template"):
            mixin._register_graph("test_project", gc)

        # Registry not touched (atomic safety)
        mock_registry.register.assert_not_called()

    def test_unknown_template_raises_configuration_error_not_value_error(self):
        """Unknown template (not yaml: prefix, not known builtin) → ConfigurationError."""
        mixin = self._make_mixin()

        gc = _FakeGraphConfig(name="bad", template="totally_unknown")
        with pytest.raises(ConfigurationError):
            mixin._register_graph("test_project", gc)


class TestRegistrationAtomicSafety:
    """Compilation failures leave registry untouched."""

    def _make_mixin(self):
        from contextunity.router.service.mixins.registration import (
            RegistrationMixin,
        )

        mixin = RegistrationMixin.__new__(RegistrationMixin)
        mixin._project_graphs = {}
        mixin._project_configs = {}
        return mixin

    @patch("contextunity.router.core.registry.graph_registry")
    def test_yaml_compile_failure_atomic(self, mock_registry):
        """If yaml template compilation fails, registry stays untouched."""
        mixin = self._make_mixin()

        gc = _FakeGraphConfig(
            name="broken",
            template="yaml:retrieval_augmented",
            config={},
            overrides={"phantom_node": {"model": "bad"}},
        )
        with pytest.raises(ConfigurationError):
            mixin._register_graph("test_project", gc)

        mock_registry.register.assert_not_called()
        assert "test_project" not in mixin._project_graphs


class TestRegistrationDeregister:
    """Deregistration cleans up all multi-graph entries."""

    def _make_mixin(self):
        from contextunity.router.service.mixins.registration import RegistrationMixin

        mixin = RegistrationMixin.__new__(RegistrationMixin)
        mixin._project_graphs = {}
        mixin._project_configs = {}
        mixin._project_tools = {}
        return mixin

    @patch("contextunity.router.modules.tools.deregister_tool")
    def test_deregister_multi_graph(self, _mock_deregister):
        """8.10: Deregister supports multiple graphs correctly."""
        mixin = self._make_mixin()
        mixin._project_graphs["test_project"] = {
            "default": "project:test_project:matcher",
            "matcher": "project:test_project:matcher",
            "chat": "project:test_project:chat",
        }

        deregistered = mixin._deregister_project("test_project")

        assert "test_project" not in mixin._project_graphs
        assert "graph:project:test_project:matcher" in deregistered
        assert "graph:project:test_project:chat" in deregistered
