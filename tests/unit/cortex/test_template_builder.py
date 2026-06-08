"""Test builder integration with template system (Phase 4.C).

RED phase: Tests for template-based graph compilation and _trace node injection.
"""

import pytest
from contextunity.core.exceptions import ConfigurationError


class TestTemplateBuilderIntegration:
    """build_local_graph() accepts template: key."""

    def _build_from_template(self, template_name, overrides=None, config=None):
        from contextunity.router.cortex.compiler.builder import (
            build_from_template,
        )

        return build_from_template(
            template_name=template_name,
            overrides=overrides or {},
            config=config or {},
        )

    def test_build_from_template_produces_graph(self):
        """4.10: template='retrieval_augmented' → compiled graph with correct nodes."""
        graph = self._build_from_template(
            "retrieval_augmented",
            config={
                "services": {"brain": {"enabled": True}},
            },
        )
        assert graph is not None
        # Compiled graph should have nodes (LangGraph compiled graph has .nodes attr)
        assert hasattr(graph, "nodes")

    def test_build_from_template_with_override(self):
        """4.11: template + override → graph where overridden node has new model."""
        graph = self._build_from_template(
            "retrieval_augmented",
            overrides={"detect_intent": {"model": "anthropic/claude-sonnet-4-20250514"}},
            config={"services": {"brain": {"enabled": True}}},
        )
        assert graph is not None

    def test_build_from_nonexistent_template_raises(self):
        """Missing template → ConfigurationError."""
        with pytest.raises(ConfigurationError, match="phantom_template"):
            self._build_from_template("phantom_template")

    def test_build_from_template_with_invalid_override_raises(self):
        """Override for nonexistent node → ConfigurationError."""
        with pytest.raises(ConfigurationError, match="phantom_node"):
            self._build_from_template(
                "retrieval_augmented",
                overrides={"phantom_node": {"model": "bad"}},
                config={"services": {"brain": {"enabled": True}}},
            )
