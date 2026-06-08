"""Test Template Loader for the Graph Compiler (Phase 4).

RED phase: These tests define the template system contracts.
All tests must FAIL before implementation — if any pass immediately,
the test is suspect (testing existing behavior or mocks).

Covers:
- 4.A Template loading (valid, missing, schema validation)
- 4.A Pydantic hardening (extra fields, Literal, bounds)
- 4.B Override merge (deep merge, unknown node, type lock)
"""

import pytest
from contextunity.core.exceptions import ConfigurationError

# ── 4.A.1: Template Loading ───────────────────────────────────────


class TestTemplateLoading:
    """Load template by name from YAML resources."""

    def test_load_valid_template_returns_definition(self):
        """4.1: load_template('retrieval_augmented') returns TemplateDefinition."""
        from contextunity.router.cortex.compiler.template_loader import (
            load_template,
        )

        result = load_template("retrieval_augmented")
        assert result.name == "retrieval_augmented"
        assert len(result.nodes) > 0
        assert len(result.edges) > 0
        assert result.defaults is not None

    def test_load_nonexistent_template_raises(self):
        """4.2: Missing template → ConfigurationError."""
        from contextunity.router.cortex.compiler.template_loader import (
            load_template,
        )

        with pytest.raises(ConfigurationError, match="not_a_real_template"):
            load_template("not_a_real_template")


# ── 4.A.2: Pydantic Schema Hardening ─────────────────────────────


class TestTemplateSchemaHardening:
    """Template Pydantic models enforce strict contracts."""

    def test_template_node_type_literal(self):
        """Node type must be Literal['llm', 'agent', 'tool']; platform is a binding namespace."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateNode,
        )

        with pytest.raises(Exception):
            TemplateNode(name="bad", type="arbitrary_code_exec")

    def test_template_node_rejects_extra_fields(self):
        """extra='forbid' on TemplateNode."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateNode,
        )

        with pytest.raises(Exception, match="extra"):
            TemplateNode(name="n", type="llm", backdoor=True)

    def test_template_node_response_format_literal(self):
        """response_format must be Literal['text', 'json'] or None."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateNode,
        )

        with pytest.raises(Exception):
            TemplateNode(name="n", type="llm", response_format="yaml")


# ── 4.B: Override Merge ──────────────────────────────────────────


class TestOverrideMerge:
    """merge_overrides() deep-merges consumer config onto template."""

    def _make_template(self):
        """Helper: minimal TemplateDefinition for override tests."""
        from contextunity.router.cortex.compiler.node_config import NodeConfig
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateConfig,
            TemplateDefaults,
            TemplateDefinition,
            TemplateEdge,
            TemplateNode,
        )

        return TemplateDefinition(
            name="test_tpl",
            version="1.0",
            description="Test",
            nodes=[
                TemplateNode(name="intent", type="llm", prompt_ref="test_intent"),
                TemplateNode(
                    name="search",
                    type="tool",
                    tool_binding="brain_search",
                    config=NodeConfig(tool_config={"top_k": 10, "rerank": True}),
                ),
            ],
            edges=[
                TemplateEdge(from_node="__start__", to_node="intent"),
                TemplateEdge(from_node="intent", to_node="search"),
                TemplateEdge(from_node="search", to_node="__end__"),
            ],
            defaults=TemplateDefaults(model="openai/gpt-4o", temperature=0.3),
            config=TemplateConfig(timeout=60),
        )

    def test_override_replaces_node_model(self):
        """Override intent node model without touching search."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        tpl = self._make_template()
        overrides = {"intent": {"model": "anthropic/claude-sonnet-4-20250514"}}
        result = merge_overrides(tpl, overrides)

        # Intent model changed
        intent = next(n for n in result.nodes if n.name == "intent")
        assert intent.model == "anthropic/claude-sonnet-4-20250514"

        # Search untouched (tool knobs live under tool_config)
        search = next(n for n in result.nodes if n.name == "search")
        assert search.config.as_manifest_dict() == {"tool_config": {"top_k": 10, "rerank": True}}

    def test_override_deep_merges_config(self):
        """Override config.top_k without losing config.rerank."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        tpl = self._make_template()
        overrides = {"search": {"config": {"tool_config": {"top_k": 20}}}}
        result = merge_overrides(tpl, overrides)

        search = next(n for n in result.nodes if n.name == "search")
        merged_cfg = search.config.as_manifest_dict()
        assert merged_cfg["tool_config"]["top_k"] == 20
        assert merged_cfg["tool_config"]["rerank"] is True  # preserved

    def test_override_nonexistent_node_raises(self):
        """Override for node not in template → ConfigurationError."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        tpl = self._make_template()
        overrides = {"phantom_node": {"model": "bad"}}
        with pytest.raises(ConfigurationError, match="phantom_node"):
            merge_overrides(tpl, overrides)

    def test_override_cannot_change_node_type(self):
        """Security: cannot change tool → llm via override."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        tpl = self._make_template()
        overrides = {"search": {"type": "llm"}}
        with pytest.raises(ConfigurationError, match="Cannot change node type"):
            merge_overrides(tpl, overrides)

    def test_override_rejects_unsafe_keys(self):
        """Override with 'name' key → ConfigurationError (identity injection)."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        tpl = self._make_template()
        overrides = {"intent": {"name": "hijacked_node"}}
        with pytest.raises(ConfigurationError, match="unsafe keys"):
            merge_overrides(tpl, overrides)


# ── Hardening Audit Tests ────────────────────────────────────────


class TestTemplateSecurityHardening:
    """Post-audit tests for Phase 4 security fixes."""

    def test_node_name_regex_rejects_uppercase(self):
        """Node names must be lowercase — CamelCase rejected."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateNode,
        )

        with pytest.raises(Exception, match="Invalid node name"):
            TemplateNode(name="CamelCase", type="llm")

    def test_node_name_regex_accepts_valid(self):
        """Valid names: lowercase, underscores, leading underscore."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateNode,
        )

        node = TemplateNode(name="extract_query", type="llm")
        assert node.name == "extract_query"

        node2 = TemplateNode(name="_internal", type="tool")
        assert node2.name == "_internal"

    def test_model_secret_ref_rejects_path_traversal(self):
        """model_secret_ref with path separators → rejected."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateDefaults,
        )

        with pytest.raises(Exception, match="model_secret_ref"):
            TemplateDefaults(model_secret_ref="../../../etc/shadow")

    def test_model_secret_ref_accepts_valid(self):
        """Alphanumeric with underscores/hyphens accepted."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateDefaults,
        )

        defaults = TemplateDefaults(model_secret_ref="my-secret_key")
        assert defaults.model_secret_ref == "my-secret_key"

    def test_validation_error_not_bare_exception(self):
        """Schema errors caught as ValidationError, not bare Exception."""
        from pydantic import ValidationError

        from contextunity.router.cortex.compiler.template_loader import (
            TemplateNode,
        )

        with pytest.raises(ValidationError):
            TemplateNode(name="valid", type="invalid_type_not_in_literal")

    def test_no_condition_value_field(self):
        """TemplateEdge has no condition_value field (removed in audit)."""
        from contextunity.router.cortex.compiler.template_loader import (
            TemplateEdge,
        )

        with pytest.raises(Exception, match="extra"):
            TemplateEdge(from_node="a", to_node="b", condition_value="old_field")
