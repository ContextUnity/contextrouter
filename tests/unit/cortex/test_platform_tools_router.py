"""Tests for Router platform tools — router_* capability registration.

TDD RED: These tests define the contract for Phase 5.
Each router_* tool wraps existing RAG/SQL node business logic
as a platform tool with validated config and router:execute scope.
"""

from __future__ import annotations

import pytest

from contextunity.router.cortex.compiler.platform_registry import (
    PlatformToolRegistry,
)

# ── 5.1: router_ prefix is known ─────────────────────────────────────


class TestRouterPrefixKnown:
    """router_ must be a known prefix in PlatformToolRegistry."""

    def test_router_prefix_accepted(self):
        """Registry accepts router_ prefixed tools without raising."""
        registry = PlatformToolRegistry()

        async def dummy_executor(state, config):
            return {}

        from pydantic import BaseModel

        class DummyConfig(BaseModel, frozen=True):
            model_config = {"extra": "forbid"}

        # Must NOT raise PlatformServiceError about unknown prefix
        registry.register(
            binding="router_test_tool",
            executor=dummy_executor,
            config_schema=DummyConfig,
            required_scopes=["router:execute"],
        )

        assert registry.has("router_test_tool")

    def test_router_prefix_service_name(self):
        """Service prefix extracted as 'router'."""
        registry = PlatformToolRegistry()

        async def dummy_executor(state, config):
            return {}

        from pydantic import BaseModel

        class DummyConfig(BaseModel, frozen=True):
            model_config = {"extra": "forbid"}

        registry.register(
            binding="router_detect_intent",
            executor=dummy_executor,
            config_schema=DummyConfig,
            required_scopes=["router:execute"],
        )

        reg = registry.get("router_detect_intent")
        assert reg.service_prefix == "router"


# ── 5.2: Router tool configs ─────────────────────────────────────────


# ── 5.2: Registration ────────────────────────────────────────────────


class TestRouterToolRegistration:
    """All 8 RAG tools must register successfully."""

    def test_register_all_rag_tools(self):
        """register_all_platform_tools populates registry with 8 RAG tools."""
        from contextunity.router.cortex.compiler.platform_tools import (
            register_all_platform_tools,
        )

        registry = PlatformToolRegistry()
        register_all_platform_tools(registry)

        expected_rag = [
            "router_extract_query",
            "router_detect_intent",
            "router_retrieve",
            "router_ground",
            "router_generate",
            "router_reflect",
            "router_suggest",
            "router_no_results",
        ]
        for binding in expected_rag:
            assert registry.has(binding), f"Missing: {binding}"

    def test_retrieve_also_requires_brain_read(self):
        """router_retrieve needs both router:execute and brain:read."""
        from contextunity.router.cortex.compiler.platform_tools import (
            register_all_platform_tools,
        )

        registry = PlatformToolRegistry()
        register_all_platform_tools(registry)

        reg = registry.get("router_retrieve")
        assert "brain:read" in reg.required_scopes


# ── 5.3: Adapter execution ──────────────────────────────────────────


class TestRouterToolExecution:
    """Adapters wrap existing node functions correctly."""

    @pytest.mark.asyncio
    async def test_extract_query_adapter(self):
        """router_extract_query wraps extract_user_query sync function."""
        from contextunity.router.cortex.compiler.platform_tools import (
            register_all_platform_tools,
        )
        from contextunity.router.cortex.compiler.platform_tools.extract import (
            ExtractQueryConfig,
        )

        registry = PlatformToolRegistry()
        register_all_platform_tools(registry)

        reg = registry.get("router_extract_query")
        config = ExtractQueryConfig()

        # Minimal state with messages
        state = {
            "messages": [type("Msg", (), {"type": "human", "content": "What is Python?"})()],
        }

        result = await reg.executor(state, config)
        assert "dynamic" in result
        dyn = result["dynamic"]
        assert "user_query" in dyn
        assert "should_retrieve" in dyn

    @pytest.mark.asyncio
    async def test_extract_query_empty_messages(self):
        """router_extract_query handles empty messages gracefully."""
        from contextunity.router.cortex.compiler.platform_tools import (
            register_all_platform_tools,
        )
        from contextunity.router.cortex.compiler.platform_tools.extract import (
            ExtractQueryConfig,
        )

        registry = PlatformToolRegistry()
        register_all_platform_tools(registry)

        reg = registry.get("router_extract_query")
        config = ExtractQueryConfig()

        state = {"messages": []}
        result = await reg.executor(state, config)
        assert "dynamic" in result
        dyn = result["dynamic"]
        assert dyn["user_query"] == ""
        assert dyn["should_retrieve"] is False


# ── 5.5: Direct output mode for platform dispatch ───────────────────


class TestDirectOutputMode:
    """Router platform tools return direct state updates, not wrapped values.

    When output_mode='direct', make_platform_node returns the executor
    result dict as-is (merged into state), instead of wrapping into
    {state_output_key: result}.
    """

    @pytest.mark.asyncio
    async def test_direct_output_passes_result_through(self):
        """output_mode='direct' returns executor result without wrapping."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from contextunity.router.cortex.compiler.node_executors.platform import (
            make_platform_node,
        )

        # Executor returns direct state field updates (like router_* tools)
        mock_executor = AsyncMock(
            return_value={
                "user_query": "What is Python?",
                "should_retrieve": True,
                "messages": [],
            }
        )

        mock_registry = MagicMock()
        mock_registration = MagicMock()
        mock_registration.executor = mock_executor
        mock_registry.get.return_value = mock_registration
        mock_registry.validate_config.return_value = MagicMock()
        mock_registry.check_scopes.return_value = None

        node_spec = {
            "name": "extract_query",
            "type": "tool",
            "tool_binding": "router_extract_query",
            "config": {"output_mode": "direct"},
        }

        with patch(
            "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
            return_value=mock_registry,
        ):
            executor = make_platform_node(node_spec, {})
            result = await executor({"__token__": MagicMock()}, {})

        # Direct mode: result returned as-is, NOT wrapped in final_output
        assert "user_query" in result
        assert result["user_query"] == "What is Python?"
        assert result["should_retrieve"] is True
        assert "final_output" not in result

    @pytest.mark.asyncio
    async def test_default_wrapped_output_unchanged(self):
        """Default behavior (no output_mode) still wraps in state_output_key."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from contextunity.router.cortex.compiler.node_executors.platform import (
            make_platform_node,
        )

        mock_executor = AsyncMock(return_value={"results": ["doc1"]})

        mock_registry = MagicMock()
        mock_registration = MagicMock()
        mock_registration.executor = mock_executor
        mock_registry.get.return_value = mock_registration
        mock_registry.validate_config.return_value = MagicMock()
        mock_registry.check_scopes.return_value = None

        node_spec = {
            "name": "search",
            "type": "tool",
            "tool_binding": "brain_search",
            "config": {},
        }

        with patch(
            "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
            return_value=mock_registry,
        ):
            executor = make_platform_node(node_spec, {})
            result = await executor({"__token__": MagicMock()}, {})

        # Default: wrapped in final_output
        assert "final_output" in result
        assert result["final_output"]["results"] == ["doc1"]


# ── 5.5: RAG Retrieval template v2.0 ───────────────────────────────


class TestRetrievalAugmentedTemplate:
    """retrieval_augmented.yaml v1.0 — unifying RAG and SQL analytics nodes."""

    def test_template_loads_successfully(self):
        """v1.0 template loads and validates via Pydantic."""
        from contextunity.router.cortex.compiler.template_loader import (
            load_template,
        )

        template = load_template("retrieval_augmented")
        assert template.name == "retrieval_augmented"
        assert template.version == "1.0"

    def test_all_nodes_have_router_tool_binding(self):
        """Every node must have a router_* tool_binding."""
        from contextunity.router.cortex.compiler.template_loader import (
            load_template,
        )

        template = load_template("retrieval_augmented")
        for node in template.nodes:
            assert node.tool_binding is not None, f"Node '{node.name}' missing tool_binding"
            assert node.tool_binding.startswith("router_"), (
                f"Node '{node.name}' binding '{node.tool_binding}' must start with router_"
            )


# ── 5.7: Override mechanism ─────────────────────────────────────────


class TestOverrideMechanism:
    """Consumer overrides propagate correctly through merge_overrides.

    Security invariants:
    - Cannot change node type (privilege escalation prevention)
    - Cannot inject phantom nodes
    - Cannot inject unsafe keys (name, type)
    """

    def _load_rag_template(self):
        from contextunity.router.cortex.compiler.template_loader import (
            load_template,
        )

        return load_template("retrieval_augmented")

    def test_model_swap_override(self):
        """Override model on generate node."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        merged = merge_overrides(
            template, {"generate": {"model": "anthropic/claude-sonnet-4-20250514"}}
        )

        gen_node = next(n for n in merged.nodes if n.name == "generate")
        assert gen_node.model == "anthropic/claude-sonnet-4-20250514"

        # Other nodes unchanged
        extract_node = next(n for n in merged.nodes if n.name == "extract_query")
        assert extract_node.model is None

    def test_retrieval_config_override(self):
        """Override retrieve node config (top_k, rerank)."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        merged = merge_overrides(
            template, {"retrieve": {"config": {"tool_config": {"top_k": 20, "rerank": False}}}}
        )

        ret_node = next(n for n in merged.nodes if n.name == "retrieve")
        assert ret_node.config.tool_config["top_k"] == 20
        assert ret_node.config.tool_config["rerank"] is False
        # output_mode preserved from base template
        assert ret_node.config.output_mode == "direct"

    def test_prompt_ref_override(self):
        """Override prompt_ref on detect_intent node."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        merged = merge_overrides(template, {"detect_intent": {"prompt_ref": "custom_intent_v2"}})

        intent_node = next(n for n in merged.nodes if n.name == "detect_intent")
        assert intent_node.prompt_ref == "custom_intent_v2"

    def test_type_escalation_blocked(self):
        """Cannot change node type via override — prevents privilege escalation."""
        from contextunity.core.exceptions import ConfigurationError

        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        with pytest.raises(ConfigurationError, match="Cannot change node type"):
            merge_overrides(template, {"generate": {"type": "llm"}})

    def test_phantom_node_blocked(self):
        """Cannot inject override for non-existent node."""
        from contextunity.core.exceptions import ConfigurationError

        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        with pytest.raises(ConfigurationError, match="non-existent"):
            merge_overrides(template, {"phantom_node": {"model": "evil/model"}})

    def test_unsafe_key_rejected(self):
        """Cannot inject 'name' key via override."""
        from contextunity.core.exceptions import ConfigurationError

        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        with pytest.raises(ConfigurationError, match="unsafe keys"):
            merge_overrides(template, {"generate": {"name": "evil_name"}})

    def test_multiple_node_overrides(self):
        """Override multiple nodes in single call."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        merged = merge_overrides(
            template,
            {
                "generate": {"model": "anthropic/claude-sonnet-4-20250514"},
                "retrieve": {"config": {"tool_config": {"top_k": 5}}},
                "reflect": {"model": "openai/gpt-4o-mini"},
            },
        )

        gen = next(n for n in merged.nodes if n.name == "generate")
        ret = next(n for n in merged.nodes if n.name == "retrieve")
        ref = next(n for n in merged.nodes if n.name == "reflect")

        assert gen.model == "anthropic/claude-sonnet-4-20250514"
        assert ret.config.tool_config["top_k"] == 5
        assert ref.model == "openai/gpt-4o-mini"

    def test_empty_overrides_returns_same_template(self):
        """Empty overrides dict returns template unchanged."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        merged = merge_overrides(template, {})
        assert merged == template

    def test_tool_binding_override(self):
        """Can override tool_binding (e.g., swap to custom brain_search variant)."""
        from contextunity.router.cortex.compiler.template_loader import (
            merge_overrides,
        )

        template = self._load_rag_template()
        merged = merge_overrides(template, {"retrieve": {"tool_binding": "router_custom_retrieve"}})

        ret = next(n for n in merged.nodes if n.name == "retrieve")
        assert ret.tool_binding == "router_custom_retrieve"
