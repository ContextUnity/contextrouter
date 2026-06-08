"""Behavioral tests for service.mixins.execution.helpers.

Tests cover:
  - _resolve_tenant_id: token tenant extraction, fallback
  - build_execution_token: attenuation, sandbox identity, provenance bootstrap
  - extract_last_user_msg: dict/object messages, reversed search, fallback
  - extract_answer: message content extraction, truncation, empty
  - _redact_sensitive_keys: key suffix matching, nested dict/list
  - merge_token_usage: max() aggregation, empty/missing
  - _build_security_flags: shield mode, pii masking detection, error flag
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import ContextToken, TokenBuilder

from contextunity.router.service.mixins.execution.helpers import (
    _build_security_flags,
    _redact_sensitive_keys,
    _resolve_tenant_id,
    build_execution_token,
    build_run_config,
    extract_answer,
    extract_last_user_msg,
    get_registered_project_config,
    merge_token_usage,
    resolve_dispatcher_tenant_id,
    resolve_execution_project_id,
)
from contextunity.router.service.mixins.execution.types import coerce_graph_run_config_input
from contextunity.router.service.payloads import ExecuteAgentPayload
from contextunity.router.service.shield_check import ShieldCheckResult

# ── _resolve_tenant_id ────────────────────────────────────────────────────


class TestResolveTenantId:
    def test_returns_first_tenant(self):
        token = ContextToken(token_id="t", allowed_tenants=("acme", "beta"))
        assert _resolve_tenant_id(token) == "acme"

    def test_raises_when_no_tenants(self):
        token = ContextToken(token_id="t", allowed_tenants=())
        with pytest.raises(SecurityError, match="at least one tenant"):
            _resolve_tenant_id(token)

    def test_raises_when_token_none(self):
        with pytest.raises(SecurityError, match="at least one tenant"):
            _resolve_tenant_id(None)


# ── resolve_execution_project_id ──────────────────────────────────────────


class TestResolveExecutionProjectId:
    def test_from_resolved_registry_name(self):
        project_id = resolve_execution_project_id(
            graph_selector="analytics",
            resolved_graph_name="project:acme-proj:analytics",
        )
        assert project_id == "acme-proj"

    def test_from_project_selector(self):
        project_id = resolve_execution_project_id(
            graph_selector="project:acme-proj:analytics",
            resolved_graph_name="project:acme-proj:analytics",
        )
        assert project_id == "acme-proj"

    def test_raises_without_project_qualifier(self):
        with pytest.raises(SecurityError, match="project-qualified"):
            resolve_execution_project_id(
                graph_selector="bare_graph",
                resolved_graph_name="bare_graph",
            )


# ── resolve_dispatcher_tenant_id ──────────────────────────────────────────


class TestResolveDispatcherTenantId:
    def test_resolves_default_from_token(self):
        token = ContextToken(token_id="t", allowed_tenants=("acme",))
        assert resolve_dispatcher_tenant_id("default", token) == "acme"

    def test_preserves_explicit_tenant(self):
        token = ContextToken(token_id="t", allowed_tenants=("acme",))
        assert resolve_dispatcher_tenant_id("explicit-org", token) == "explicit-org"

    def test_raises_when_default_and_token_unscoped(self):
        token = ContextToken(token_id="t", allowed_tenants=())
        with pytest.raises(SecurityError, match="at least one tenant"):
            resolve_dispatcher_tenant_id("default", token)


# ── get_registered_project_config ─────────────────────────────────────────


class TestGetRegisteredProjectConfig:
    def test_explicit_project_id(self):
        configs = {"proj-a": {"policy": {"mode": "strict"}}}
        result = get_registered_project_config(configs, "org-xyz", project_id="proj-a")
        assert result == {"policy": {"mode": "strict"}}

    def test_tenant_fallback_when_project_id_omitted(self):
        """Default registration sets tenant_id=project_id; lookup by tenant works."""
        configs = {"acme": {"tools": []}}
        assert get_registered_project_config(configs, "acme") == {"tools": []}

    def test_distinct_tenant_does_not_alias_project_key(self):
        configs = {"proj-a": {"policy": {}}}
        assert get_registered_project_config(configs, "org-xyz") == {}
        assert get_registered_project_config(configs, "org-xyz", project_id="missing") == {}


# ── build_execution_token ─────────────────────────────────────────────────


class TestBuildExecutionToken:
    def test_none_token_returns_none(self):
        assert build_execution_token(None) is None

    def test_no_identity_returns_unchanged(self):
        token = ContextToken(token_id="t", user_id="u", permissions=("x",))
        result = build_execution_token(token)
        assert result.token_id == token.token_id

    def test_agent_id_sets_provenance(self):
        root = TokenBuilder().mint_root(
            user_ctx={"user_id": "user-1"},
            permissions=["tool:*"],
            ttl_s=3600,
        )
        result = build_execution_token(root, agent_id="my_graph")
        assert result.agent_id == "agent:my_graph"

    def test_platform_provenance_bootstrap(self):
        """Token without provenance gets platform origin injected."""
        token = ContextToken(token_id="t", user_id="u", permissions=("x",), provenance=())
        result = build_execution_token(token, platform="grpc")
        assert result.provenance and "grpc" in result.provenance

    def test_existing_provenance_not_overwritten(self):
        """Token WITH provenance does NOT get platform origin injected."""
        token = ContextToken(
            token_id="t",
            user_id="u",
            permissions=("x",),
            provenance=("upstream:origin",),
        )
        result = build_execution_token(token, platform="grpc", agent_id="g")
        # Original provenance preserved — platform NOT injected
        assert "upstream:origin" in result.provenance
        # grpc platform NOT in provenance (only injected when no existing provenance)
        assert "grpc" not in result.provenance


# ── extract_last_user_msg ─────────────────────────────────────────────────


class TestExtractLastUserMsg:
    def test_dict_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert extract_last_user_msg({"messages": msgs}) == "hello"

    def test_object_messages(self):
        msgs = [
            SimpleNamespace(role="user", content="first"),
            SimpleNamespace(role="user", content="last"),
        ]
        assert extract_last_user_msg({"messages": msgs}) == "last"

    def test_no_user_msg_returns_empty(self):
        msgs = [{"role": "system", "content": "init"}]
        assert extract_last_user_msg({"messages": msgs}) == ""

    def test_fallback_to_input_key(self):
        assert extract_last_user_msg({"input": "direct text"}) == "direct text"


# ── extract_answer ─────────────────────────────────────────────────────────


class TestExtractAnswer:
    def test_extracts_last_message_content(self):
        msgs = [SimpleNamespace(content="answer text")]
        assert extract_answer({"messages": msgs}) == "answer text"

    def test_truncates_long_content(self):
        msgs = [SimpleNamespace(content="x" * 10000)]
        result = extract_answer({"messages": msgs})
        assert len(result) == 5000

    def test_no_messages_returns_empty(self):
        assert extract_answer({}) == ""


# ── _redact_sensitive_keys ────────────────────────────────────────────────


class TestRedactSensitiveKeys:
    def test_redacts_signature_suffix(self):
        result = _redact_sensitive_keys({"prompt_signature": "abc123"})
        assert result["prompt_signature"] == "**REDACTED**"

    def test_preserves_normal_keys(self):
        result = _redact_sensitive_keys({"name": "test", "value": 42})
        assert result == {"name": "test", "value": 42}

    def test_handles_nested_dicts(self):
        result = _redact_sensitive_keys({"outer": {"inner_prompt": "secret", "safe": 1}})
        assert result["outer"]["inner_prompt"] == "**REDACTED**"
        assert result["outer"]["safe"] == 1

    def test_handles_lists(self):
        result = _redact_sensitive_keys([{"node_prompt": "x"}, {"name": "y"}])
        assert result[0]["node_prompt"] == "**REDACTED**"
        assert result[1]["name"] == "y"

    def test_passthrough_scalar(self):
        assert _redact_sensitive_keys("string") == "string"
        assert _redact_sensitive_keys(42) == 42


# ── merge_token_usage ─────────────────────────────────────────────────────


class TestMergeTokenUsage:
    def test_takes_max_of_both_sources(self):
        tracer = SimpleNamespace(
            get_token_usage=lambda: {"input_tokens": 100, "output_tokens": 50, "total_cost": 0.01}
        )
        state = {"_token_usage": {"input_tokens": 80, "output_tokens": 120, "total_cost": 0.02}}
        result = merge_token_usage(tracer, state)
        assert result["input_tokens"] == 100  # max(100, 80)
        assert result["output_tokens"] == 120  # max(50, 120)
        assert result["total_cost"] == 0.02  # max(0.01, 0.02)

    def test_empty_state_uses_tracer(self):
        tracer = SimpleNamespace(
            get_token_usage=lambda: {"input_tokens": 50, "output_tokens": 30, "total_cost": 0.005}
        )
        result = merge_token_usage(tracer, {})
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 30


# ── _build_security_flags ─────────────────────────────────────────────────


class TestBuildSecurityFlags:
    def test_shield_enabled_when_mode_is_shield(self):
        guard = ShieldCheckResult(blocked=False, reason="", mode="shield")
        flags = _build_security_flags(guard, [], "")
        assert flags["shield_enabled"] is True
        assert flags["shield_mode"] == "shield"

    def test_shield_disabled_when_passthrough(self):
        guard = ShieldCheckResult(blocked=False, reason="", mode="passthrough")
        flags = _build_security_flags(guard, [], "")
        assert flags["shield_enabled"] is False

    def test_pii_masking_enabled_when_provenance_has_pii(self):
        flags = _build_security_flags(None, ["node:planner", "pii:applied"], "")
        assert flags["pii_masking_enabled"] is True

    def test_pii_masking_enabled_when_provenance_has_privacy_pii_applied(self):
        flags = _build_security_flags(None, ["node:planner", "privacy:pii_applied"], "")
        assert flags["pii_masking_enabled"] is True

    def test_pii_masking_disabled_without_pii_provenance(self):
        flags = _build_security_flags(None, ["node:planner"], "")
        assert flags["pii_masking_enabled"] is False

    def test_error_flag_set(self):
        flags = _build_security_flags(None, [], "graph crashed")
        assert flags["error"] == "graph crashed"

    def test_none_guard_result_no_shield_keys(self):
        flags = _build_security_flags(None, [], "")
        assert "shield_enabled" not in flags


# ── resolve_graph ─────────────────────────────────────────────────────────


class TestResolveGraph:
    def test_multi_graph_default(self):
        """resolve_graph picks 'default' entry when no sub-graph is specified."""
        from unittest.mock import MagicMock, patch

        from contextunity.router.service.mixins.execution.helpers import resolve_graph

        project_graphs = {
            "multi_project": {
                "default": "project:multi_project:matcher",
                "matcher": "project:multi_project:matcher",
                "gardener": "project:multi_project:gardener",
            }
        }

        with patch(
            "contextunity.router.service.mixins.execution.helpers.graph_registry"
        ) as mock_reg:
            mock_reg.has.side_effect = lambda n: n.startswith("project:")
            mock_reg.get.return_value = lambda: MagicMock()

            resolved = resolve_graph("multi_project", "multi_project", project_graphs)
            assert resolved.name == "project:multi_project:matcher"

    def test_multi_graph_explicit_routing(self):
        """resolve_graph routes to explicit sub-graph when colon-separated."""
        from unittest.mock import MagicMock, patch

        from contextunity.router.service.mixins.execution.helpers import resolve_graph

        project_graphs = {
            "multi_project": {
                "default": "project:multi_project:matcher",
                "matcher": "project:multi_project:matcher",
                "gardener": "project:multi_project:gardener",
            }
        }

        with patch(
            "contextunity.router.service.mixins.execution.helpers.graph_registry"
        ) as mock_reg:
            mock_reg.has.side_effect = lambda n: n.startswith("project:")
            mock_reg.get.return_value = lambda: MagicMock()

            resolved = resolve_graph("multi_project:gardener", "multi_project", project_graphs)
            assert resolved.name == "project:multi_project:gardener"


# ── graph run config ──────────────────────────────────────────────────────


class TestGraphRunConfigInput:
    def test_coerce_ignores_unknown_keys(self):
        raw = {"configurable": {"thread_id": "t1"}, "callbacks": [], "extra": 1}
        coerced = coerce_graph_run_config_input(raw)
        assert coerced == {"configurable": {"thread_id": "t1"}}

    def test_coerce_normalizes_tags(self):
        coerced = coerce_graph_run_config_input({"tags": ["a", None, 2]})
        assert coerced == {"tags": ["a", "2"]}

    def test_payload_property_coerces_once(self):
        payload = ExecuteAgentPayload(
            agent_id="demo",
            input={"messages": []},
            config={"run_name": "trace-run", "recursion_limit": 25},
        )
        assert payload.graph_run_config == {"run_name": "trace-run", "recursion_limit": 25}

    def test_coerce_rejects_unbounded_runtime_knobs(self):
        raw = {
            "recursion_limit": 10_000,
            "max_concurrency": 10_000,
        }
        assert coerce_graph_run_config_input(raw) is None

    def test_coerce_rejects_bool_runtime_knobs(self):
        raw = {
            "recursion_limit": True,
            "max_concurrency": False,
        }
        assert coerce_graph_run_config_input(raw) is None

    def test_build_run_config_injects_callbacks(self):
        from langchain_core.callbacks.base import BaseCallbackHandler

        class _Cb(BaseCallbackHandler):
            pass

        callbacks = [_Cb()]
        cfg = build_run_config({"metadata": {"k": "v"}}, callbacks)
        assert cfg["callbacks"] is callbacks
        assert cfg["metadata"] == {"k": "v"}

    def test_build_run_config_applies_default_recursion_limit(self):
        cfg = build_run_config(None, [], default_recursion_limit=12)
        assert cfg["recursion_limit"] == 12

    def test_build_run_config_user_recursion_limit_can_lower_default(self):
        cfg = build_run_config({"recursion_limit": 8}, [], default_recursion_limit=12)
        assert cfg["recursion_limit"] == 8

    def test_build_run_config_manifest_recursion_limit_caps_user_limit(self):
        cfg = build_run_config({"recursion_limit": 50}, [], default_recursion_limit=12)
        assert cfg["recursion_limit"] == 12


class TestResolveRecursionLimit:
    def _graph(self, node_count: int):
        return SimpleNamespace(nodes={f"n{i}": object() for i in range(node_count)})

    def test_maps_nested_max_retries_to_limit(self):
        from contextunity.router.service.mixins.execution.helpers import (
            resolve_recursion_limit,
        )

        project_config = {"graph": {"analytics": {"config": {"max_retries": 2}}}}
        assert (
            resolve_recursion_limit(project_config, "project:nszu:analytics", self._graph(4)) == 14
        )

    def test_top_level_max_retries(self):
        from contextunity.router.service.mixins.execution.helpers import (
            resolve_recursion_limit,
        )

        project_config = {"graph": {"g": {"max_retries": 0}}}
        assert resolve_recursion_limit(project_config, "project:p:g", self._graph(3)) == 5

    def test_falls_back_to_sole_entry_when_key_missing(self):
        from contextunity.router.service.mixins.execution.helpers import (
            resolve_recursion_limit,
        )

        project_config = {"graph": {"only": {"config": {"max_retries": 1}}}}
        assert resolve_recursion_limit(project_config, "project:p:other", self._graph(2)) == 6

    def test_returns_none_without_max_retries(self):
        from contextunity.router.service.mixins.execution.helpers import (
            resolve_recursion_limit,
        )

        project_config = {"graph": {"g": {"config": {}}}}
        assert resolve_recursion_limit(project_config, "project:p:g", self._graph(3)) is None

    def test_returns_none_for_missing_config(self):
        from contextunity.router.service.mixins.execution.helpers import (
            resolve_recursion_limit,
        )

        assert resolve_recursion_limit(None, "project:p:g", self._graph(3)) is None


# ── execution_metadata_from_payload: allowed/denied tools ─────────────────


class TestExecutionMetadataToolLists:
    def test_allowed_tools_propagated_from_payload(self):
        """allowed_tools from wire payload must be captured in ExecutionMetadata."""
        from contextunity.router.service.mixins.execution.metadata_helpers import (
            execution_metadata_from_payload,
        )

        meta = execution_metadata_from_payload({"allowed_tools": ["sql", "search"]})
        assert meta.get("allowed_tools") == ["sql", "search"]

    def test_denied_tools_propagated_from_payload(self):
        """denied_tools from wire payload must be captured in ExecutionMetadata."""
        from contextunity.router.service.mixins.execution.metadata_helpers import (
            execution_metadata_from_payload,
        )

        meta = execution_metadata_from_payload({"denied_tools": ["dangerous_tool"]})
        assert meta.get("denied_tools") == ["dangerous_tool"]

    def test_missing_tool_lists_not_included(self):
        """Missing allowed/denied_tools does not inject empty lists."""
        from contextunity.router.service.mixins.execution.metadata_helpers import (
            execution_metadata_from_payload,
        )

        meta = execution_metadata_from_payload({"agent_id": "demo"})
        assert "allowed_tools" not in meta
        assert "denied_tools" not in meta

    def test_non_list_tool_fields_ignored(self):
        """Non-list values for allowed/denied_tools are silently dropped."""
        from contextunity.router.service.mixins.execution.metadata_helpers import (
            execution_metadata_from_payload,
        )

        meta = execution_metadata_from_payload({"allowed_tools": "sql", "denied_tools": 42})
        assert "allowed_tools" not in meta
        assert "denied_tools" not in meta
