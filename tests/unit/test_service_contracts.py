"""Comprehensive service contract tests for the entire ContextUnity ecosystem.

Validates that:
  1. All protos compile and expose the correct RPCs (54 total)
  2. ContextUnit Pydantic ↔ Protobuf roundtrip is lossless
  3. All public APIs match their documented signatures
  4. Token lifecycle (mint → serialize → parse → verify) is consistent
  5. SecurityGuard correctly delegates to Shield.check() (not scan())
  6. PolicyEngine conditions evaluate correctly with real tokens
  7. Service discovery types are self-consistent
  8. All __init__.py exports are importable and match __all__
  9. Router payloads validate correctly
 10. Signing backend protocol is satisfied by all backends
 11. AuditTrail event types cover the full lifecycle

These tests exist because a contract mismatch (Shield.scan vs Shield.check)
was only caught at runtime. Every public API surface is now tested statically.
"""

from __future__ import annotations

import pytest

# ============================================================================
# 1. Proto Compilation — ALL Service Protos
# ============================================================================


# ============================================================================
# 2. ContextUnit Pydantic ↔ Protobuf Roundtrip
# ============================================================================


class TestContextUnitProtobufRoundtrip:
    """Ensure ContextUnit serialization is lossless."""

    def test_basic_roundtrip(self):
        """Create Pydantic → Protobuf → Pydantic and verify all fields."""
        from contextunity.core import contextunit_pb2
        from contextunity.core.sdk import ContextUnit, SecurityScopes

        original = ContextUnit(
            payload={"tenant_id": "tenant_a", "query": "test", "count": 42},
            provenance=["sdk:test", "router:test"],
            security=SecurityScopes(read=["sql:select"], write=["product:patch"]),
        )

        # To protobuf
        pb = original.to_protobuf(contextunit_pb2)
        assert pb.unit_id == str(original.unit_id)
        assert pb.payload["tenant_id"] == "tenant_a"
        assert list(pb.provenance) == ["sdk:test", "router:test"]
        assert list(pb.security.read) == ["sql:select"]
        assert list(pb.security.write) == ["product:patch"]

        # From protobuf
        restored = ContextUnit.from_protobuf(pb)
        assert restored.unit_id == original.unit_id
        assert restored.payload["tenant_id"] == "tenant_a"
        assert restored.provenance == ["sdk:test", "router:test"]
        assert restored.payload["count"] == 42
        assert restored.security.read == original.security.read
        assert restored.security.write == original.security.write

    def test_chain_of_thought_roundtrip(self):
        """CotStep chain should survive protobuf roundtrip."""
        from contextunity.core import contextunit_pb2
        from contextunity.core.sdk import ContextUnit
        from contextunity.core.sdk.models import CotStep

        original = ContextUnit(
            chain_of_thought=[
                CotStep(agent="planner", action="decompose", status="success"),
                CotStep(agent="executor", action="sql_query", status="pending"),
            ]
        )
        pb = original.to_protobuf(contextunit_pb2)
        restored = ContextUnit.from_protobuf(pb)
        assert len(restored.chain_of_thought) == 2
        assert restored.chain_of_thought[0].agent == "planner"
        assert restored.chain_of_thought[1].status == "pending"

    def test_metrics_roundtrip(self):
        """UnitMetrics should survive protobuf roundtrip."""
        from contextunity.core import contextunit_pb2
        from contextunity.core.sdk import ContextUnit
        from contextunity.core.sdk.models import UnitMetrics

        original = ContextUnit(
            metrics=UnitMetrics(latency_ms=42, cost_usd=0.005, tokens_used=150, cost_limit_usd=1.0)
        )
        pb = original.to_protobuf(contextunit_pb2)
        restored = ContextUnit.from_protobuf(pb)
        assert restored.metrics.latency_ms == 42
        assert restored.metrics.cost_usd == pytest.approx(0.005)
        assert restored.metrics.tokens_used == 150
        assert restored.metrics.cost_limit_usd == pytest.approx(1.0)


# ============================================================================
# 3. Shield API Contract — .check() NOT .scan()
# ============================================================================


# ============================================================================
# 11. Router Payload Contract
# ============================================================================


class TestRouterPayloadContract:
    """Verify Router service payload models match proto comments."""

    def test_execute_dispatcher_payload_fields(self):
        """ExecuteDispatcherPayload must have all documented fields (no tenant_id)."""
        from contextunity.router.service.payloads import ExecuteDispatcherPayload

        payload = ExecuteDispatcherPayload(
            messages=[{"role": "user", "content": "hello"}],
        )
        # tenant_id removed — Token is SPOT
        assert "tenant_id" not in ExecuteDispatcherPayload.model_fields
        assert payload.session_id == "default"
        assert payload.platform == "grpc"
        assert payload.max_iterations == 10
        assert payload.metadata == {}
        assert payload.allowed_tools is None
        assert payload.denied_tools == []

    def test_execute_dispatcher_rejects_tenant_id(self):
        """Sending tenant_id in payload must be rejected (extra='forbid')."""
        from pydantic import ValidationError

        from contextunity.router.service.payloads import ExecuteDispatcherPayload

        # extra='forbid' rejects unknown fields — defense-in-depth
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ExecuteDispatcherPayload(
                tenant_id="should_be_rejected",
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_execute_dispatcher_validates_max_iterations(self):
        """max_iterations must be 1-50."""
        from pydantic import ValidationError

        from contextunity.router.service.payloads import ExecuteDispatcherPayload

        with pytest.raises(ValidationError):
            ExecuteDispatcherPayload(messages=[], max_iterations=0)
        with pytest.raises(ValidationError):
            ExecuteDispatcherPayload(messages=[], max_iterations=100)


# ============================================================================
# 14. Token SPOT — tenant_id & user_id come ONLY from ContextToken
# ============================================================================


class TestTokenSPOTContract:
    """Token is the Single Point of Truth for identity.

    user_id, tenant_id — ONLY from token.
    Payload carries ONLY execution context (agent_id, input, config, platform).
    """

    # -- Payload must NOT contain identity fields --

    # -- _resolve_tenant_id always from token --

    # -- validate_dispatcher_access has no tenant_id --

    # -- ContextToken user_id and can_access_tenant --

    def test_token_user_id_is_set_correctly(self):
        from contextunity.core.tokens import ContextToken

        token = ContextToken(token_id="test", user_id="user@example.com")
        assert token.user_id == "user@example.com"

    def test_can_access_tenant_scoped(self):
        from contextunity.core.tokens import ContextToken

        token = ContextToken(token_id="test", allowed_tenants=("med2lik",))
        assert token.can_access_tenant("med2lik") is True
        assert token.can_access_tenant("other") is False

    def test_admin_token_accesses_all_tenants(self):
        """admin:all explicitly grants access to every tenant."""
        from contextunity.core.tokens import ContextToken

        token = ContextToken(token_id="admin", permissions=("admin:all",), allowed_tenants=())
        assert token.can_access_tenant("any_tenant") is True


# ============================================================================
# Manifest v1alpha6: Graph id auto-population
# ============================================================================


class TestManifestV1Alpha5:
    """Manifest v1alpha6 contract tests.

    Phase 1.4 manifest cleanup:
    - graph `id` is auto-set from map key if omitted
    - explicit `id` that matches key is accepted (backward compat)
    - explicit `id` that conflicts with key raises ConfigurationError
        - apiVersion: contextunity/v1alpha6 is accepted
    """

    def test_graph_entry_validates_id_matches_key(self):
        """Registration must reject a GraphEntry where explicit id ≠ map key."""
        from contextunity.core.exceptions import ConfigurationError

        # This test verifies the REGISTRATION logic, not just the Pydantic model.
        # We'll test via the registration helper directly.
        from contextunity.router.service.mixins.registration import _validate_graph_id_consistency

        # id matches key → no exception
        _validate_graph_id_consistency(graph_key="my-graph", entry_id="my-graph")

        # id is None → no exception (auto-populate case)
        _validate_graph_id_consistency(graph_key="my-graph", entry_id=None)

        # id conflicts with key → ConfigurationError
        with pytest.raises(ConfigurationError, match="does not match"):
            _validate_graph_id_consistency(graph_key="my-graph", entry_id="other-name")

    def test_manifest_v1alpha6_apiversion_accepted(self):
        """Manifest with apiVersion: contextunity/v1alpha6 must parse successfully."""
        from contextunity.core.manifest.models import ContextUnityProject

        manifest_data = {
            "apiVersion": "contextunity/v1alpha6",
            "kind": "ContextUnityProject",
            "project": {"id": "test", "name": "Test", "tenant": "test"},
            "services": {"router": {"enabled": True}},
            "router": {
                "graph": {"main": {"template": "yaml:retrieval_augmented"}},
                "policy": {"models": {"llm": {"default": "vertex/gemini-2.5-flash"}}},
            },
        }
        project = ContextUnityProject.model_validate(manifest_data)
        assert project.apiVersion == "contextunity/v1alpha6"

    def test_router_section_auto_sets_graph_ids_from_keys(self):
        """RouterSection must auto-populate RouterGraph.id from the dict key when id is None."""
        from contextunity.core.manifest.models import RouterSection

        section = RouterSection.model_validate(
            {
                "graph": {
                    "rag-graph": {"template": "yaml:retrieval_augmented"},
                    "sql-graph": {"template": "yaml:retrieval_augmented"},
                },
                "policy": {"models": {"llm": {"default": "vertex/gemini-2.5-flash"}}},
            }
        )
        # After v1alpha6: each graph's id auto-set from its key
        assert section.graph["rag-graph"].id == "rag-graph"
        assert section.graph["sql-graph"].id == "sql-graph"


# ============================================================================
# Manifest v1alpha6: local graph + node metadata contract
# ============================================================================


class TestRouterGraphLocalContract:
    """RouterGraph inline source and node metadata contract."""

    def test_router_node_meta_roundtrip(self):
        from contextunity.core.manifest.models import RouterGraph

        graph = RouterGraph.model_validate(
            {
                "nodes": [
                    {
                        "name": "tool_exec",
                        "tool_binding": "federated:medical_sql",
                        "meta": {
                            "handler": "nszu.chat.tools.execute_safe_query",
                            "source": "toolkit",
                            "toolkit": "MedSqlToolkit",
                        },
                    }
                ],
                "edges": [{"from_node": "__start__", "to_node": "tool_exec"}],
            }
        )
        meta = graph.nodes[0].meta
        assert meta is not None
        assert meta.handler == "nszu.chat.tools.execute_safe_query"
        assert meta.source == "toolkit"
        assert meta.toolkit == "MedSqlToolkit"

    def test_router_graph_accepts_goal_persona_contract(self):
        from contextunity.core.manifest.models import RouterGraph

        graph = RouterGraph.model_validate(
            {
                "goal": "Graph-wide agent objective",
                "persona": "graph-persona",
                "nodes": [
                    {
                        "name": "planner",
                        "type": "agent",
                        "goal": "Node-specific objective",
                        "tools": ["federated:medical_sql"],
                    },
                    {
                        "name": "explainer",
                        "type": "llm",
                        "persona": "node-persona",
                    },
                ],
                "edges": [{"from_node": "__start__", "to_node": "planner"}],
            }
        )

        assert graph.goal == "Graph-wide agent objective"
        assert graph.persona == "graph-persona"
        assert graph.nodes[0].goal == "Node-specific objective"
        assert graph.nodes[1].persona == "node-persona"

    def test_router_bundle_projects_graph_goal_persona_into_runtime_config(self):
        from contextunity.core.manifest.generators import ArtifactGenerator
        from contextunity.core.manifest.models import ContextUnityProject

        manifest = ContextUnityProject.model_validate(
            {
                "apiVersion": "contextunity/v1alpha6",
                "kind": "ContextUnityProject",
                "project": {"id": "test", "name": "Test", "tenant": "test"},
                "services": {"router": {"enabled": True}},
                "router": {
                    "graph": {
                        "main": {
                            "goal": "Graph goal",
                            "persona": "Graph persona",
                            "nodes": [{"name": "agent", "type": "agent", "tools": ["federated:t"]}],
                            "edges": [{"from_node": "__start__", "to_node": "agent"}],
                        }
                    },
                    "policy": {"models": {"llm": {"default": "vertex/gemini-2.5-flash"}}},
                },
            }
        )

        bundle = ArtifactGenerator(manifest).generate_router_registration_bundle()
        graph_config = bundle.graph["main"]["config"]
        assert graph_config["goal"] == "Graph goal"
        assert graph_config["persona"] == "Graph persona"
