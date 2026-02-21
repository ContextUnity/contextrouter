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

import inspect
import time

import pytest

# ============================================================================
# 1. Proto Compilation — ALL Service Protos
# ============================================================================


class TestProtoCompilationAll:
    """Every proto in contextcore/protos/ must compile and expose correct stubs."""

    def test_context_unit_proto(self):
        """context_unit.proto: ContextUnit message with all fields."""
        from contextcore import context_unit_pb2

        unit = context_unit_pb2.ContextUnit(unit_id="test-1")
        assert unit.unit_id == "test-1"
        assert hasattr(unit, "payload")
        assert hasattr(unit, "trace_id")
        assert hasattr(unit, "provenance")
        assert hasattr(unit, "chain_of_thought")
        assert hasattr(unit, "metrics")
        assert hasattr(unit, "security")
        assert hasattr(unit, "created_at")

    def test_brain_proto_rpcs(self):
        """brain.proto: BrainService with 12 RPCs."""
        from contextcore import brain_pb2_grpc

        servicer = brain_pb2_grpc.BrainServiceServicer()
        expected = [
            "Search",
            "GraphSearch",
            "CreateKGRelation",
            "Upsert",
            "QueryMemory",
            "UpsertNewsItem",
            "GetNewsItems",
            "UpsertNewsPost",
            "CheckNewsPostExists",
            "AddEpisode",
            "UpsertFact",
            "UpsertTaxonomy",
            "GetTaxonomy",
        ]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"BrainService missing RPC: {rpc}"

    def test_router_proto_rpcs(self):
        """router.proto: RouterService with 4 RPCs."""
        from contextcore import router_pb2_grpc

        servicer = router_pb2_grpc.RouterServiceServicer()
        expected = ["ExecuteAgent", "StreamAgent", "ExecuteDispatcher", "StreamDispatcher"]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"RouterService missing RPC: {rpc}"

    def test_commerce_proto_rpcs(self):
        """commerce.proto: CommerceService with 7 RPCs."""
        from contextcore import commerce_pb2_grpc

        servicer = commerce_pb2_grpc.CommerceServiceServicer()
        expected = [
            "GetProduct",
            "UpdateProduct",
            "GetProducts",
            "UpsertDealerProduct",
            "UpdateEnrichment",
            "TriggerHarvest",
            "GetPendingVerifications",
            "SubmitVerification",
        ]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"CommerceService missing RPC: {rpc}"

    def test_worker_proto_rpcs(self):
        """worker.proto: WorkerService with 3 RPCs."""
        from contextcore import worker_pb2_grpc

        servicer = worker_pb2_grpc.WorkerServiceServicer()
        expected = ["StartWorkflow", "GetTaskStatus", "ExecuteCode"]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"WorkerService missing RPC: {rpc}"

    def test_shield_proto_rpcs(self):
        """shield.proto: ShieldService with 8 RPCs."""
        from contextcore import shield_pb2_grpc

        servicer = shield_pb2_grpc.ShieldServiceServicer()
        expected = [
            "Scan",
            "EvaluatePolicy",
            "CheckCompliance",
            "RecordAudit",
            "MintToken",
            "VerifyToken",
            "RevokeToken",
            "GetStats",
        ]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"ShieldService missing RPC: {rpc}"

    def test_zero_proto_rpcs(self):
        """zero.proto: ZeroService with 6 RPCs."""
        from contextcore import zero_pb2_grpc

        servicer = zero_pb2_grpc.ZeroServiceServicer()
        expected = [
            "Anonymize",
            "Deanonymize",
            "ScanPII",
            "ProcessPrompt",
            "DestroySession",
            "GetStats",
        ]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"ZeroService missing RPC: {rpc}"

    def test_admin_proto_rpcs(self):
        """admin.proto: AdminService with 14 RPCs."""
        from contextcore import admin_pb2_grpc

        servicer = admin_pb2_grpc.AdminServiceServicer()
        expected = [
            "GetServiceHealth",
            "GetServiceMetrics",
            "CheckServiceConnectivity",
            "GetMemoryStats",
            "GetMemoryLayerStats",
            "ListAgents",
            "GetAgentConfig",
            "UpdateAgentPermissions",
            "UpdateAgentTools",
            "GetAgentActivity",
            "GetTraceDetails",
            "SearchTraces",
            "GetTraceChainOfThought",
            "GetSystemAnalytics",
            "GetErrorAnalytics",
            "DetectSystemErrors",
            "TriggerSelfHealing",
            "GetHealingStatus",
        ]
        for rpc in expected:
            assert hasattr(servicer, rpc), f"AdminService missing RPC: {rpc}"

    def test_modality_enum(self):
        """Modality enum must have TEXT, AUDIO, SPATIAL, IMAGE."""
        from contextcore import context_unit_pb2

        assert context_unit_pb2.TEXT == 0
        assert context_unit_pb2.AUDIO == 1
        assert context_unit_pb2.SPATIAL == 2
        assert context_unit_pb2.IMAGE == 3


# ============================================================================
# 2. ContextUnit Pydantic ↔ Protobuf Roundtrip
# ============================================================================


class TestContextUnitProtobufRoundtrip:
    """Ensure ContextUnit serialization is lossless."""

    def test_basic_roundtrip(self):
        """Create Pydantic → Protobuf → Pydantic and verify all fields."""
        from contextcore import context_unit_pb2
        from contextcore.sdk import ContextUnit, SecurityScopes

        original = ContextUnit(
            payload={"tenant_id": "nszu", "query": "test", "count": 42},
            provenance=["router:dispatcher", "brain:search"],
            security=SecurityScopes(read=["sql:select"], write=["product:patch"]),
        )

        # To protobuf
        pb = original.to_protobuf(context_unit_pb2)
        assert pb.unit_id == str(original.unit_id)
        assert pb.payload["tenant_id"] == "nszu"
        assert list(pb.provenance) == ["router:dispatcher", "brain:search"]
        assert list(pb.security.read) == ["sql:select"]
        assert list(pb.security.write) == ["product:patch"]

        # From protobuf
        restored = ContextUnit.from_protobuf(pb)
        assert restored.unit_id == original.unit_id
        assert restored.payload["tenant_id"] == "nszu"
        assert restored.payload["count"] == 42
        assert restored.provenance == original.provenance
        assert restored.security.read == original.security.read
        assert restored.security.write == original.security.write

    def test_empty_payload_roundtrip(self):
        """Empty payload should roundtrip cleanly."""
        from contextcore import context_unit_pb2
        from contextcore.sdk import ContextUnit

        original = ContextUnit()
        pb = original.to_protobuf(context_unit_pb2)
        restored = ContextUnit.from_protobuf(pb)
        assert restored.payload == {}

    def test_nested_payload_roundtrip(self):
        """Nested dicts in payload should roundtrip via Struct."""
        from contextcore import context_unit_pb2
        from contextcore.sdk import ContextUnit

        original = ContextUnit(
            payload={
                "metadata": {"source": "test", "version": 1},
                "tags": ["a", "b", "c"],
            }
        )
        pb = original.to_protobuf(context_unit_pb2)
        restored = ContextUnit.from_protobuf(pb)
        assert restored.payload["metadata"]["source"] == "test"
        assert restored.payload["tags"] == ["a", "b", "c"]

    def test_chain_of_thought_roundtrip(self):
        """CotStep chain should survive protobuf roundtrip."""
        from contextcore import context_unit_pb2
        from contextcore.sdk import ContextUnit
        from contextcore.sdk.models import CotStep

        original = ContextUnit(
            chain_of_thought=[
                CotStep(agent="planner", action="decompose", status="success"),
                CotStep(agent="executor", action="sql_query", status="pending"),
            ]
        )
        pb = original.to_protobuf(context_unit_pb2)
        restored = ContextUnit.from_protobuf(pb)
        assert len(restored.chain_of_thought) == 2
        assert restored.chain_of_thought[0].agent == "planner"
        assert restored.chain_of_thought[1].status == "pending"

    def test_metrics_roundtrip(self):
        """UnitMetrics should survive protobuf roundtrip."""
        from contextcore import context_unit_pb2
        from contextcore.sdk import ContextUnit
        from contextcore.sdk.models import UnitMetrics

        original = ContextUnit(
            metrics=UnitMetrics(latency_ms=42, cost_usd=0.005, tokens_used=150, cost_limit_usd=1.0)
        )
        pb = original.to_protobuf(context_unit_pb2)
        restored = ContextUnit.from_protobuf(pb)
        assert restored.metrics.latency_ms == 42
        assert restored.metrics.cost_usd == pytest.approx(0.005)
        assert restored.metrics.tokens_used == 150
        assert restored.metrics.cost_limit_usd == pytest.approx(1.0)


# ============================================================================
# 3. Shield API Contract — .check() NOT .scan()
# ============================================================================


class TestShieldAPIContract:
    """Verify Shield's public API matches documentation exactly."""

    def test_shield_has_check_method(self):
        """Shield.check() is the correct API — NOT .scan()."""
        from contextshield import Shield

        shield = Shield()
        assert hasattr(shield, "check"), "Shield must have .check() method"
        assert not hasattr(shield, "scan"), "Shield must NOT have .scan() — use .check()"

    def test_shield_check_signature(self):
        """Shield.check() must accept user_input and context params."""
        from contextshield import Shield

        sig = inspect.signature(Shield.check)
        params = list(sig.parameters.keys())
        assert "user_input" in params, "Shield.check must accept user_input"
        assert "context" in params, "Shield.check must accept context"

    def test_shield_check_returns_shield_result(self):
        """Shield.check() must return a ShieldResult."""
        from contextshield import Shield, ShieldResult

        shield = Shield()
        result = shield.check(user_input="Hello")
        assert isinstance(result, ShieldResult)

    def test_shield_result_has_correct_attributes(self):
        """ShieldResult must have allowed, blocked, reason, flags, latency_ms, severity."""
        from contextshield import Shield

        shield = Shield()
        result = shield.check(user_input="test")
        assert hasattr(result, "allowed")
        assert hasattr(result, "blocked")  # property
        assert hasattr(result, "reason")
        assert hasattr(result, "flags")
        assert hasattr(result, "latency_ms")
        assert hasattr(result, "severity")

    def test_shield_result_blocked_is_inverse_of_allowed(self):
        """ShieldResult.blocked must be the inverse of .allowed."""
        from contextshield import Shield

        shield = Shield()
        result = shield.check(user_input="Safe query")
        assert result.blocked == (not result.allowed)

    def test_shield_filter_or_raise_exists(self):
        """Shield.filter_or_raise() must exist as a convenience method."""
        from contextshield import Shield

        assert hasattr(Shield, "filter_or_raise")

    def test_shield_blocked_error_type(self):
        """ShieldBlockedError must be importable and be an Exception."""
        from contextshield import ShieldBlockedError

        assert issubclass(ShieldBlockedError, Exception)


# ============================================================================
# 4. SecurityGuard Contract — delegates to Shield.check()
# ============================================================================


class TestSecurityGuardContract:
    """Verify SecurityGuard wraps Shield correctly."""

    def test_guard_check_input_signature(self):
        """SecurityGuard.check_input() must accept user_input and related params."""
        from contextcore.security import SecurityGuard

        sig = inspect.signature(SecurityGuard.check_input)
        params = list(sig.parameters.keys())
        assert "user_input" in params
        assert "context_text" in params
        assert "rag_chunks" in params

    def test_guard_result_has_allowed_and_blocked(self):
        """GuardResult must have .allowed and .blocked properties."""
        from contextcore.security import GuardResult

        r = GuardResult(allowed=True)
        assert r.allowed is True
        assert r.blocked is False

        r2 = GuardResult(allowed=False, reason="test")
        assert not r2.allowed
        assert r2.blocked

    def test_guard_validate_token_signature(self):
        """SecurityGuard.validate_token() must accept context and require params."""
        from contextcore.security import SecurityGuard

        sig = inspect.signature(SecurityGuard.validate_token)
        params = list(sig.parameters.keys())
        assert "context" in params
        assert "require" in params

    def test_security_config_defaults(self):
        """SecurityConfig defaults: security OFF, shield ON, fail_open OFF."""
        from contextcore.security import SecurityConfig

        cfg = SecurityConfig()
        assert cfg.security_enabled is False
        assert cfg.shield_enabled is True
        assert cfg.fail_open is False

    def test_shield_status_returns_dict(self):
        """shield_status() must return a dict with shield_installed key."""
        from contextcore.security import shield_status

        status = shield_status()
        assert isinstance(status, dict)
        assert "shield_installed" in status
        assert "shield_active" in status
        assert "security_enabled" in status


# ============================================================================
# 5. Token Lifecycle — mint → serialize → parse → verify
# ============================================================================


class TestTokenLifecycle:
    """End-to-end token lifecycle across contextcore modules."""

    def test_mint_token_has_all_fields(self):
        """Minted token must have token_id, permissions, allowed_tenants, exp_unix, revocation_id."""
        from contextcore.tokens import TokenBuilder

        builder = TokenBuilder()
        token = builder.mint_root(
            user_ctx={"user_id": "u1"},
            permissions=["catalog:read", "brain:search"],
            ttl_s=3600,
            allowed_tenants=["nszu"],
        )
        assert token.token_id
        assert "catalog:read" in token.permissions
        assert "brain:search" in token.permissions
        assert "nszu" in token.allowed_tenants
        assert token.exp_unix is not None
        assert token.exp_unix > time.time()
        assert token.revocation_id is not None

    def test_serialize_and_parse_roundtrip(self):
        """serialize_token → parse_token_string must produce equivalent token."""
        from contextcore.token_utils import (
            parse_token_string,
            reset_default_backend,
            serialize_token,
        )
        from contextcore.tokens import TokenBuilder

        reset_default_backend()

        builder = TokenBuilder()
        original = builder.mint_root(
            user_ctx={},
            permissions=["test:read", "test:write"],
            ttl_s=3600,
            allowed_tenants=["traverse"],
        )

        wire = serialize_token(original)
        assert isinstance(wire, str)
        assert len(wire) > 10

        restored = parse_token_string(wire)
        assert restored is not None
        assert restored.token_id == original.token_id
        assert set(restored.permissions) == set(original.permissions)
        assert set(restored.allowed_tenants) == set(original.allowed_tenants)

    def test_serialize_wire_format_3_parts(self):
        """Wire format must be kid.payload.signature (3 dot-separated parts)."""
        from contextcore.token_utils import reset_default_backend, serialize_token
        from contextcore.tokens import TokenBuilder

        reset_default_backend()
        builder = TokenBuilder()
        token = builder.mint_root(user_ctx={}, permissions=["test"], ttl_s=60)
        wire = serialize_token(token)
        parts = wire.split(".")
        assert len(parts) == 3, f"Wire format must have 3 parts, got {len(parts)}: {wire}"

    def test_verify_valid_token(self):
        """TokenBuilder.verify() must pass for valid, non-expired token with correct permission."""
        from contextcore.tokens import TokenBuilder

        builder = TokenBuilder()
        token = builder.mint_root(user_ctx={}, permissions=["catalog:read"], ttl_s=3600)
        # Should not raise
        builder.verify(token, required_permission="catalog:read")

    def test_verify_rejects_wrong_permission(self):
        """TokenBuilder.verify() must raise PermissionError for wrong permission."""
        from contextcore.tokens import TokenBuilder

        builder = TokenBuilder()
        token = builder.mint_root(user_ctx={}, permissions=["catalog:read"], ttl_s=3600)
        with pytest.raises(PermissionError, match="Missing permission"):
            builder.verify(token, required_permission="catalog:write")

    def test_verify_rejects_expired_token(self):
        """TokenBuilder.verify() must raise PermissionError for expired token."""
        from contextcore.tokens import ContextToken, TokenBuilder

        builder = TokenBuilder()
        expired = ContextToken(token_id="exp", permissions=("test",), exp_unix=0.0)
        with pytest.raises(PermissionError, match="expired"):
            builder.verify(expired, required_permission="test")

    def test_attenuation_preserves_token_id(self):
        """Attenuated token keeps the same token_id for audit trail continuity."""
        from contextcore.tokens import TokenBuilder

        builder = TokenBuilder()
        root = builder.mint_root(user_ctx={}, permissions=["a", "b", "c"], ttl_s=3600)
        child = builder.attenuate(root, permissions=["a"])
        assert child.token_id == root.token_id
        assert child.has_permission("a")
        assert not child.has_permission("b")

    def test_attenuation_cannot_escalate_ttl(self):
        """Attenuated token can't have a longer TTL than the parent."""
        from contextcore.tokens import TokenBuilder

        builder = TokenBuilder()
        root = builder.mint_root(user_ctx={}, permissions=["a"], ttl_s=60)
        child = builder.attenuate(root, ttl_s=9999)
        # The child's exp must not exceed root's exp
        assert child.exp_unix is not None
        assert root.exp_unix is not None
        assert child.exp_unix <= root.exp_unix + 1.0  # 1s tolerance


# ============================================================================
# 6. Signing Backend Protocol
# ============================================================================


class TestSigningBackendProtocol:
    """Verify signing backends satisfy the protocol."""

    def test_unsigned_backend_satisfies_protocol(self):
        """UnsignedBackend must implement the SigningBackend protocol."""
        from contextcore.signing import SigningBackend, UnsignedBackend

        backend = UnsignedBackend()
        assert isinstance(backend, SigningBackend)

    def test_unsigned_backend_properties(self):
        """UnsignedBackend must have algorithm='none' and active_kid='unsigned'."""
        from contextcore.signing import UnsignedBackend

        backend = UnsignedBackend()
        assert backend.algorithm == "none"
        assert backend.active_kid == "unsigned"

    def test_unsigned_sign_and_verify(self):
        """UnsignedBackend sign → verify must return original payload."""
        from contextcore.signing import UnsignedBackend

        backend = UnsignedBackend()
        payload = b'{"token_id":"test","permissions":["a"]}'
        signed = backend.sign(payload)
        assert signed.kid == "unsigned"
        assert signed.algorithm == "none"

        wire = signed.serialize()
        restored = backend.verify(wire)
        assert restored == payload

    def test_signed_payload_wire_format(self):
        """SignedPayload.serialize() must produce kid.payload.signature format."""
        from contextcore.signing import SignedPayload

        sp = SignedPayload(payload="abc", signature="xyz", kid="key1", algorithm="test")
        wire = sp.serialize()
        assert wire == "key1.abc.xyz"

    def test_get_signing_backend_without_config_returns_unsigned(self):
        """get_signing_backend(None) must return UnsignedBackend."""
        from contextcore.signing import UnsignedBackend, get_signing_backend

        backend = get_signing_backend(None)
        assert isinstance(backend, UnsignedBackend)


# ============================================================================
# 7. PolicyEngine Condition Contracts
# ============================================================================


class TestPolicyEngineConditionContracts:
    """Verify every condition type evaluates correctly against ContextToken."""

    def test_permission_condition_exact_match(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import PermissionCondition

        cond = PermissionCondition("catalog:read")
        token = ContextToken(token_id="t1", permissions=("catalog:read",))
        assert cond.evaluate(token, {})

    def test_permission_condition_wildcard(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import PermissionCondition

        cond = PermissionCondition("admin:*")
        token = ContextToken(token_id="t1", permissions=("admin:users", "admin:config"))
        assert cond.evaluate(token, {})

    def test_permission_condition_no_match(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import PermissionCondition

        cond = PermissionCondition("catalog:write")
        token = ContextToken(token_id="t1", permissions=("catalog:read",))
        assert not cond.evaluate(token, {})

    def test_tenant_condition_literal(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import TenantCondition

        cond = TenantCondition("nszu")
        token = ContextToken(token_id="t1", allowed_tenants=("nszu",))
        assert cond.evaluate(token, {})

    def test_tenant_condition_context_reference(self):
        """TenantCondition('context.tenant_id') resolves from context dict."""
        from contextcore.tokens import ContextToken
        from contextshield.policy import TenantCondition

        cond = TenantCondition("context.tenant_id")
        token = ContextToken(token_id="t1", allowed_tenants=("nszu",))
        assert cond.evaluate(token, {"tenant_id": "nszu"})
        assert not cond.evaluate(token, {"tenant_id": "other"})

    def test_operation_condition(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import OperationCondition

        cond = OperationCondition("read")
        token = ContextToken(token_id="t1")
        assert cond.evaluate(token, {"operation": "read"})
        assert not cond.evaluate(token, {"operation": "write"})

    def test_context_condition_equals(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import ContextCondition

        cond = ContextCondition(field="source", equals="api")
        token = ContextToken(token_id="t1")
        assert cond.evaluate(token, {"source": "api"})
        assert not cond.evaluate(token, {"source": "web"})

    def test_context_condition_in_list(self):
        from contextcore.tokens import ContextToken
        from contextshield.policy import ContextCondition

        cond = ContextCondition(field="env", in_list=("prod", "staging"))
        token = ContextToken(token_id="t1")
        assert cond.evaluate(token, {"env": "prod"})
        assert not cond.evaluate(token, {"env": "dev"})

    def test_policy_engine_default_deny(self):
        """With no matching policies, default_effect='deny' must deny."""
        from contextcore.tokens import ContextToken
        from contextshield.policy import PolicyEngine

        engine = PolicyEngine([], default_effect="deny")
        token = ContextToken(token_id="t1", permissions=("test",))
        result = engine.evaluate(token)
        assert not result.allowed
        assert result.matched_policy is None

    def test_policy_engine_default_allow(self):
        """With no matching policies, default_effect='allow' must allow."""
        from contextcore.tokens import ContextToken
        from contextshield.policy import PolicyEngine

        engine = PolicyEngine([], default_effect="allow")
        token = ContextToken(token_id="t1")
        result = engine.evaluate(token)
        assert result.allowed

    def test_policy_engine_priority_ordering(self):
        """Higher priority policies should be evaluated first."""
        from contextcore.tokens import ContextToken
        from contextshield.policy import PermissionCondition, Policy, PolicyEngine

        engine = PolicyEngine(
            [
                Policy(
                    name="low-priority-deny",
                    effect="deny",
                    conditions=(PermissionCondition("test"),),
                    priority=1,
                ),
                Policy(
                    name="high-priority-allow",
                    effect="allow",
                    conditions=(PermissionCondition("test"),),
                    priority=10,
                ),
            ]
        )
        token = ContextToken(token_id="t1", permissions=("test",))
        result = engine.evaluate(token)
        assert result.allowed
        assert result.matched_policy == "high-priority-allow"


# ============================================================================
# 8. ContextZero ProxyService Contract
# ============================================================================


class TestProxyServiceContract:
    """Verify ProxyService API matches documented contract."""

    def test_proxy_service_has_all_methods(self):
        """ProxyService must have anonymize, deanonymize, destroy_session, get_stats, cleanup_expired_sessions."""
        from contextzero import ProxyService

        required_methods = [
            "anonymize",
            "deanonymize",
            "destroy_session",
            "get_stats",
            "cleanup_expired_sessions",
        ]
        for method in required_methods:
            assert hasattr(ProxyService, method), f"ProxyService missing method: {method}"

    def test_proxy_request_fields(self):
        """ProxyRequest must have prompt, session_id, persona_name, metadata."""
        from contextzero import ProxyRequest

        req = ProxyRequest(
            prompt="test", session_id="s1", persona_name="analyst", metadata={"key": "val"}
        )
        assert req.prompt == "test"
        assert req.session_id == "s1"
        assert req.persona_name == "analyst"
        assert req.metadata == {"key": "val"}

    def test_proxy_response_fields(self):
        """ProxyResponse must have anonymized_prompt, persona_injected, entities_masked, entity_types, session_id."""
        from contextzero import ProxyResponse

        resp = ProxyResponse(anonymized_prompt="test", entities_masked=3, entity_types=["phone"])
        assert resp.anonymized_prompt == "test"
        assert resp.entities_masked == 3
        assert resp.entity_types == ["phone"]
        assert resp.persona_injected is False

    def test_anonymize_returns_proxy_response(self):
        """ProxyService.anonymize() must return a ProxyResponse."""
        from contextzero import ProxyRequest, ProxyResponse, ProxyService
        from contextzero.masking import MaskingConfig

        svc = ProxyService.create(masking_config=MaskingConfig())
        resp = svc.anonymize(ProxyRequest(prompt="test text"))
        assert isinstance(resp, ProxyResponse)
        assert resp.session_id  # auto-generated if not provided

    def test_deanonymize_roundtrip(self):
        """anonymize → deanonymize must restore original text when no PII."""
        from contextzero import DeanonymizeRequest, ProxyRequest, ProxyService
        from contextzero.masking import MaskingConfig

        svc = ProxyService.create(masking_config=MaskingConfig())
        resp = svc.anonymize(ProxyRequest(prompt="no pii here", session_id="test-rt"))
        restored = svc.deanonymize(
            DeanonymizeRequest(text=resp.anonymized_prompt, session_id="test-rt")
        )
        assert "no pii here" in restored


# ============================================================================
# 9. Audit Trail Event Contract
# ============================================================================


class TestAuditEventContract:
    """Verify AuditEvent types cover full security lifecycle."""

    def test_all_event_types_exist(self):
        """AuditEventType must cover shield, token, policy, delegation, key, and pii events."""
        from contextshield.audit import AuditEventType

        required = [
            "SHIELD_CHECK",
            "SHIELD_BLOCK",
            "TOKEN_MINT",
            "TOKEN_VERIFY",
            "TOKEN_VERIFY_FAIL",
            "TOKEN_REVOKE",
            "POLICY_EVALUATE",
            "POLICY_DENY",
            "DELEGATION_CREATE",
            "DELEGATION_ATTENUATE",
            "KEY_ROTATE",
            "KEY_GENERATE",
            "PII_MASK",
            "PII_UNMASK",
            "PII_LEAK_DETECTED",
        ]
        for name in required:
            assert hasattr(AuditEventType, name), f"Missing AuditEventType: {name}"

    def test_event_serialization_round_trip(self):
        """AuditEvent.to_dict() must include event_type as string value."""
        import json

        from contextshield.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.TOKEN_MINT,
            actor="admin",
            tenant="nszu",
            details={"token_id": "t1"},
        )
        d = event.to_dict()
        assert d["event_type"] == "token.mint"
        assert d["actor"] == "admin"

        j = event.to_json()
        parsed = json.loads(j)
        assert parsed["event_type"] == "token.mint"

    def test_audit_trail_log_shield_check(self):
        """AuditTrail.log_shield_check() must accept all documented params."""
        from contextshield.audit import AuditTrail

        trail = AuditTrail(enabled=True)
        # Should not raise
        trail.log_shield_check(
            allowed=True,
            flags=["injection"],
            latency_ms=1.5,
            request_id="r1",
            actor="user",
            tenant="nszu",
            input_preview="test",
        )

    def test_audit_trail_log_token_operation(self):
        """AuditTrail.log_token_operation() must accept operation, token_id, actor, etc."""
        from contextshield.audit import AuditTrail

        trail = AuditTrail(enabled=True)
        trail.log_token_operation(
            "mint",
            token_id="tok-123",
            actor="admin",
            tenant="traverse",
            request_id="r2",
        )

    def test_audit_trail_log_policy_decision(self):
        """AuditTrail.log_policy_decision() must accept effect, policy_name, permissions."""
        from contextshield.audit import AuditTrail

        trail = AuditTrail(enabled=True)
        trail.log_policy_decision(
            effect="deny",
            policy_name="rate-limit",
            permissions=["write"],
            actor="user",
            tenant="nszu",
        )


# ============================================================================
# 10. Service Discovery Contract
# ============================================================================


class TestServiceDiscoveryContract:
    """Verify ServiceInfo type and discovery functions are correctly defined."""

    def test_service_info_fields(self):
        """ServiceInfo must have service, instance, endpoint, tenants, metadata."""
        from contextcore.discovery import ServiceInfo

        info = ServiceInfo(
            service="brain",
            instance="shared",
            endpoint="localhost:50051",
            tenants=["traverse", "pinkpony"],
            metadata={"version": "0.10.0"},
        )
        assert info.service == "brain"
        assert info.instance == "shared"
        assert info.endpoint == "localhost:50051"
        assert "traverse" in info.tenants

    def test_serves_tenant_shared(self):
        """Empty tenants list = shared service that serves all tenants."""
        from contextcore.discovery import ServiceInfo

        info = ServiceInfo(service="brain", instance="shared", endpoint="x", tenants=[])
        assert info.serves_tenant("nszu")
        assert info.serves_tenant("any-tenant")

    def test_serves_tenant_scoped(self):
        """Scoped service only serves listed tenants."""
        from contextcore.discovery import ServiceInfo

        info = ServiceInfo(service="brain", instance="nszu", endpoint="x", tenants=["nszu"])
        assert info.serves_tenant("nszu")
        assert not info.serves_tenant("traverse")

    def test_discovery_functions_exist(self):
        """All discovery functions must be importable from contextcore."""
        from contextcore import (
            deregister_service,
            discover_endpoints,
            discover_services,
            register_service,
        )

        assert callable(register_service)
        assert callable(deregister_service)
        assert callable(discover_services)
        assert callable(discover_endpoints)


# ============================================================================
# 11. Router Payload Contract
# ============================================================================


class TestRouterPayloadContract:
    """Verify Router service payload models match proto comments."""

    def test_execute_dispatcher_payload_fields(self):
        """ExecuteDispatcherPayload must have all documented fields (no tenant_id)."""
        from contextrouter.service.payloads import ExecuteDispatcherPayload

        payload = ExecuteDispatcherPayload(
            messages=[{"role": "user", "content": "hello"}],
        )
        # tenant_id removed — Token is SPOT
        assert "tenant_id" not in ExecuteDispatcherPayload.model_fields
        assert payload.session_id == "default"
        assert payload.platform == "grpc"
        assert payload.max_iterations == 10
        assert payload.metadata == {}
        assert payload.allowed_tools == []
        assert payload.denied_tools == []

    def test_execute_dispatcher_ignores_tenant_id(self):
        """Sending tenant_id in payload should be silently ignored (not raise)."""
        from contextrouter.service.payloads import ExecuteDispatcherPayload

        # Legacy callers may still send tenant_id — it must be silently dropped
        _payload = ExecuteDispatcherPayload(
            tenant_id="should_be_ignored",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert "tenant_id" not in ExecuteDispatcherPayload.model_fields

    def test_execute_dispatcher_validates_max_iterations(self):
        """max_iterations must be 1-50."""
        from pydantic import ValidationError

        from contextrouter.service.payloads import ExecuteDispatcherPayload

        with pytest.raises(ValidationError):
            ExecuteDispatcherPayload(messages=[], max_iterations=0)
        with pytest.raises(ValidationError):
            ExecuteDispatcherPayload(messages=[], max_iterations=100)

    def test_dispatcher_response_payload(self):
        """DispatcherResponsePayload must have messages, session_id, metadata."""
        from contextrouter.service.payloads import DispatcherResponsePayload

        resp = DispatcherResponsePayload(
            messages=[{"role": "assistant", "content": "hi"}],
            session_id="s1",
        )
        assert resp.messages[0]["content"] == "hi"
        assert resp.session_id == "s1"

    def test_stream_event_payload(self):
        """StreamDispatcherEventPayload must have event_type, data, timestamp."""
        from contextrouter.service.payloads import StreamDispatcherEventPayload

        event = StreamDispatcherEventPayload(
            event_type="agent_start",
            data={"agent": "dispatcher"},
        )
        assert event.event_type == "agent_start"
        assert event.timestamp is None


# ============================================================================
# 12. __init__.py Export Consistency
# ============================================================================


class TestExportConsistency:
    """Verify that all __all__ exports are actually importable."""

    def test_contextcore_exports(self):
        """Every item in contextcore.__all__ must be importable."""
        import contextcore

        for name in contextcore.__all__:
            assert hasattr(contextcore, name), (
                f"contextcore.__all__ lists '{name}' but it's not importable"
            )

    def test_contextshield_exports(self):
        """Every item in contextshield.__all__ must be importable."""
        import contextshield

        for name in contextshield.__all__:
            assert hasattr(contextshield, name), (
                f"contextshield.__all__ lists '{name}' but it's not importable"
            )

    def test_contextzero_exports(self):
        """Every item in contextzero.__all__ must be importable."""
        import contextzero

        for name in contextzero.__all__:
            assert hasattr(contextzero, name), (
                f"contextzero.__all__ lists '{name}' but it's not importable"
            )

    def test_contextrouter_exports(self):
        """contextrouter.__all__ must list all expected exports.

        Some exports require optional project-specific graphs (e.g., cortex.graphs.brain)
        which may not exist in all environments. We verify the __all__ list is defined
        and that non-lazy exports are accessible.
        """
        import contextrouter

        assert hasattr(contextrouter, "__all__")
        assert "__version__" in contextrouter.__all__
        # __version__ is always available (not lazy)
        assert contextrouter.__version__

        # Verify lazy exports are defined but may fail in test env
        lazy_exports = {
            "stream_agent",
            "invoke_agent",
            "invoke_dispatcher",
            "stream_dispatcher",
            "get_dispatcher_service",
            "get_langfuse_callbacks",
            "trace_context",
            "langfuse_flush",
        }
        for name in lazy_exports:
            assert name in contextrouter.__all__, f"'{name}' should be in contextrouter.__all__"

    def test_router_tools_exports(self):
        """Every item in contextrouter.modules.tools.__all__ must be importable."""
        from contextrouter.modules import tools

        for name in tools.__all__:
            assert hasattr(tools, name), f"tools.__all__ lists '{name}' but it's not importable"


# ============================================================================
# 13. Cross-Service Type Consistency
# ============================================================================


class TestCrossServiceTypeConsistency:
    """Verify shared types are the same across services."""

    def test_context_unit_is_same_class(self):
        """ContextUnit imported from contextcore.sdk must be the same class."""
        from contextcore import ContextUnit as CU1
        from contextcore.sdk import ContextUnit as CU2
        from contextcore.sdk.context_unit import ContextUnit as CU3

        assert CU1 is CU2
        assert CU2 is CU3

    def test_context_token_is_same_class(self):
        """ContextToken imported from contextcore must be same class as tokens module."""
        from contextcore import ContextToken as CT1
        from contextcore.tokens import ContextToken as CT2

        assert CT1 is CT2

    def test_security_scopes_is_same_class(self):
        """SecurityScopes imported from contextcore must be same class as models."""
        from contextcore import SecurityScopes as SS1
        from contextcore.sdk.models import SecurityScopes as SS2

        assert SS1 is SS2

    def test_shield_policy_uses_contextcore_token(self):
        """Shield's PolicyEngine must use the same ContextToken from contextcore."""
        from contextcore.tokens import ContextToken
        from contextshield.policy import PermissionCondition, Policy, PolicyEngine

        # Create a token from contextcore
        token = ContextToken(token_id="cross-svc", permissions=("test",))
        engine = PolicyEngine(
            [
                Policy(name="test", effect="allow", conditions=(PermissionCondition("test"),)),
            ]
        )
        # Must work — proves same class
        result = engine.evaluate(token)
        assert result.allowed


"""
    __import__("contextshield.policy").policy.PermissionCondition uses contextcore.tokens.ContextToken
    This test verifies the cross-service contract is intact.
"""


# ============================================================================
# 14. Token SPOT — tenant_id & user_id come ONLY from ContextToken
# ============================================================================


class TestTokenSPOTContract:
    """Token is the Single Point of Truth for identity.

    user_id, tenant_id — ONLY from token.
    Payload carries ONLY execution context (agent_id, input, config, platform).
    """

    # -- Payload must NOT contain identity fields --

    def test_execute_agent_payload_has_no_tenant_id(self):
        from contextrouter.service.payloads import ExecuteAgentPayload

        assert "tenant_id" not in ExecuteAgentPayload.model_fields, (
            "ExecuteAgentPayload must not have tenant_id — Token is SPOT"
        )

    def test_execute_agent_payload_required_fields_are_minimal(self):
        """Only agent_id and input are required. No identity fields."""
        from contextrouter.service.payloads import ExecuteAgentPayload

        required = {
            name for name, field in ExecuteAgentPayload.model_fields.items() if field.is_required()
        }
        assert required == {"agent_id", "input"}, (
            f"ExecuteAgentPayload required fields should be only agent_id+input, got {required}"
        )

    def test_execute_dispatcher_payload_required_fields(self):
        """Only messages is required. No identity fields."""
        from contextrouter.service.payloads import ExecuteDispatcherPayload

        required = {
            name
            for name, field in ExecuteDispatcherPayload.model_fields.items()
            if field.is_required()
        }
        assert required == {"messages"}, (
            f"ExecuteDispatcherPayload required fields should be only messages, got {required}"
        )

    def test_identity_fields_not_in_any_payload(self):
        """No payload should contain user_id or tenant_id as model fields."""
        from contextrouter.service.payloads import (
            ExecuteAgentPayload,
            ExecuteDispatcherPayload,
        )

        identity_fields = {"user_id", "tenant_id", "allowed_tenants", "permissions"}

        for model in (ExecuteAgentPayload, ExecuteDispatcherPayload):
            field_names = set(model.model_fields.keys())
            leaked = identity_fields & field_names
            assert not leaked, (
                f"{model.__name__} leaks identity fields: {leaked}. "
                "Identity must come from ContextToken only."
            )

    # -- _resolve_tenant_id always from token --

    def test_resolve_tenant_single(self):
        from unittest.mock import MagicMock

        from contextrouter.service.mixins.execution import _resolve_tenant_id

        token = MagicMock()
        token.allowed_tenants = ("med2lik",)
        assert _resolve_tenant_id(token) == "med2lik"

    def test_resolve_tenant_multi_takes_first(self):
        from unittest.mock import MagicMock

        from contextrouter.service.mixins.execution import _resolve_tenant_id

        token = MagicMock()
        token.allowed_tenants = ("clinic_kyiv", "clinic_lviv")
        assert _resolve_tenant_id(token) == "clinic_kyiv"

    def test_resolve_tenant_empty_returns_default(self):
        from unittest.mock import MagicMock

        from contextrouter.service.mixins.execution import _resolve_tenant_id

        token = MagicMock()
        token.allowed_tenants = ()
        assert _resolve_tenant_id(token) == "default"

    def test_resolve_tenant_none_token_returns_default(self):
        from contextrouter.service.mixins.execution import _resolve_tenant_id

        assert _resolve_tenant_id(None) == "default"

    def test_resolve_tenant_id_signature_has_only_token(self):
        """_resolve_tenant_id must accept ONLY token — no payload/params."""
        from contextrouter.service.mixins.execution import _resolve_tenant_id

        sig = inspect.signature(_resolve_tenant_id)
        param_names = list(sig.parameters.keys())
        assert param_names == ["token"], (
            f"_resolve_tenant_id must accept only 'token', got {param_names}"
        )

    # -- validate_dispatcher_access has no tenant_id --

    def test_validate_dispatcher_access_no_tenant_id_param(self):
        from contextrouter.service.security import validate_dispatcher_access

        sig = inspect.signature(validate_dispatcher_access)
        param_names = set(sig.parameters.keys())
        assert "tenant_id" not in param_names, (
            "validate_dispatcher_access must not accept tenant_id — "
            "tenant is derived from token, cross-validation is a tautology"
        )

    # -- ContextToken user_id and can_access_tenant --

    def test_token_user_id_is_set_correctly(self):
        from contextcore.tokens import ContextToken

        token = ContextToken(token_id="test", user_id="user@example.com")
        assert token.user_id == "user@example.com"

    def test_can_access_tenant_scoped(self):
        from contextcore.tokens import ContextToken

        token = ContextToken(token_id="test", allowed_tenants=("med2lik",))
        assert token.can_access_tenant("med2lik") is True
        assert token.can_access_tenant("other") is False

    def test_admin_token_accesses_all_tenants(self):
        """Empty allowed_tenants = admin token, can access anything."""
        from contextcore.tokens import ContextToken

        token = ContextToken(token_id="admin", allowed_tenants=())
        assert token.can_access_tenant("any_tenant") is True
