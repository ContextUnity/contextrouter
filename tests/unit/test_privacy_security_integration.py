"""Tests for Privacy & Security integration.

Covers:
  1. ZeroServicer RPC end-to-end calls
  2. Tool discovery and registration
  3. Dual-mode tool operation (local + RPC)
  4. Compliance checker contract

Proto compilation, token lifecycle, policy engine, Shield API, and audit trail
tests are in test_service_contracts.py — not duplicated here.
"""

from __future__ import annotations

import uuid

import pytest

# ============================================================================
# 1. ZeroServicer Implementation
# ============================================================================


class FakeGrpcContext:
    """Minimal gRPC context stub for unit tests."""

    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _make_unit(payload: dict):
    from contextcore import ContextUnit, context_unit_pb2

    unit = ContextUnit(
        payload=payload,
        provenance=["test:privacy_security"],
    )
    return unit.to_protobuf(context_unit_pb2)


def _read_payload(unit_pb) -> dict:
    from contextcore import ContextUnit

    unit = ContextUnit.from_protobuf(unit_pb)
    return unit.payload or {}


@pytest.fixture
def zero_servicer():
    from contextzero.masking import MaskingConfig
    from contextzero.proxy import ProxyService
    from contextzero.service import ZeroServicer

    proxy = ProxyService.create(masking_config=MaskingConfig())
    return ZeroServicer(proxy=proxy)


class TestZeroServicerRPC:
    """Test all ZeroServicer RPCs end-to-end."""

    def test_anonymize_returns_session_id(self, zero_servicer):
        ctx = FakeGrpcContext()
        resp = zero_servicer.Anonymize(
            _make_unit({"prompt": "Лікар Петренко, тел +380501234567"}), ctx
        )
        data = _read_payload(resp)
        assert ctx.code is None
        assert data["session_id"]
        assert isinstance(data["entities_masked"], (int, float))

    def test_anonymize_rejects_empty_prompt(self, zero_servicer):
        ctx = FakeGrpcContext()
        zero_servicer.Anonymize(_make_unit({}), ctx)
        assert ctx.code is not None  # INVALID_ARGUMENT

    def test_deanonymize_roundtrip(self, zero_servicer):
        sid = f"test-{uuid.uuid4().hex[:6]}"
        ctx1 = FakeGrpcContext()
        resp1 = zero_servicer.Anonymize(
            _make_unit({"prompt": "Пацієнт Коваленко", "session_id": sid}), ctx1
        )
        anon_data = _read_payload(resp1)

        ctx2 = FakeGrpcContext()
        resp2 = zero_servicer.Deanonymize(
            _make_unit({"text": anon_data["anonymized_text"], "session_id": sid}), ctx2
        )
        deano = _read_payload(resp2)
        assert ctx2.code is None
        assert "restored_text" in deano

    def test_deanonymize_rejects_no_session(self, zero_servicer):
        ctx = FakeGrpcContext()
        zero_servicer.Deanonymize(_make_unit({"text": "hello"}), ctx)
        assert ctx.code is not None

    def test_scan_pii_returns_boolean(self, zero_servicer):
        ctx = FakeGrpcContext()
        resp = zero_servicer.ScanPII(_make_unit({"text": "just a test"}), ctx)
        data = _read_payload(resp)
        assert ctx.code is None
        assert isinstance(data["contains_pii"], bool)

    def test_destroy_session(self, zero_servicer):
        sid = f"destroy-{uuid.uuid4().hex[:6]}"
        ctx1 = FakeGrpcContext()
        zero_servicer.Anonymize(_make_unit({"prompt": "test", "session_id": sid}), ctx1)
        ctx2 = FakeGrpcContext()
        resp = zero_servicer.DestroySession(_make_unit({"session_id": sid}), ctx2)
        assert _read_payload(resp)["destroyed"] is True

    def test_get_stats(self, zero_servicer):
        ctx = FakeGrpcContext()
        resp = zero_servicer.GetStats(_make_unit({}), ctx)
        data = _read_payload(resp)
        assert ctx.code is None
        assert "active_sessions" in data


# ============================================================================
# 2. Tool Discovery and Registration
# ============================================================================


class TestToolDiscovery:
    """Verify tool registration system works correctly."""

    def test_register_tool_is_callable(self):
        from contextrouter.modules.tools import register_tool

        assert callable(register_tool)

    def test_privacy_tools_module_defines_all_tools(self):
        """Privacy tools module should define 5 tools in __all__."""
        from contextrouter.modules.tools import privacy_tools

        assert len(privacy_tools.__all__) == 5
        expected = {
            "anonymize_text",
            "deanonymize_text",
            "check_pii",
            "apply_persona",
            "destroy_privacy_session",
        }
        assert set(privacy_tools.__all__) == expected

    def test_security_tools_module_defines_all_tools(self):
        """Security tools module should define 4 tools in __all__."""
        from contextrouter.modules.tools import security_tools

        assert len(security_tools.__all__) == 4
        expected = {"shield_scan", "check_policy", "check_compliance", "audit_event"}
        assert set(security_tools.__all__) == expected

    def test_privacy_tools_are_langchain_tools(self):
        """Each privacy tool should be a LangChain StructuredTool."""
        from langchain_core.tools import BaseTool

        from contextrouter.modules.tools.privacy_tools import (
            anonymize_text,
            apply_persona,
            check_pii,
            deanonymize_text,
            destroy_privacy_session,
        )

        for t in [
            anonymize_text,
            deanonymize_text,
            check_pii,
            apply_persona,
            destroy_privacy_session,
        ]:
            assert isinstance(t, BaseTool), f"{t.name} is not a BaseTool"

    def test_security_tools_are_langchain_tools(self):
        """Each security tool should be a LangChain StructuredTool."""
        from langchain_core.tools import BaseTool

        from contextrouter.modules.tools.security_tools import (
            audit_event,
            check_compliance,
            check_policy,
            shield_scan,
        )

        for t in [shield_scan, check_policy, check_compliance, audit_event]:
            assert isinstance(t, BaseTool), f"{t.name} is not a BaseTool"

    def test_tools_have_descriptions(self):
        """All tools must have non-empty docstrings for the LLM."""
        from contextrouter.modules.tools.privacy_tools import _PRIVACY_TOOLS
        from contextrouter.modules.tools.security_tools import _SECURITY_TOOLS

        for t in _PRIVACY_TOOLS + _SECURITY_TOOLS:
            assert t.description, f"Tool '{t.name}' has no description"
            assert len(t.description) > 20, f"Tool '{t.name}' description too short"

    def test_privacy_tools_dual_mode_functions_exist(self):
        """Privacy tools should have _use_rpc() and _grpc_call() for dual-mode."""
        from contextrouter.modules.tools import privacy_tools

        assert hasattr(privacy_tools, "_use_rpc")
        assert hasattr(privacy_tools, "_grpc_call")
        assert callable(privacy_tools._use_rpc)

    def test_security_tools_dual_mode_functions_exist(self):
        """Security tools should have _use_rpc() and _grpc_call() for dual-mode."""
        from contextrouter.modules.tools import security_tools

        assert hasattr(security_tools, "_use_rpc")
        assert hasattr(security_tools, "_grpc_call")
        assert callable(security_tools._use_rpc)

    def test_rpc_mode_disabled_by_default(self):
        """Without env vars, tools should use local mode."""
        import os

        # Ensure env vars are not set
        os.environ.pop("CONTEXTZERO_GRPC_HOST", None)
        os.environ.pop("CONTEXTSHIELD_GRPC_HOST", None)

        from contextrouter.modules.tools.privacy_tools import _use_rpc as z_rpc
        from contextrouter.modules.tools.security_tools import _use_rpc as s_rpc

        assert not z_rpc()
        assert not s_rpc()


# ============================================================================
# 3. Compliance Checker
# ============================================================================


class TestComplianceIntegration:
    """Test ComplianceChecker produces structured reports."""

    def test_check_returns_report(self):
        from contextshield import ComplianceChecker

        checker = ComplianceChecker()
        report = checker.check()
        assert hasattr(report, "overall_score")
        assert hasattr(report, "findings")
        assert hasattr(report, "summary")
        assert 0 <= report.overall_score <= 100

    def test_compliance_summary_is_string(self):
        from contextshield import ComplianceChecker

        checker = ComplianceChecker()
        report = checker.check()
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
