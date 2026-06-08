"""Unit tests for security_tools: _use_rpc mode detection and SecurityResult schema.

gRPC integration tests are in tests/integration/test_security_tools_grpc.py.
"""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.modules.tools.schemas import SecurityResult


class TestUseRpcMode:
    def test_no_shield_url_returns_false(self, monkeypatch):
        """When shield_url is empty, _use_rpc returns False."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            "contextunity.router.core.get_core_config",
            lambda: SimpleNamespace(
                shield_url="",
                router=SimpleNamespace(brain_index_tools=False),
            ),
        )
        from contextunity.router.modules.tools.security_tools import _use_rpc

        assert _use_rpc() is False


class TestGrpcCallNoStub:
    def test_grpc_call_raises_when_no_stub(self, monkeypatch):
        """_grpc_call raises ConfigurationError when stub is None."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            "contextunity.router.core.get_core_config",
            lambda: SimpleNamespace(
                shield_url="",
                router=SimpleNamespace(brain_index_tools=False),
            ),
        )
        import contextunity.router.modules.tools.security_tools as module

        module._grpc_stub = None
        monkeypatch.setattr("contextunity.core.grpc_utils.tls_enabled", lambda: False)
        monkeypatch.setattr(
            "contextunity.router.service.shield_client._shield_metadata",
            lambda: [],
        )
        from contextunity.router.modules.tools.security_tools import _grpc_call

        with pytest.raises(ConfigurationError, match="not configured"):
            _grpc_call("Scan", {"text": "hello"})


class TestSecurityResult:
    def test_clean_result(self):
        result = SecurityResult(
            success=True,
            allowed=True,
            blocked=False,
            threats=[],
            risk_score=0.0,
            severity="none",
        )
        assert result["allowed"] is True
        assert result["blocked"] is False
        assert result["threats"] == []

    def test_blocked_result_with_threats(self):
        result = SecurityResult(
            success=True,
            allowed=False,
            blocked=True,
            threats=[{"validator": "injection", "reason": "test"}],
            risk_score=0.85,
            severity="critical",
        )
        assert result["blocked"] is True
        assert len(result["threats"]) == 1
        assert result["risk_score"] == 0.85

    def test_result_with_latency(self):
        result = SecurityResult(
            success=True,
            allowed=True,
            blocked=False,
            threats=[],
            risk_score=0.0,
            severity="none",
            latency_ms=3.14,
        )
        assert result["latency_ms"] == 3.14
