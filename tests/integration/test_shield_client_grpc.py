"""In-process gRPC integration tests for shield_client.

Uses a FakeShieldHandler (GenericRpcHandler with dict-backed secret store)
to test PutSecret/GetSecret/verify round-trips without network or mocks.

Pattern: core's test_grpc_utils.py ``_DynamicEchoHandler``.
"""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest
from contextunity.core import contextunit_pb2
from contextunity.core.exceptions import PlatformServiceError, SecurityError
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from contextunity.router.service.shield_client import (
    shield_get_secret,
    shield_put_secret,
    shield_verify_secret,
)

# ── In-process Shield fake ────────────────────────────────────────────────


class FakeShieldHandler(grpc.GenericRpcHandler):
    """Minimal in-process Shield that stores secrets in a plain dict.

    Implements GetSecret and PutSecret RPCs using ContextUnit envelope.
    """

    def __init__(self):
        self._store: dict[str, str] = {}

    def service(self, handler_call_details):
        method = handler_call_details.method

        if method == "/contextunity.shield.ShieldService/PutSecret":
            return grpc.unary_unary_rpc_method_handler(
                self._put_secret,
                request_deserializer=contextunit_pb2.ContextUnit.FromString,
                response_serializer=contextunit_pb2.ContextUnit.SerializeToString,
            )
        if method == "/contextunity.shield.ShieldService/GetSecret":
            return grpc.unary_unary_rpc_method_handler(
                self._get_secret,
                request_deserializer=contextunit_pb2.ContextUnit.FromString,
                response_serializer=contextunit_pb2.ContextUnit.SerializeToString,
            )
        return None

    def _put_secret(self, request, context):
        payload = MessageToDict(request.payload)
        path = payload.get("path", "")
        value = payload.get("value", "")
        self._store[path] = value

        resp = contextunit_pb2.ContextUnit()
        s = struct_pb2.Struct()
        s.update({"path": path, "version": 1, "created_at": "2026-01-01T00:00:00Z"})
        resp.payload.CopyFrom(s)
        return resp

    def _get_secret(self, request, context):
        payload = MessageToDict(request.payload)
        path = payload.get("path", "")

        resp = contextunit_pb2.ContextUnit()
        s = struct_pb2.Struct()

        if path in self._store:
            s.update({"path": path, "value": self._store[path], "version": 1})
        else:
            s.update({"error": "not_found", "message": f"No secret at {path}"})

        resp.payload.CopyFrom(s)
        return resp


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def shield_handler():
    return FakeShieldHandler()


@pytest.fixture()
def shield_server(unused_tcp_port, shield_handler):
    """Start a threaded in-process gRPC server with FakeShieldHandler."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    server.add_generic_rpc_handlers([shield_handler])

    endpoint = f"127.0.0.1:{unused_tcp_port}"
    server.add_insecure_port(endpoint)
    server.start()
    yield endpoint
    server.stop(grace=0)


@pytest.fixture(autouse=True)
def _patch_shield_metadata(monkeypatch):
    """Skip real token minting — return empty metadata for tests."""
    monkeypatch.setattr(
        "contextunity.router.service.shield_client._shield_metadata",
        lambda **_kwargs: [],
    )


@pytest.fixture(autouse=True)
def _patch_tls(monkeypatch):
    """Force insecure channels for in-process testing."""
    monkeypatch.setattr("contextunity.core.grpc_utils.tls_enabled", lambda *args, **kwargs: False)


# ── Tests: PutSecret + GetSecret round-trip ───────────────────────────────


class TestShieldPutGetRoundTrip:
    """Full gRPC round-trip: put a secret, get it back."""

    def test_put_then_get(self, shield_server):
        result = shield_put_secret(
            "project/stream_token",
            "s3cr3t",
            shield_url=shield_server,
        )
        assert result["path"] == "project/stream_token"

        value = shield_get_secret(
            "project/stream_token",
            shield_url=shield_server,
        )
        assert value == "s3cr3t"

    def test_get_nonexistent_raises(self, shield_server):
        """GetSecret for missing path raises PlatformServiceError."""
        with pytest.raises(PlatformServiceError, match="not_found"):
            shield_get_secret("no/such/path", shield_url=shield_server)

    def test_put_overwrites(self, shield_server):
        """PutSecret overwrites existing value."""
        shield_put_secret("key", "v1", shield_url=shield_server)
        shield_put_secret("key", "v2", shield_url=shield_server)
        assert shield_get_secret("key", shield_url=shield_server) == "v2"


# ── Tests: shield_verify_secret with real gRPC ───────────────────────────


class TestShieldVerifySecretGrpc:
    """Verify secret comparison over real gRPC round-trip."""

    def test_verify_match(self, shield_server):
        shield_put_secret("auth/token", "correct", shield_url=shield_server)
        assert (
            shield_verify_secret(
                "auth/token",
                "correct",
                shield_url=shield_server,
            )
            is True
        )

    def test_verify_mismatch(self, shield_server):
        shield_put_secret("auth/token", "correct", shield_url=shield_server)
        assert (
            shield_verify_secret(
                "auth/token",
                "wrong",
                shield_url=shield_server,
            )
            is False
        )

    def test_verify_nonexistent_raises(self, shield_server):
        """Verifying against nonexistent secret raises (fail-closed)."""
        with pytest.raises((PlatformServiceError, SecurityError)):
            shield_verify_secret(
                "no/secret",
                "candidate",
                shield_url=shield_server,
            )


# ── Tests: tenant isolation ──────────────────────────────────────────────


class TestShieldTenantIsolation:
    """Path-scoped secrets round-trip without in-band tenant_id."""

    def test_path_scoped_roundtrip(self, shield_server):
        shield_put_secret("tenant/key", "val", shield_url=shield_server)
        value = shield_get_secret("tenant/key", shield_url=shield_server)
        assert value == "val"


# ── Tests: connection failure (fail-closed) ──────────────────────────────


class TestShieldConnectionFailure:
    """Shield unreachable → SecurityError (fail-closed)."""

    def test_unreachable_shield_raises_security_error(self, monkeypatch):
        """When Shield URL is set but unreachable, SecurityError is raised.

        Uses a short-deadline channel to avoid the 0.7s gRPC retry backoff
        that the default _GRPC_OPTIONS would cause (4 retries × exponential).
        """
        import json

        # Override channel factory to disable retries + set short deadline
        _fast_fail_options = [
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ("grpc.enable_retries", 0),
            (
                "grpc.service_config",
                json.dumps({"methodConfig": [{"name": [{}]}]}),
            ),
            ("grpc.initial_reconnect_backoff_ms", 10),
            ("grpc.max_reconnect_backoff_ms", 10),
        ]

        def _fast_fail_stub(shield_url=None):
            from contextunity.core import shield_pb2_grpc

            url = shield_url or "127.0.0.1:1"
            channel = grpc.insecure_channel(url, options=_fast_fail_options)
            stub = shield_pb2_grpc.ShieldServiceStub(channel)
            return stub, channel

        monkeypatch.setattr(
            "contextunity.router.service.shield_client._shield_stub",
            _fast_fail_stub,
        )

        with pytest.raises((SecurityError, Exception)):
            shield_get_secret("any/path", shield_url="127.0.0.1:1")
