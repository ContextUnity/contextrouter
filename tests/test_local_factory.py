"""Tests for local Router service factory wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_local_router_passes_config_to_permission_interceptor(monkeypatch):
    from contextunity.router.service import interceptors, local

    cfg = SimpleNamespace(shield_url="shield.local:50054", port=55050)
    seen = {}

    class FakeInterceptor:
        def __init__(self, *, shield_url, config):
            seen["shield_url"] = shield_url
            seen["config"] = config

    class FakeServer:
        def __init__(self, *, interceptors):
            seen["interceptors"] = interceptors

        def add_insecure_port(self, endpoint):
            seen["endpoint"] = endpoint
            return 1

    monkeypatch.setattr("contextunity.router.core.config.get_core_config", lambda: cfg)
    monkeypatch.setattr(interceptors, "RouterPermissionInterceptor", FakeInterceptor)
    monkeypatch.setattr(
        local.grpc.aio, "server", lambda *, interceptors: FakeServer(interceptors=interceptors)
    )
    monkeypatch.setattr(local, "DispatcherService", lambda: object())
    monkeypatch.setattr(
        local.router_pb2_grpc, "add_RouterServiceServicer_to_server", lambda service, server: None
    )

    server = await local.create_local_router()

    assert isinstance(server, FakeServer)
    assert seen["shield_url"] == "shield.local:50054"
    assert seen["config"] is cfg
    assert seen["endpoint"] == "[::]:55050"
