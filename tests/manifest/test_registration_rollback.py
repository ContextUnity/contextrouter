"""Registration rollback tests — partial failures must not leave stale state."""

from __future__ import annotations

from threading import Lock
from unittest.mock import MagicMock, patch

import pytest

from contextunity.router.service.mixins.registration import RegistrationMixin


class _StubGrpcContext:
    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self._metadata = metadata

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    def set_trailing_metadata(self, metadata: tuple[tuple[str, str], ...]) -> None:
        _ = metadata

    async def abort(self, code: object, details: str) -> None:
        _ = code, details

    def set_code(self, code: object) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details


class RollbackHarness(RegistrationMixin):
    """RegistrationMixin harness that can simulate graph registration failures."""

    def __init__(self) -> None:
        self._project_tools = {}
        self._project_graphs = {}
        self._project_configs = {"tenant_a": {"stale": True}}
        self._project_router_callbacks = {}
        self._stream_secrets = {}
        self._stream_secrets_lock = Lock()
        self._register_graph_attempts = 0
        self.fail_on_graph_attempt = 2
        self.deregister_calls = 0

    async def _persist_registration(self, pid, bundle) -> None:
        _ = pid, bundle

    async def _check_manifest_hash(self, pid, h) -> bool:
        _ = pid, h
        return False

    async def _save_manifest_hash(self, pid, h) -> None:
        _ = pid, h

    async def _persist_stream_secret(self, pid, secret) -> None:
        _ = pid, secret

    def _register_graph(self, project_id, graph_config, **kwargs):
        _ = kwargs
        self._register_graph_attempts += 1
        if self._register_graph_attempts >= self.fail_on_graph_attempt:
            raise RuntimeError("simulated graph registration failure")
        graph_key = graph_config.name
        graph_map = self._project_graphs.setdefault(project_id, {})
        reg_name = f"project:{project_id}:{graph_key}"
        graph_map[graph_key] = reg_name
        return reg_name

    def _deregister_project(self, project_id: str) -> list[str]:
        self.deregister_calls += 1
        tools = list(self._project_tools.get(project_id, []))
        self._project_tools.pop(project_id, None)
        self._project_graphs.pop(project_id, None)
        self._project_configs.pop(project_id, None)
        self._project_router_callbacks.pop(project_id, None)
        return tools


def _make_token():
    from contextunity.core.tokens import ContextToken

    return ContextToken(
        token_id="rollback-test",
        permissions=("tools:register:tenant_a",),
        allowed_tenants=("tenant_a",),
    )


def _make_context():
    from contextunity.core.signing import HmacBackend
    from contextunity.core.token_utils import serialize_token

    token = _make_token()
    backend = HmacBackend("tenant_a", "bootstrap-secret")
    token_str = serialize_token(token, backend=backend)
    return _StubGrpcContext((("authorization", f"Bearer {token_str}"),))


@pytest.fixture
def multi_graph_bundle() -> dict:
    return {
        "project_id": "tenant_a",
        "allowed_tenants": ["tenant_a"],
        "default_graph": "graph_a",
        "graph": {
            "graph_a": {
                "nodes": [{"name": "n1", "type": "llm", "model": "openai/gpt-5-mini"}],
                "edges": [
                    {"from_node": "__start__", "to_node": "n1"},
                    {"from_node": "n1", "to_node": "__end__"},
                ],
            },
            "graph_b": {
                "nodes": [{"name": "n2", "type": "llm", "model": "openai/gpt-5-mini"}],
                "edges": [
                    {"from_node": "__start__", "to_node": "n2"},
                    {"from_node": "n2", "to_node": "__end__"},
                ],
            },
        },
        "tools": [],
        "policy": {},
    }


@pytest.mark.asyncio
async def test_register_manifest_rolls_back_on_graph_failure(multi_graph_bundle):
    service = RollbackHarness()
    service._project_tools["tenant_a"] = ["old_tool"]

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": multi_graph_bundle, "hash": "hash-1"}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextunity.core import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register:tenant_a"])
    context = _make_context()

    with (
        patch("contextunity.router.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextunity.core.discovery.verify_project_owner", return_value=True),
        patch("contextunity.core.discovery.register_project"),
        patch("contextunity.core.discovery.update_project_stream_secret"),
        patch("contextunity.core.token_utils.verify_token_string", return_value=_make_token()),
        patch(
            "contextunity.core.discovery.get_project_key",
            return_value={"project_secret": "mock_secret"},
        ),
        patch("contextunity.router.modules.tools.register_tool"),
    ):
        await service.RegisterManifest(MagicMock(), context)

    assert service.deregister_calls >= 1
    assert service._project_tools["tenant_a"] == ["old_tool"]
    assert "tenant_a" not in service._project_graphs
    assert service._project_configs["tenant_a"] == {"stale": True}
    assert context._code is not None
