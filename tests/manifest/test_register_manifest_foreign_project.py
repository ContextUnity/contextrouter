"""Registration must not allow cross-project takeover (H6)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from contextunity.core.permissions import Permissions

from contextunity.router.service.mixins.registration import RegistrationMixin


class _StubGrpcContext:
    """Minimal gRPC context (satisfies ``GrpcServicerContext`` for error handler)."""

    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self._metadata = metadata
        self.aborted = False

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    def set_trailing_metadata(self, metadata: tuple[tuple[str, str], ...]) -> None:
        _ = metadata

    async def abort(self, code: object, details: str) -> None:
        _ = code, details
        self.aborted = True

    def set_code(self, code: object) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details


class DummyService(RegistrationMixin):
    def __init__(self) -> None:
        self._project_tools: dict[str, list[str]] = {}
        self._project_graphs: dict[str, dict[str, str]] = {}
        self._project_configs: dict[str, dict] = {}
        self._stream_secrets: dict[str, str] = {}
        self._project_router_callbacks: dict[str, dict[str, list[str]]] = {}
        from threading import Lock

        self._stream_secrets_lock = Lock()

    async def _persist_registration(self, pid: str, bundle: dict) -> None:
        _ = pid, bundle

    async def _check_manifest_hash(self, pid: str, h: str) -> bool:
        _ = pid, h
        return False

    async def _save_manifest_hash(self, pid: str, h: str) -> None:
        _ = pid, h

    async def _persist_stream_secret(self, pid: str, secret: str) -> None:
        _ = pid, secret

    def _register_graph(self, pid: str, config, **kwargs) -> str:
        _ = kwargs
        return f"project:{pid}:{config.name}"


def _make_mock_token(permissions: tuple[str, ...]):
    from contextunity.core.tokens import ContextToken

    return ContextToken(
        token_id="reg-test",
        permissions=permissions,
        allowed_tenants=("tenant-x",),
    )


def _make_context(project_secret: str = "bootstrap-secret"):
    from contextunity.core.signing import HmacBackend
    from contextunity.core.token_utils import serialize_token

    # Kid matches target project_b, but permissions only cover project_a.
    token = _make_mock_token((Permissions.register("project_a"),))
    backend = HmacBackend("project_b", project_secret)
    token_str = serialize_token(token, backend=backend)
    return _StubGrpcContext((("authorization", f"Bearer {token_str}"),))


@pytest.mark.asyncio
async def test_register_manifest_rejects_token_scoped_to_other_project() -> None:
    """``tools:register:project_a`` cannot register a bundle for ``project_b``."""
    bundle = {
        "project_id": "project_b",
        "allowed_tenants": ["tenant-x"],
        "default_graph": "main",
        "graph": {
            "main": {
                "nodes": [{"name": "n1", "type": "llm", "model": "openai/gpt-5-mini"}],
                "edges": [{"from_node": "__start__", "to_node": "n1"}],
            }
        },
        "tools": [],
        "policy": {},
    }

    service = DummyService()
    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": bundle}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextunity.core import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register:project_a"])

    mock_context = _make_context()
    with (
        patch("contextunity.router.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextunity.core.discovery.verify_project_owner", return_value=True),
        patch("contextunity.core.discovery.register_project"),
        patch(
            "contextunity.core.token_utils.verify_token_string",
            return_value=_make_mock_token((Permissions.register("project_a"),)),
        ),
        patch(
            "contextunity.core.discovery.get_project_key",
            return_value={"project_secret": "mock_secret"},
        ),
    ):
        response = await service.RegisterManifest(MagicMock(), mock_context)

    assert mock_context._code is not None
    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    assert "error" in payload or payload.get("status") == "error"
