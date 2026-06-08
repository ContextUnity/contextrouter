"""Hash-match RegisterManifest must reuse stream_secret (H12)."""

from __future__ import annotations

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


class DummyService(RegistrationMixin):
    def __init__(self) -> None:
        self._project_tools = {"tenant_a": ["execute_test_sql"]}
        self._project_graphs = {
            "tenant_a": {
                "default": "project:tenant_a:my_graph",
                "my_graph": "project:tenant_a:my_graph",
            }
        }
        self._project_configs = {}
        self._stream_secrets = {}
        self._project_router_callbacks = {}
        from threading import Lock

        self._stream_secrets_lock = Lock()

    async def _persist_registration(self, pid: str, bundle: dict) -> None:
        _ = pid, bundle

    async def _check_manifest_hash(self, pid: str, h: str) -> bool:
        _ = pid, h
        return True

    async def _save_manifest_hash(self, pid: str, h: str) -> None:
        _ = pid, h

    async def _persist_stream_secret(self, pid: str, secret: str) -> None:
        _ = pid, secret


def _make_mock_token():
    from contextunity.core.tokens import ContextToken

    return ContextToken(
        token_id="reg-test",
        permissions=("tools:register:tenant_a",),
        allowed_tenants=("tenant_a",),
    )


def _make_context():
    from contextunity.core.signing import HmacBackend
    from contextunity.core.token_utils import serialize_token

    token = _make_mock_token()
    backend = HmacBackend("tenant_a", "bootstrap-secret")
    token_str = serialize_token(token, backend=backend)
    return _StubGrpcContext((("authorization", f"Bearer {token_str}"),))


@pytest.mark.asyncio
async def test_hash_match_reuses_existing_stream_secret_without_minting() -> None:
    """Idempotent path calls get_or_create — must not rotate active BiDi sessions."""
    existing_secret = "stable-stream-secret-existing"
    service = DummyService()

    mock_unit = MagicMock()
    mock_unit.payload = {
        "bundle": {
            "project_id": "tenant_a",
            "allowed_tenants": ["tenant_a"],
            "default_graph": "my_graph",
            "graph": {"my_graph": {"template": "yaml:retrieval_augmented"}},
            "tools": [],
            "policy": {},
        },
        "hash": "same-hash",
    }
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextunity.core import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register:tenant_a"])

    with (
        patch("contextunity.router.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextunity.core.discovery.verify_project_owner", return_value=True),
        patch("contextunity.core.discovery.register_project"),
        patch("contextunity.core.token_utils.verify_token_string", return_value=_make_mock_token()),
        patch(
            "contextunity.core.discovery.get_project_key",
            return_value={"project_secret": "mock_secret"},
        ),
        patch(
            "contextunity.core.discovery.get_or_create_project_stream_secret",
            return_value=existing_secret,
        ) as get_secret,
        patch("secrets.token_urlsafe") as mint_secret,
    ):
        response = await service.RegisterManifest(MagicMock(), _make_context())

    get_secret.assert_called_once_with("tenant_a")
    mint_secret.assert_not_called()

    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    assert payload["hash_matched"] is True
    assert payload["stream_secret"] == existing_secret
    assert service._stream_secrets["tenant_a"] == existing_secret
