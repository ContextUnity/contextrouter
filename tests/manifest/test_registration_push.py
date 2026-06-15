from unittest.mock import MagicMock, patch

import grpc
import pytest

from contextunity.router.service.mixins.registration import RegistrationMixin


class _StubGrpcContext:
    """Minimal gRPC context for registration tests (satisfies ``GrpcServicerContext``)."""

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
    def __init__(self):
        self._project_tools = {}
        self._project_graphs = {}
        self._project_configs = {}
        self._stream_secrets = {}
        self._project_router_callbacks = {}
        from threading import Lock

        self._stream_secrets_lock = Lock()

    async def _persist_registration(self, pid, bundle):
        pass

    async def _check_manifest_hash(self, pid, h):
        return getattr(self, "mock_hash_match", False)

    async def _save_manifest_hash(self, pid, h):
        pass

    async def _persist_stream_secret(self, pid, secret):
        pass

    def _register_graph(self, pid, config, **kwargs):
        self._project_graphs[pid] = config.name
        return config.name


@pytest.fixture
def test_bundle() -> dict:
    """Pre-compiled registration bundle (as output from ArtifactGenerator)."""
    return {
        "project_id": "tenant_a",
        "allowed_tenants": ["tenant_a"],
        "default_graph": "my_graph",
        "graph": {
            "my_graph": {
                "nodes": [
                    {"name": "process", "type": "llm", "model": "openai/gpt-5-mini"},
                ],
                "edges": [
                    {"from_node": "__start__", "to_node": "process"},
                    {"from_node": "process", "to_node": "__end__"},
                ],
            }
        },
        "tools": [
            {
                "name": "execute_test_sql",
                "type": "sql",
                "description": "Execute SQL on medical DB",
                "config": {"read_only": True, "execution": "federated"},
            }
        ],
        "policy": {
            "allowed_tools": ["execute_test_sql"],
        },
    }


def _make_mock_token(permissions=("tools:register:tenant_a",)):
    """Create a valid ContextToken for test security."""
    from contextunity.core.tokens import ContextToken

    return ContextToken(
        token_id="test-register",
        permissions=permissions,
        allowed_tenants=("tenant_a",),
    )


def _make_context(project_secret: str | None = None):
    from contextunity.core.signing import HmacBackend
    from contextunity.core.token_utils import serialize_token

    token = _make_mock_token()
    backend = HmacBackend("tenant_a", project_secret or "bootstrap-secret")
    token_str = serialize_token(token, backend=backend)
    return _StubGrpcContext((("authorization", f"Bearer {token_str}"),))


@pytest.mark.asyncio
async def test_register_manifest_success(test_bundle):
    service = DummyService()

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle}
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
    ):
        response = await service.RegisterManifest(MagicMock(), _make_context())

        from google.protobuf.json_format import MessageToDict

        payload_dict = MessageToDict(response.payload)

        assert payload_dict["status"] == "ok"
        assert payload_dict["hash_matched"] is False
        assert payload_dict["graph"] == "my_graph"
        assert "execute_test_sql" in payload_dict["registered_tools"]
        assert "execute_test_sql" in service._project_tools["tenant_a"]


@pytest.mark.asyncio
async def test_register_manifest_hash_match(test_bundle):
    service = DummyService()
    service.mock_hash_match = True

    # Pre-populate with existing tools so hash-match path has something to return
    service._project_tools["tenant_a"] = ["execute_test_sql"]
    service._project_graphs["tenant_a"] = {
        "default": "project:tenant_a:tenant_a",
        "tenant_a": "project:tenant_a:tenant_a",
    }
    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle}
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
    ):
        # Hash matches so it should skip registration
        response = await service.RegisterManifest(MagicMock(), _make_context())
        from google.protobuf.json_format import MessageToDict

        payload_dict = MessageToDict(response.payload)
        assert payload_dict["hash_matched"] is True
        assert payload_dict["stream_secret"]
        assert service._stream_secrets["tenant_a"] == payload_dict["stream_secret"]


@pytest.mark.asyncio
async def test_register_manifest_rejects_bundle_with_project_secret():
    """Security hardening: bundles containing project_secret are rejected."""
    service = DummyService()

    # Bundle with project_secret — should be rejected
    bad_bundle = {
        "project_id": "tenant_a",
        "tenant_id": "tenant_a",
        "graph": {
            "tenant_a": {"id": "tenant_a", "template": "yaml:retrieval_augmented", "config": {}}
        },
        "tools": [],
        "policy": {},
        "project_secret": "leaked-secret",
    }

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": bad_bundle}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextunity.core import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register:tenant_a"])

    with (
        patch("contextunity.router.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextunity.core.token_utils.verify_token_string", return_value=_make_mock_token()),
    ):
        mock_context = _make_context()
        await service.RegisterManifest(MagicMock(), mock_context)

    # grpc_error_handler should set error code (SecurityError for secret leakage)
    assert mock_context._code is not None


@pytest.mark.asyncio
async def test_register_manifest_rejects_non_string_inline_secret(test_bundle):
    """Inline secrets must not be coerced from arbitrary JSON values."""
    service = DummyService()
    bad_bundle = dict(test_bundle)
    bad_bundle["secrets"] = {"CU_ROUTER_DEFAULT_MODEL_KEY": {"nested": "secret"}}

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": bad_bundle}
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
    ):
        mock_context = _make_context()
        await service.RegisterManifest(MagicMock(), mock_context)

    assert mock_context._code == grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.asyncio
async def test_register_manifest_empty_permission_token_rejects(test_bundle):
    """RegisterManifest requires tools:register:<project_id>."""
    service = DummyService()

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextunity.core import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register:tenant_a"])

    with (
        patch("contextunity.router.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextunity.core.discovery.verify_project_owner", return_value=True),
        patch(
            "contextunity.core.token_utils.verify_token_string", return_value=_make_mock_token(())
        ),
        patch(
            "contextunity.core.discovery.get_project_key",
            return_value={"project_secret": "mock_secret"},
        ),
    ):
        mock_context = _make_context()
        await service.RegisterManifest(MagicMock(), mock_context)

    assert mock_context._code == grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.asyncio
async def test_register_manifest_no_token_rejects(test_bundle):
    """Security always enforced: no token → PermissionError."""
    service = DummyService()

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextunity.core import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register:tenant_a"])

    with (
        patch("contextunity.router.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextunity.router.service.helpers.parse_unit", return_value=mock_unit),
    ):
        mock_context = _StubGrpcContext()
        await service.RegisterManifest(MagicMock(), mock_context)

        # grpc_error_handler sets PERMISSION_DENIED status
        assert mock_context._code == grpc.StatusCode.PERMISSION_DENIED
