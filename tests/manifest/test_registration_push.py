from unittest.mock import MagicMock, patch

import grpc
import pytest

from contextrouter.service.mixins.registration import RegistrationMixin


class DummyService(RegistrationMixin):
    def __init__(self):
        self._project_tools = {}
        self._project_graphs = {}
        self._project_configs = {}
        self._stream_secrets = {}
        from threading import Lock

        self._stream_secrets_lock = Lock()

    async def _persist_registration(self, pid, bundle):
        pass

    async def _check_manifest_hash(self, pid, h):
        return getattr(self, "mock_hash_match", False)

    async def _save_manifest_hash(self, pid, h):
        pass

    def _register_graph(self, pid, config):
        self._project_graphs[pid] = config.name
        return config.name


@pytest.fixture
def test_bundle() -> dict:
    """Pre-compiled registration bundle (as output from ArtifactGenerator)."""
    return {
        "project_id": "tenant_a",
        "tenant_id": "tenant_a",
        "graph": {
            "id": "tenant_a",
            "template": "sql_analytics",
            "config": {
                "planner_prompt": "You are test analyst",
            },
        },
        "tools": [
            {
                "name": "execute_test_sql",
                "type": "sql",
                "description": "Execute SQL on medical DB",
                "config": {"read_only": True},
                "execution": "federated",
            }
        ],
        "policy": {
            "allowed_tools": ["execute_test_sql"],
        },
    }


def _make_mock_token():
    """Create a valid ContextToken for test security."""
    from contextcore.tokens import ContextToken

    return ContextToken(
        token_id="test-register",
        permissions=("tools:register", "tools:register:tenant_a"),
        allowed_tenants=("tenant_a",),
    )


def _make_context(project_secret: str | None = None):
    from contextcore.signing import HmacBackend
    from contextcore.token_utils import serialize_token

    token = _make_mock_token()
    backend = HmacBackend("tenant_a", project_secret or "bootstrap-secret")
    token_str = serialize_token(token, backend=backend)
    context = MagicMock()
    context.invocation_metadata.return_value = [("authorization", f"Bearer {token_str}")]
    return context


@pytest.mark.asyncio
async def test_register_manifest_success(test_bundle):
    service = DummyService()

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle, "hash": "abcd123"}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextcore import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register"])

    with (
        patch("contextrouter.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextcore.discovery.verify_project_owner", return_value=True),
        patch("contextcore.discovery.register_project"),
        patch("contextcore.token_utils.verify_token_string", return_value=_make_mock_token()),
        patch(
            "contextcore.discovery.get_project_key", return_value={"project_secret": "mock_secret"}
        ),
    ):
        response = await service.RegisterManifest(request=MagicMock(), context=_make_context())

        from google.protobuf.json_format import MessageToDict

        payload_dict = MessageToDict(response.payload)

        assert payload_dict["status"] == "ok"
        assert payload_dict["hash_matched"] is False
        assert payload_dict["graph"] == "tenant_a"
        assert "execute_test_sql" in payload_dict["registered_tools"]
        assert "execute_test_sql" in service._project_tools["tenant_a"]


@pytest.mark.asyncio
async def test_register_manifest_hash_match(test_bundle):
    service = DummyService()
    service.mock_hash_match = True

    # Pre-populate with existing tools so hash-match path has something to return
    service._project_tools["tenant_a"] = ["execute_test_sql"]
    service._project_graphs["tenant_a"] = "project:tenant_a:tenant_a"
    service._stream_secrets["tenant_a"] = "existing-secret"

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle, "hash": "abcd123"}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextcore import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register"])

    with (
        patch("contextrouter.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextcore.discovery.verify_project_owner", return_value=True),
        patch("contextcore.discovery.register_project"),
        patch("contextcore.token_utils.verify_token_string", return_value=_make_mock_token()),
        patch(
            "contextcore.discovery.get_project_key", return_value={"project_secret": "mock_secret"}
        ),
    ):
        # Hash matches so it should skip registration
        response = await service.RegisterManifest(request=MagicMock(), context=_make_context())
        from google.protobuf.json_format import MessageToDict

        payload_dict = MessageToDict(response.payload)
        assert payload_dict["hash_matched"] is True


@pytest.mark.asyncio
async def test_register_manifest_rejects_bundle_with_project_secret():
    """Security hardening: bundles containing project_secret are rejected."""
    service = DummyService()

    # Bundle with project_secret — should be rejected
    bad_bundle = {
        "project_id": "tenant_a",
        "tenant_id": "tenant_a",
        "graph": {"id": "tenant_a", "template": "sql_analytics", "config": {}},
        "tools": [],
        "policy": {},
        "project_secret": "leaked-secret",
    }

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": bad_bundle, "hash": "abcd123"}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextcore import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register"])

    with (
        patch("contextrouter.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextcore.token_utils.verify_token_string", return_value=_make_mock_token()),
    ):
        mock_context = _make_context()
        await service.RegisterManifest(request=MagicMock(), context=mock_context)

    # grpc_error_handler should set INTERNAL (ValueError is wrapped)
    mock_context.set_code.assert_called()


@pytest.mark.asyncio
async def test_register_manifest_no_token_rejects(test_bundle):
    """Security always enforced: no token → PermissionError."""
    service = DummyService()

    mock_unit = MagicMock()
    mock_unit.payload = {"bundle": test_bundle, "hash": "abcd123"}
    mock_unit.trace_id = __import__("uuid").uuid4()
    from contextcore import SecurityScopes

    mock_unit.security = SecurityScopes(write=["tools:register"])

    with (
        patch("contextrouter.service.mixins.registration.parse_unit", return_value=mock_unit),
        patch("contextrouter.service.helpers.parse_unit", return_value=mock_unit),
    ):
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = []
        await service.RegisterManifest(request=MagicMock(), context=mock_context)

        # grpc_error_handler sets PERMISSION_DENIED status
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)
