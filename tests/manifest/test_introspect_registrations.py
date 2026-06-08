"""Tests for IntrospectRegistrations RPC handler.

RED → GREEN → REFACTOR cycle.
Tests the IntrospectionMixin using the same DummyService pattern
established in test_registration_push.py.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from contextunity.core.exceptions import SecurityError

from contextunity.router.service.mixins.introspection import IntrospectionMixin


class DummyIntrospectionService(IntrospectionMixin):
    """Minimal service stub with in-memory state for testing."""

    def __init__(self):
        self._project_tools: dict[str, list[str]] = {}
        self._project_graphs: dict[str, dict[str, str]] = {}
        self._project_configs: dict[str, dict] = {}


@pytest.fixture
def populated_service() -> DummyIntrospectionService:
    """Service with one project registered (mirrors test_registration_push fixture)."""
    svc = DummyIntrospectionService()
    svc._project_tools["test_project"] = ["execute_test_sql"]
    svc._project_graphs["test_project"] = {
        "analytics": "project:test_project:analytics",
        "default": "project:test_project:analytics",
    }
    svc._project_configs["test_project"] = {
        "policy": {
            "models": {
                "llm": {
                    "default": "openai/gpt-5-mini",
                    "fallback": ["inception/mercury-2"],
                },
            },
            "langfuse": {"tracing_enabled": True},
        },
        "tools": [
            {"name": "execute_test_sql", "type": "sql", "description": "Execute SQL", "config": {}},
        ],
        "services": {
            "router": {"enabled": True},
            "brain": {"enabled": True},
        },
        "graph": {
            "analytics": {
                "nodes": [
                    {
                        "name": "planner",
                        "type": "llm",
                        "model": "openai/gpt-5-mini",
                        "pii_masking": True,
                    },
                    {
                        "name": "executor",
                        "type": "tool",
                        "tool_binding": "federated:execute_test_sql",
                    },
                    {"name": "verifier", "type": "llm", "model": "inception/mercury-2"},
                ],
                "edges": [
                    {"from_node": "__start__", "to_node": "planner"},
                    {"from_node": "planner", "to_node": "executor"},
                    {"from_node": "executor", "to_node": "verifier"},
                    {
                        "from_node": "verifier",
                        "condition_key": "valid",
                        "condition_map": {"true": "__end__", "false": "planner"},
                    },
                ],
            }
        },
    }
    return svc


@pytest.fixture
def introspect_token():
    """Token scoped to introspect ``test_project`` via project permission."""
    from contextunity.core.permissions import Permissions
    from contextunity.core.tokens import ContextToken

    return ContextToken(
        token_id="introspect-test",
        permissions=(
            Permissions.ROUTER_INTROSPECT,
            Permissions.introspect("test_project"),
        ),
        allowed_tenants=("clinical-tenant-a",),
    )


def _mock_unit(payload: dict | None = None):
    """Create a mock ContextUnit."""
    from contextunity.core import SecurityScopes

    unit = MagicMock()
    unit.payload = payload or {}
    unit.trace_id = __import__("uuid").uuid4()
    unit.security = SecurityScopes(read=["router:introspect"])
    return unit


def _validate_introspection_side_effect(token):
    """Apply project-scoped introspection checks without full gRPC auth stack."""

    def _validate(_unit, _context, *, project_id: str | None = None):
        from contextunity.core.permissions.access import has_introspection_access

        if project_id is not None and not has_introspection_access(token.permissions, project_id):
            raise SecurityError(f"Introspection denied for project '{project_id}'")
        return token

    return _validate


@contextmanager
def _introspect_auth(token, payload: dict | None = None):
    """Patch auth + parse_unit for IntrospectRegistrations tests."""
    unit = _mock_unit(payload)
    with (
        patch(
            "contextunity.router.service.mixins.introspection.parse_unit",
            return_value=unit,
        ),
        patch(
            "contextunity.router.service.mixins.introspection.validate_introspection_access",
            side_effect=_validate_introspection_side_effect(token),
        ),
    ):
        yield unit


@pytest.mark.asyncio
async def test_introspect_returns_all_projects(populated_service, introspect_token):
    """IntrospectRegistrations with no project_id returns visible registered projects."""
    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    assert "projects" in payload
    assert len(payload["projects"]) == 1
    assert payload["projects"][0]["project_id"] == "test_project"


@pytest.mark.asyncio
async def test_introspect_returns_specific_project(populated_service, introspect_token):
    """IntrospectRegistrations with project_id filter returns only that project."""
    with _introspect_auth(introspect_token, {"project_id": "test_project"}):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    assert len(payload["projects"]) == 1
    assert payload["projects"][0]["project_id"] == "test_project"


@pytest.mark.asyncio
async def test_introspect_nonexistent_project_returns_empty(populated_service, introspect_token):
    """Authorized project_id with no registration returns empty list."""
    from contextunity.core.permissions import Permissions
    from contextunity.core.tokens import ContextToken

    admin_token = ContextToken(
        token_id="admin-introspect",
        permissions=(Permissions.ADMIN_ALL, Permissions.ROUTER_INTROSPECT),
        allowed_tenants=("test_project",),
    )

    with _introspect_auth(admin_token, {"project_id": "nonexistent"}):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    assert payload["projects"] == []


@pytest.mark.asyncio
async def test_introspect_denied_for_unauthorized_project(populated_service, introspect_token):
    """Caller without project introspection permission cannot read a foreign project."""
    populated_service._project_tools["other_project"] = ["other_tool"]
    populated_service._project_configs["other_project"] = {"tools": [], "graph": {}}

    with _introspect_auth(introspect_token, {"project_id": "other_project"}):
        with pytest.raises(SecurityError, match="Introspection denied"):
            await populated_service.IntrospectRegistrations(
                request=MagicMock(), context=MagicMock()
            )


@pytest.mark.asyncio
async def test_introspect_scoped_to_visible_projects_only(populated_service, introspect_token):
    """List-all returns only projects the token may introspect."""
    populated_service._project_tools["secret_project"] = ["secret_tool"]
    populated_service._project_configs["secret_project"] = {"tools": [], "graph": {}}

    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    project_ids = {p["project_id"] for p in MessageToDict(response.payload)["projects"]}
    assert project_ids == {"test_project"}


@pytest.mark.asyncio
async def test_introspect_contains_tools(populated_service, introspect_token):
    """Response contains tool names and types."""
    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    project = MessageToDict(response.payload)["projects"][0]
    assert len(project["tools"]) == 1
    assert project["tools"][0]["name"] == "execute_test_sql"
    assert project["tools"][0]["type"] == "sql"


@pytest.mark.asyncio
async def test_introspect_contains_graph_structure(populated_service, introspect_token):
    """Response contains graph with sanitized nodes and edges."""
    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    project = MessageToDict(response.payload)["projects"][0]
    assert "analytics" in project["graphs"]
    graph = project["graphs"]["analytics"]

    node_names = [n["name"] for n in graph["nodes"]]
    assert "planner" in node_names
    assert "executor" in node_names
    assert "verifier" in node_names

    planner = next(n for n in graph["nodes"] if n["name"] == "planner")
    assert planner["model"] == "openai/gpt-5-mini"
    assert planner["pii_masking"] is True

    executor = next(n for n in graph["nodes"] if n["name"] == "executor")
    assert executor["tool_binding"] == "federated:execute_test_sql"

    assert len(graph["edges"]) == 4
    conditional = next(e for e in graph["edges"] if e.get("condition_key"))
    assert conditional["condition_key"] == "valid"
    assert conditional["condition_map"]["true"] == "__end__"


@pytest.mark.asyncio
async def test_introspect_contains_policy(populated_service, introspect_token):
    """Response contains sanitized policy with model names."""
    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    project = MessageToDict(response.payload)["projects"][0]
    policy = project["policy"]
    assert policy["default_llm"] == "openai/gpt-5-mini"
    assert "inception/mercury-2" in policy["fallback_llms"]
    assert policy["langfuse_tracing"] is True


@pytest.mark.asyncio
async def test_introspect_marks_default_graph(populated_service, introspect_token):
    """Default graph is marked with default=True."""
    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    project = MessageToDict(response.payload)["projects"][0]
    analytics = project["graphs"]["analytics"]
    assert analytics.get("default") is True


@pytest.mark.asyncio
async def test_introspect_strips_secrets(populated_service, introspect_token):
    """Response must NOT contain project_secret, api_keys, stream_secret, or model_secret_ref."""
    populated_service._project_configs["test_project"]["graph"]["analytics"]["nodes"].append(
        {
            "name": "secret_node",
            "type": "llm",
            "model": "secret/model",
            "model_secret_ref": "env:MY_SECRET",
        }
    )

    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    import json

    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    payload_str = json.dumps(payload)

    assert "model_secret_ref" not in payload_str
    assert "project_secret" not in payload_str
    assert "stream_secret" not in payload_str
    assert "api_keys" not in payload_str


@pytest.mark.asyncio
async def test_introspect_empty_service(introspect_token):
    """IntrospectRegistrations on empty service returns empty list."""
    svc = DummyIntrospectionService()

    with _introspect_auth(introspect_token):
        response = await svc.IntrospectRegistrations(request=MagicMock(), context=MagicMock())

    from google.protobuf.json_format import MessageToDict

    payload = MessageToDict(response.payload)
    assert payload["projects"] == []


@pytest.mark.asyncio
async def test_introspect_contains_services(populated_service, introspect_token):
    """Response contains enabled services."""
    with _introspect_auth(introspect_token):
        response = await populated_service.IntrospectRegistrations(
            request=MagicMock(), context=MagicMock()
        )

    from google.protobuf.json_format import MessageToDict

    project = MessageToDict(response.payload)["projects"][0]
    services = project["services"]
    assert services["router"]["enabled"] is True
    assert services["brain"]["enabled"] is True
    assert "zero" not in services
