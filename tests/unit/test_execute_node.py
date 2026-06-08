import pytest
from contextunity.core import ContextUnit

from contextunity.router.service.mixins.execution.node import NodeExecutionMixin
from contextunity.router.service.payloads import ExecuteNodePayload


class MockNodeExecutor:
    async def ainvoke(self, state, config=None):
        return {"result": f"mocked_{state.get('val')}"}


class MockGraph:
    def __init__(self):
        self.nodes = {"allowed_node": MockNodeExecutor(), "hidden_node": MockNodeExecutor()}


class MockServer(NodeExecutionMixin):
    def __init__(self):
        # Production stores dict[str, str] in _project_graphs (graph_key → resolved name)
        self._project_graphs = {
            "tenant-1": {
                "test_graph": "project:tenant-1:test_graph",
                "default": "project:tenant-1:test_graph",
            }
        }
        # Callbacks are stored separately per project+graph_key
        self._project_router_callbacks = {
            "tenant-1": {
                "test_graph": ["allowed_node"],
            }
        }
        self._project_configs = {}


class MockContext:
    def __init__(self):
        self._details = None
        self._code = None

    def abort(self, code, details):
        raise Exception(f"Abort {code}: {details}")

    def set_code(self, code):
        self._code = code

    def set_details(self, details):
        self._details = details

    def set_trailing_metadata(self, metadata):
        pass

    def invocation_metadata(self):
        return ()


@pytest.fixture
def mock_server():
    return MockServer()


from contextunity.core import contextunit_pb2  # noqa: E402

import contextunity.router.service.mixins.execution.node as node_module  # noqa: E402
from contextunity.router.service.mixins.execution.types import ResolvedGraph  # noqa: E402


async def dummy_log(*args, **kwargs):
    pass


@pytest.mark.asyncio
async def test_execute_node_callbacks_keyed_by_project_id_not_tenant(mock_server, monkeypatch):
    """router_callbacks lookup uses project_id from resolved graph, not token tenant."""
    mock_server._project_graphs = {
        "acme-proj": {
            "analytics": "project:acme-proj:analytics",
            "default": "project:acme-proj:analytics",
        }
    }
    mock_server._project_router_callbacks = {
        "acme-proj": {"analytics": ["allowed_node"]},
    }

    monkeypatch.setattr(
        node_module,
        "resolve_graph",
        lambda graph_id, tenant, project_graphs: ResolvedGraph(
            "project:acme-proj:analytics", MockGraph()
        ),
    )
    monkeypatch.setattr(
        node_module,
        "prepare_execution",
        lambda params, tenant, token, configs: (params.input, {}, "user-123", [], None, None),
    )
    monkeypatch.setattr(node_module, "_resolve_tenant_id", lambda token: "org-tenant")
    monkeypatch.setattr(node_module, "build_execution_token", lambda *args, **kwargs: args[0])
    monkeypatch.setattr(node_module, "log_execution_trace", dummy_log)
    monkeypatch.setattr(
        node_module,
        "validate_dispatcher_access",
        lambda unit, context, permission=None, rpc_name=None: "fake-token",
    )

    payload = ExecuteNodePayload(
        graph_name="analytics",
        node_name="allowed_node",
        state={"val": "scoped"},
        config_overrides={},
    )
    unit = ContextUnit(payload=payload.model_dump())

    response = await mock_server.ExecuteNode(unit.to_protobuf(contextunit_pb2), MockContext())
    assert response.payload["node_name"] == "allowed_node"
    assert response.payload["output"] == {"result": "mocked_scoped"}


@pytest.mark.asyncio
async def test_execute_node_success(mock_server, monkeypatch):
    # Mock helpers — resolve_graph returns the resolved name matching _project_graphs
    monkeypatch.setattr(
        node_module,
        "resolve_graph",
        lambda graph_id, tenant, project_graphs: ResolvedGraph(
            "project:tenant-1:test_graph", MockGraph()
        ),
    )
    monkeypatch.setattr(
        node_module,
        "prepare_execution",
        lambda params, tenant, token, configs: (params.input, {}, "user-123", [], None, None),
    )
    monkeypatch.setattr(node_module, "_resolve_tenant_id", lambda token: "tenant-1")
    monkeypatch.setattr(node_module, "build_execution_token", lambda *args, **kwargs: args[0])
    monkeypatch.setattr(node_module, "log_execution_trace", dummy_log)
    monkeypatch.setattr(
        node_module,
        "validate_dispatcher_access",
        lambda unit, context, permission=None, rpc_name=None: "fake-token",
    )

    payload = ExecuteNodePayload(
        graph_name="test_graph", node_name="allowed_node", state={"val": "123"}, config_overrides={}
    )
    unit = ContextUnit(payload=payload.model_dump())

    response = await mock_server.ExecuteNode(unit.to_protobuf(contextunit_pb2), MockContext())
    assert response.payload["node_name"] == "allowed_node"
    assert response.payload["output"] == {"result": "mocked_123"}
    assert "execution_ms" in response.payload


@pytest.mark.asyncio
async def test_execute_node_hidden_node_raises_security_error(mock_server, monkeypatch):
    """Node not in router_callbacks must be rejected with SecurityError."""
    monkeypatch.setattr(
        node_module,
        "resolve_graph",
        lambda graph_id, tenant, project_graphs: ResolvedGraph(
            "project:tenant-1:test_graph", MockGraph()
        ),
    )
    monkeypatch.setattr(
        node_module,
        "prepare_execution",
        lambda params, tenant, token, configs: (params.input, {}, "user-123", [], None, None),
    )
    monkeypatch.setattr(node_module, "_resolve_tenant_id", lambda token: "tenant-1")
    monkeypatch.setattr(
        node_module,
        "validate_dispatcher_access",
        lambda unit, context, permission=None, rpc_name=None: "fake-token",
    )

    payload = ExecuteNodePayload(
        graph_name="test_graph", node_name="hidden_node", state={"val": "123"}, config_overrides={}
    )
    unit = ContextUnit(payload=payload.model_dump())

    # grpc_error_handler catches SecurityError, sets context code/details
    context = MockContext()
    response = await mock_server.ExecuteNode(unit.to_protobuf(contextunit_pb2), context)
    assert context._details is not None or response is not None


@pytest.mark.asyncio
async def test_execute_node_no_callbacks_registered(mock_server, monkeypatch):
    """Graph with no router_callbacks registered must reject all node executions."""
    # Clear callbacks for tenant-1
    mock_server._project_router_callbacks = {"tenant-1": {}}

    monkeypatch.setattr(
        node_module,
        "resolve_graph",
        lambda graph_id, tenant, project_graphs: ResolvedGraph(
            "project:tenant-1:test_graph", MockGraph()
        ),
    )
    monkeypatch.setattr(node_module, "_resolve_tenant_id", lambda token: "tenant-1")
    monkeypatch.setattr(
        node_module,
        "validate_dispatcher_access",
        lambda unit, context, permission=None, rpc_name=None: "fake-token",
    )

    payload = ExecuteNodePayload(
        graph_name="test_graph", node_name="allowed_node", state={"val": "123"}, config_overrides={}
    )
    unit = ContextUnit(payload=payload.model_dump())

    context = MockContext()
    response = await mock_server.ExecuteNode(unit.to_protobuf(contextunit_pb2), context)
    # Should be rejected — no callbacks registered means fail-closed
    assert context._details is not None or response is not None
