"""Test the multi-graph manifest structures and validations (Phase 8)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from contextunity.router.service.payloads import GraphEntry, RegisterManifestPayload


def test_graph_entry_rejects_ambiguous_missing_source():
    """Graph entry requires one explicit source."""
    import pytest

    with pytest.raises(Exception, match="exactly one source"):
        GraphEntry()


def test_manifest_payload_graph_structures():
    """RegisterManifestPayload accepts single-graph and multi-graph bundles."""
    # Single graph (legacy)
    payload_single = RegisterManifestPayload(
        bundle={"graph": {"id": "my_rag", "template": "yaml:retrieval_augmented"}}
    )
    assert payload_single.bundle["graph"]["id"] == "my_rag"

    # Multi graphs (new)
    payload_multi = RegisterManifestPayload(
        bundle={
            "default_graph": "my_rag",
            "graph": {
                "my_rag": {"template": "yaml:retrieval_augmented"},
                "gardener": {
                    "template": "yaml:gardener",
                },
            },
        }
    )
    assert "graph" in payload_multi.bundle
    assert "gardener" in payload_multi.bundle["graph"]


@pytest.fixture
def registration_mixin():
    from contextunity.router.service.mixins.registration import RegistrationMixin

    mixin = RegistrationMixin.__new__(RegistrationMixin)
    mixin._project_graphs = {}
    mixin._project_configs = {}
    mixin._project_tools = {}
    mixin._stream_secrets = {}

    import threading

    mixin._stream_secrets_lock = threading.Lock()

    # Mock to avoid async auth dependencies
    mixin._check_manifest_hash = AsyncMock(return_value=False)
    mixin._persist_registration = AsyncMock()
    mixin._save_manifest_hash = MagicMock()
    mixin._persist_stream_secret = AsyncMock()
    return mixin


@patch("contextunity.router.cortex.compiler.builder.build_local_graph")
@patch("contextunity.router.service.mixins.registration.get_verified_registration_auth_context")
@patch("contextunity.core.authz.authorize")
@patch("contextunity.core.discovery.verify_project_owner")
@patch("contextunity.core.discovery.register_project")
@patch("contextunity.router.core.registry.graph_registry")
@patch("contextunity.router.service.mixins.registration.parse_unit")
@pytest.mark.asyncio
async def test_register_manifest_multi_graph(
    mock_parse_unit,
    mock_graph_register,
    mock_reg_proj,
    mock_verify,
    mock_authz,
    mock_get_ctx,
    mock_build,
    registration_mixin,
):
    """8.8 + 8.9: RegisterManifest iterates through 'graphs' map and registers all."""
    # Setup mocks
    mock_verify.return_value = True
    mock_authz.return_value = MagicMock(denied=False)
    from contextunity.core.tokens import ContextToken

    mock_get_ctx.return_value = ContextToken(
        token_id="multi-graph",
        permissions=("tools:register:multi_project",),
        allowed_tenants=("multi_project",),
    )

    request = MagicMock()
    import uuid

    from contextunity.core.sdk.models import SecurityScopes

    mock_unit = MagicMock()
    mock_unit.trace_id = uuid.uuid4()
    mock_unit.security = SecurityScopes()

    request = MagicMock()
    mock_parse_unit.return_value = mock_unit

    request.payload = {
        "bundle": {
            "project_id": "multi_project",
            "allowed_tenants": ["multi_project"],
            "default_graph": "matcher",
            "graph": {
                "matcher": {
                    "nodes": [{"name": "step1", "type": "llm"}],
                    "edges": [{"from_node": "__start__", "to_node": "step1"}],
                },
                "gardener": {
                    "nodes": [{"name": "classify", "type": "llm"}],
                    "edges": [{"from_node": "__start__", "to_node": "classify"}],
                },
            },
        }
    }
    mock_unit.payload = request.payload
    mock_parse_unit.return_value = mock_unit

    response = await registration_mixin.RegisterManifest(request, MagicMock())

    # Verification
    assert response is not None

    # Registry should be called twice, for project:multi_project:matcher and :gardener
    assert mock_graph_register.register.call_count == 2
    calls = mock_graph_register.register.call_args_list
    registered_names = [call[0][0] for call in calls]

    assert "project:multi_project:matcher" in registered_names
    assert "project:multi_project:gardener" in registered_names

    # Internal state holds dictionary of graphs for resolution
    assert "multi_project" in registration_mixin._project_graphs
    project_map = registration_mixin._project_graphs["multi_project"]

    # Should resolve default as fallback
    assert project_map["default"] == "project:multi_project:matcher"
    assert project_map["matcher"] == "project:multi_project:matcher"
    assert project_map["gardener"] == "project:multi_project:gardener"


def test_graph_entry_rejects_extra_fields():
    """8.1: GraphEntry with extra='forbid' rejects unknown fields."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GraphEntry(
            template="yaml:gardener",
            unknown_field="x",  # not a valid GraphEntry field
        )
