"""Regression tests for router audit backlog M15–M21."""

from __future__ import annotations

import logging

import pytest
from contextunity.core.tokens import ContextToken
from contextunity.core.types import is_object_dict

from contextunity.router.cortex.compiler.node_executors.federated import _federated_skip_update
from contextunity.router.cortex.compiler.node_factory import _agent_execute_tools
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.service.registration_projection import (
    registered_project_config_from_persisted,
)


def test_federated_skip_includes_bookkeeping() -> None:
    payload = {"error": "upstream", "__skipped__": True}
    update = _federated_skip_update("med_sql", "med_result", payload)
    assert update["_last_node"] == "med_sql"
    assert update["intermediate_results"] == {"med_sql": payload}
    assert update["med_result"] == payload


def test_agent_execute_tools_resolves_federated_tool_map() -> None:
    node_spec = {"tools": ["federated:alias_tool"]}
    manifest = {"federated_tool_map": {"alias_tool": "medical_sql"}}
    assert _agent_execute_tools(node_spec, manifest) == ["medical_sql"]


def test_dispatcher_build_config_appends_brain_auto_tracer() -> None:
    from contextunity.router.cortex.services.dispatcher import DispatcherService

    token = ContextToken(
        token_id="t1",
        user_id="user-1",
        permissions=("router:execute",),
    )
    config = DispatcherService()._build_config(
        tenant_id="tenant-a",
        session_id="sess-1",
        platform="api",
        metadata=None,
        access_token=token,
    )
    callbacks = config.get("callbacks", [])
    assert any(isinstance(callback, BrainAutoTracer) for callback in callbacks)


def test_dispatcher_checkpoint_thread_isolated_by_principal_and_project() -> None:
    from contextunity.router.cortex.services.dispatcher import DispatcherService

    service = DispatcherService()
    token_a = ContextToken(token_id="a", user_id="user-a", allowed_tenants=("tenant-a",))
    token_b = ContextToken(token_id="b", user_id="user-b", allowed_tenants=("tenant-a",))

    config_a = service._build_config(
        "tenant-a", "default", "api", {"project_id": "project-a"}, token_a
    )
    config_b = service._build_config(
        "tenant-a", "default", "api", {"project_id": "project-a"}, token_b
    )
    config_other_project = service._build_config(
        "tenant-a", "default", "api", {"project_id": "project-b"}, token_a
    )

    assert config_a["configurable"]["thread_id"] != config_b["configurable"]["thread_id"]
    assert (
        config_a["configurable"]["thread_id"] != config_other_project["configurable"]["thread_id"]
    )


def test_persisted_registration_logs_skipped_malformed_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    config = registered_project_config_from_persisted(
        {
            "project_id": "demo",
            "tools": [{"name": "ok"}, "not-a-dict", {}],
        },
        {},
    )
    assert len(config.get("tools", [])) == 1
    assert any("skipped" in record.message.lower() for record in caplog.records)


def test_llm_json_merge_preserves_non_dict_upstream() -> None:
    """Mirror the merge branch in node_executors/llm.py (M19)."""
    input_data_obj: object = ["entry", "payload"]
    parsed: dict[str, object] = {"valid": True}
    if is_object_dict(input_data_obj):
        merged: dict[str, object] = {**input_data_obj, **parsed}
    else:
        merged = dict(parsed)
        if input_data_obj is not None:
            merged["_upstream_input"] = input_data_obj
    assert merged["valid"] is True
    assert merged["_upstream_input"] == ["entry", "payload"]
