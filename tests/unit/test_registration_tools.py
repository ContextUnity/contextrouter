"""Tests for registration_tools strict parsing and project_id injection."""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.service.registration_tools import create_tools_from_bundle


def test_create_tools_injects_project_id_for_federated_sql():
    bundle = {
        "tools": [
            {
                "name": "execute_analytics_sql",
                "type": "sql",
                "description": "Run SQL",
                "config": {"read_only": True, "execution": "federated"},
            }
        ]
    }

    with pytest.MonkeyPatch.context() as mp:
        created: list[object] = []

        def _fake_create(**kwargs):
            created.append(kwargs)
            from langchain_core.tools import StructuredTool

            return [
                StructuredTool.from_function(
                    func=lambda: None,
                    name=kwargs["name"],
                    description=kwargs.get("description") or "",
                )
            ]

        mp.setattr(
            "contextunity.router.service.registration_tools.create_tool_from_config",
            _fake_create,
        )
        tools = create_tools_from_bundle(bundle, project_id="acme-proj")

    assert len(tools) == 1
    assert created[0]["config"]["project_id"] == "acme-proj"


def test_create_tools_rejects_mismatched_project_id():
    bundle = {
        "tools": [
            {
                "name": "execute_analytics_sql",
                "type": "sql",
                "description": "Run SQL",
                "config": {"project_id": "other-proj", "execution": "federated"},
            }
        ]
    }

    with pytest.raises(ConfigurationError, match="does not match bundle project_id"):
        create_tools_from_bundle(bundle, project_id="acme-proj")


def test_strict_tool_dicts_rejects_non_object_entries():
    bundle = {"tools": ["not-an-object"]}

    with pytest.raises(ConfigurationError, match="must be an object"):
        create_tools_from_bundle(bundle, project_id="acme-proj")
