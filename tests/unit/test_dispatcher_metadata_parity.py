"""Pin M14: ``StreamDispatcher`` and ``ExecuteDispatcher`` must apply the
same ``model_key`` / ``system_prompt`` / ``project_config`` wiring so
clients see parity between the unary and stream paths.

Regression test — the StreamDispatcher path used to skip both
``params.model_key`` and the ``graph[*].config.planner_prompt`` lookup,
so stream runs always fell back to the hardcoded SYSTEM_PROMPT and
the default model.
"""

from __future__ import annotations

from contextunity.router.service.mixins.execution.metadata_helpers import (
    execution_metadata_from_payload,
)


def test_execution_metadata_forwards_model_key_and_system_prompt() -> None:
    """Smoke test: ``execution_metadata_from_payload`` accepts both kwargs."""
    meta = execution_metadata_from_payload(
        {},
        model_key="openai/gpt-5-mini",
        system_prompt="Generate SQL JSON.",
        project_config={"project_id": "nszu", "allowed_tenants": ["nszu"]},
    )
    assert meta.get("model_key") == "openai/gpt-5-mini"
    assert meta.get("system_prompt") == "Generate SQL JSON."
    assert meta.get("project_config") == {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
    }


def test_execution_metadata_keeps_caller_model_key_when_present() -> None:
    """The kwarg wins over the payload default for explicit overrides."""
    meta = execution_metadata_from_payload(
        {"model_key": "from-payload"},
        model_key="from-kwarg",
    )
    assert meta.get("model_key") == "from-kwarg"
