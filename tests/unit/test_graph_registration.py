"""Unit tests for graph registration manifest projection."""

import pytest
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.service.graph_registration import _project_manifest_config
from contextunity.router.service.registration_projection import coerce_compiler_node_spec


def test_project_manifest_config_promotes_nested_max_retries() -> None:
    """Nested config.max_retries must surface at manifest top level for cycle guards."""
    manifest = _project_manifest_config(
        {
            "model": "openai/gpt-5-mini",
            "config": {"max_retries": 5, "timeout": 30},
        }
    )
    assert manifest is not None
    assert manifest.get("max_retries") == 5
    assert manifest.get("timeout") == 30


def test_project_manifest_config_prefers_top_level_max_retries() -> None:
    """Explicit top-level max_retries wins over nested config value."""
    manifest = _project_manifest_config(
        {
            "max_retries": 2,
            "config": {"max_retries": 5},
        }
    )
    assert manifest is not None
    assert manifest.get("max_retries") == 2


def test_project_manifest_config_coerces_float_max_retries() -> None:
    """gRPC/protobuf may deliver whole-number floats — coerce to int."""
    manifest = _project_manifest_config({"max_retries": 2.0, "model_key": "openai/gpt-5-mini"})
    assert manifest is not None
    assert manifest.get("max_retries") == 2


def test_project_manifest_config_coerces_nested_float_max_retries() -> None:
    """Nested config.max_retries must coerce from float wire values."""
    manifest = _project_manifest_config({"config": {"max_retries": 5.0}})
    assert manifest is not None
    assert manifest.get("max_retries") == 5


def test_project_manifest_config_preserves_graph_prompt_config() -> None:
    """Resolved graph prompt keys must reach LLM executors through manifest config."""
    manifest = _project_manifest_config(
        {
            "max_retries": 2,
            "planner_prompt": "Generate SQL JSON.",
            "verifier_prompt": "Validate SQL JSON.",
        }
    )
    assert manifest is not None
    assert manifest.get("config") == {
        "max_retries": 2,
        "planner_prompt": "Generate SQL JSON.",
        "verifier_prompt": "Validate SQL JSON.",
    }


def test_project_manifest_config_merges_nested_config_without_losing_prompts() -> None:
    """Nested legacy config should not erase graph-only prompt keys."""
    manifest = _project_manifest_config(
        {
            "planner_prompt": "Generate SQL JSON.",
            "config": {"max_tokens": 4096, "max_retries": 1},
        }
    )
    assert manifest is not None
    assert manifest.get("config") == {
        "planner_prompt": "Generate SQL JSON.",
        "max_tokens": 4096,
        "max_retries": 1,
    }


def test_compiler_node_projection_rejects_invalid_node_type() -> None:
    with pytest.raises(ConfigurationError, match="Invalid node type"):
        coerce_compiler_node_spec({"name": "bad", "type": "worker"})


def test_compiler_node_projection_rejects_tool_bindings_without_type() -> None:
    with pytest.raises(ConfigurationError, match="without an explicit valid type"):
        coerce_compiler_node_spec({"name": "bad", "tool_binding": "federated:sql"})
