"""Tests for telemetry helper functions.

Tests for _resolve_llm_display_name and _resolve_prompt_version
which are pure functions without infrastructure dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from contextunity.router.cortex.compiler.node_executors.telemetry import (
    _resolve_llm_display_name,
    _resolve_prompt_version,
)

# ═══════════════════════════════════════════════════════════════════
# _resolve_llm_display_name
# ═══════════════════════════════════════════════════════════════════


class TestResolveLlmDisplayName:
    """Resolution order: candidate_keys[0] → model_key → fallback → class name."""

    def test_model_key_used(self):
        llm = MagicMock()
        llm.model_key = "openai/gpt-4o"
        llm._candidate_keys = None
        assert _resolve_llm_display_name(llm) == "openai/gpt-4o"

    def test_candidate_keys_takes_priority(self):
        """FallbackModel with _candidate_keys prefers first candidate."""
        llm = MagicMock()
        llm._candidate_keys = ["gpt-4o", "claude-3"]
        llm.model_key = "fallback/gpt-4o/claude-3"
        assert _resolve_llm_display_name(llm) == "gpt-4o"

    def test_fallback_model_key_skipped(self):
        """model_key starting with 'fallback/' is skipped."""
        llm = MagicMock()
        llm._candidate_keys = None
        llm.model_key = "fallback/gpt-4o/claude-3"
        assert _resolve_llm_display_name(llm, "manifest-model") == "manifest-model"

    def test_fallback_name_used_when_no_model_key(self):
        llm = MagicMock()
        llm._candidate_keys = None
        llm.model_key = ""
        assert _resolve_llm_display_name(llm, "my-llm") == "my-llm"

    def test_class_name_as_last_resort(self):
        llm = MagicMock(spec=[])  # No model_key, no _candidate_keys
        llm.__class__ = type("CustomLLM", (), {})
        assert _resolve_llm_display_name(llm) == "CustomLLM"

    def test_empty_candidate_keys_ignored(self):
        llm = MagicMock()
        llm._candidate_keys = []
        llm.model_key = "openai/gpt-4o"
        assert _resolve_llm_display_name(llm) == "openai/gpt-4o"


# ═══════════════════════════════════════════════════════════════════
# _resolve_prompt_version
# ═══════════════════════════════════════════════════════════════════


class TestResolvePromptVersion:
    """Extracts prompt_version from state → metadata → project_config."""

    def test_returns_none_for_no_metadata(self):
        assert _resolve_prompt_version("node_a", {}) is None

    def test_returns_none_for_non_dict_metadata(self):
        assert _resolve_prompt_version("node_a", {"metadata": "string"}) is None

    def test_returns_none_for_no_project_config(self):
        assert _resolve_prompt_version("node_a", {"metadata": {}}) is None

    def test_returns_none_for_non_dict_project_config(self):
        state = {"metadata": {"project_config": "string"}}
        assert _resolve_prompt_version("node_a", state) is None

    def test_returns_prompt_version_when_present(self):
        state = {
            "metadata": {
                "project_config": {
                    "graph": {},
                    "nodes": [{"name": "node_a", "prompt_version": "v2.1"}],
                }
            }
        }
        result = _resolve_prompt_version("node_a", state)
        assert result == "v2.1"

    def test_returns_none_for_missing_node(self):
        state = {
            "metadata": {
                "project_config": {
                    "graph": {},
                    "nodes": [{"name": "other_node", "prompt_version": "v1"}],
                }
            }
        }
        assert _resolve_prompt_version("node_a", state) is None
