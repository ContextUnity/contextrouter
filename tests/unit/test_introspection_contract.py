"""Tests for IntrospectRegistrations wire sanitizers (server-side)."""

from __future__ import annotations

from contextunity.router.service.introspection_contract import (
    sanitize_introspection_policy,
)


def test_sanitize_policy_uses_router_wire_keys() -> None:
    raw = {
        "models": {
            "llm": {"default": "gpt-test", "fallback": ["gpt-backup"]},
            "embeddings": {"default": "embed-model"},
        },
        "langfuse": {"tracing_enabled": True},
    }
    policy = sanitize_introspection_policy(raw)
    assert policy["default_llm"] == "gpt-test"
    assert policy["fallback_llms"] == ["gpt-backup"]
    assert policy["default_embeddings"] == "embed-model"
    assert policy["langfuse_tracing"] is True
