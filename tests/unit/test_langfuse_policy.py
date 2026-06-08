"""Tests for LangfuseRequestCtx manifest policy resolution."""

from __future__ import annotations

from contextunity.router.modules.observability import LangfuseRequestCtx


def test_langfuse_ctx_from_metadata_respects_project_policy() -> None:
    # 1. No policy, request metadata present -> enabled=True (default baseline)
    ctx1 = LangfuseRequestCtx.from_metadata({"tenant_id": "some_tenant"})
    assert ctx1.enabled is True

    # 2. Manifest policy disables tracing
    meta_disabled = {"project_config": {"policy": {"langfuse": {"tracing_enabled": False}}}}
    ctx2 = LangfuseRequestCtx.from_metadata(meta_disabled)
    assert ctx2.enabled is False

    # 3. Request-level metadata overrides manifest policy
    meta_override = {
        "langfuse_enabled": True,
        "project_config": {"policy": {"langfuse": {"tracing_enabled": False}}},
    }
    ctx3 = LangfuseRequestCtx.from_metadata(meta_override)
    assert ctx3.enabled is True


def test_langfuse_ctx_from_metadata_coerces_strings() -> None:
    meta_str = {
        "langfuse_enabled": "false",
    }
    ctx = LangfuseRequestCtx.from_metadata(meta_str)
    assert ctx.enabled is False


def test_langfuse_ctx_ignores_request_controlled_credentials_and_host() -> None:
    ctx = LangfuseRequestCtx.from_metadata(
        {
            "langfuse_secret_key": "attacker-secret",
            "langfuse_public_key": "attacker-public",
            "langfuse_host": "http://attacker.invalid",
        }
    )

    assert ctx.secret_key == ""
    assert ctx.public_key == ""
    assert ctx.host == ""
