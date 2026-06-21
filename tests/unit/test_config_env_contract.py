"""Router config environment contract regressions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from contextunity.router.core.config import get_core_config, reset_core_config
from contextunity.router.core.config.main import RouterConfig


def test_router_nested_env_overrides_do_not_wipe_siblings(monkeypatch):
    """Scalar env overrides must preserve sibling model keys.

    fallback_llms is now YAML-only (compound env var parsing removed).
    """
    reset_core_config()
    monkeypatch.setenv("CU_ROUTER_DEFAULT_LLM", "my-model")

    cfg = get_core_config()

    assert cfg.models.default_llm == "my-model"
    # fallback_llms is YAML-only; env var no longer parsed
    assert cfg.models.fallback_llms == []
    reset_core_config()


def test_router_brain_grpc_url_is_canonical(monkeypatch):
    """CU_BRAIN_GRPC_URL sets brain_url in SharedConfig (core factory)."""
    from contextunity.core.config import get_core_config as get_shared_config
    from contextunity.core.config import reset_core_config as reset_shared_config

    reset_core_config()
    reset_shared_config()
    monkeypatch.setenv("CU_BRAIN_GRPC_URL", "cu-url")

    shared = get_shared_config()
    assert shared.brain_url == "cu-url"

    reset_core_config()
    reset_shared_config()


def test_router_config_rejects_unknown_security_field() -> None:
    with pytest.raises(ValidationError, match="tls_require_client_authx"):
        RouterConfig.model_validate({"tls_require_client_authx": False})
