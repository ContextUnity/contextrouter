"""RED phase: Tests for PlatformToolRegistry.

Tests define the contract for the registry mechanism
before any implementation exists.
"""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import PlatformServiceError, SecurityError
from contextunity.core.tokens import ContextToken
from pydantic import BaseModel


class SampleToolConfig(BaseModel):
    """Sample config schema for testing."""

    top_k: int = 10
    collection: str = "default"


class BadToolConfig(BaseModel):
    """Config that requires a field."""

    required_field: str


# ── Registry Core ───────────────────────────────────────────────────


class TestPlatformToolRegistryCore:
    """Test basic register/get mechanics."""

    def test_register_and_get_tool(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def dummy_executor(state: dict, config: BaseModel) -> dict:
            return {"result": "ok"}

        registry.register(
            binding="brain_search",
            executor=dummy_executor,
            config_schema=SampleToolConfig,
            required_scopes=["brain:read"],
        )

        registration = registry.get("brain_search")
        assert registration is not None
        assert registration.executor is dummy_executor
        assert registration.config_schema is SampleToolConfig
        assert registration.required_scopes == ["brain:read"]
        assert registration.service_prefix == "brain"

    def test_get_unknown_binding_raises(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        with pytest.raises(PlatformServiceError, match="not registered"):
            registry.get("nonexistent_tool")

    def test_list_bindings(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])
        registry.register("shield_scan", noop, SampleToolConfig, ["shield:scan"])

        bindings = registry.list_bindings()
        assert "brain_search" in bindings
        assert "shield_scan" in bindings

    def test_duplicate_registration_raises(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

        with pytest.raises(PlatformServiceError, match="already registered"):
            registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

    def test_service_prefix_extracted_from_binding(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("shield_scan", noop, SampleToolConfig, ["shield:scan"])
        reg = registry.get("shield_scan")
        assert reg.service_prefix == "shield"

        registry.register("worker_start_workflow", noop, SampleToolConfig, ["worker:execute"])
        reg = registry.get("worker_start_workflow")
        assert reg.service_prefix == "worker"


# ── Config Validation ───────────────────────────────────────────────


class TestPlatformToolConfigValidation:
    """Test config validation via Pydantic schemas."""

    def test_valid_config_passes(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

        validated = registry.validate_config("brain_search", {"top_k": 5, "collection": "docs"})
        assert isinstance(validated, SampleToolConfig)
        assert validated.top_k == 5

    def test_config_with_defaults(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

        validated = registry.validate_config("brain_search", {})
        assert validated.top_k == 10
        assert validated.collection == "default"

    def test_invalid_config_raises(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_upsert", noop, BadToolConfig, ["brain:write"])

        with pytest.raises(PlatformServiceError, match="config validation"):
            registry.validate_config("brain_upsert", {})  # missing required_field


# ── Scope Enforcement ───────────────────────────────────────────────


class TestPlatformToolScopeEnforcement:
    """Test token scope checks before tool dispatch."""

    def test_valid_scopes_pass(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

        # Token with matching scope — should not raise
        token = ContextToken(token_id="test-token", permissions=("brain:read", "brain:write"))
        registry.check_scopes("brain_search", token)

    def test_inherited_scope_passes(self):
        """brain:write expands to brain:read via permission inheritance."""
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])
        token = ContextToken(token_id="test-token", permissions=("brain:write",))
        registry.check_scopes("brain_search", token)

    def test_missing_scope_raises_security_error(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

        token = ContextToken(token_id="test-token", permissions=("memory:write",))

        with pytest.raises(SecurityError, match="brain:read"):
            registry.check_scopes("brain_search", token)

    def test_no_token_raises_security_error(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )

        registry = PlatformToolRegistry()

        async def noop(state, config):
            return {}

        registry.register("brain_search", noop, SampleToolConfig, ["brain:read"])

        with pytest.raises(SecurityError, match="token"):
            registry.check_scopes("brain_search", None)
