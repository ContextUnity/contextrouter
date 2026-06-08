"""Tests for router_rlm_process platform tool — RLM deep processing.

Phase 5d: Universal RLM platform tool that wraps model_registry.create_llm("rlm/...")
for massive-context processing (50k+ items). Zero domain imports.

Config security: frozen=True, extra=forbid, bounded fields.
Scope security: requires router:execute.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ── Config Schema Tests ─────────────────────────────────────────────


class TestRLMProcessConfig:
    """RLMProcessConfig schema validation."""

    def test_reasoning_effort_enum(self):
        from contextunity.router.cortex.compiler.platform_tools.rlm import (
            RLMProcessConfig,
        )

        RLMProcessConfig(reasoning_effort="none")
        RLMProcessConfig(reasoning_effort="low")
        RLMProcessConfig(reasoning_effort="medium")
        RLMProcessConfig(reasoning_effort="high")

        with pytest.raises(ValidationError):
            RLMProcessConfig(reasoning_effort="ultra")  # type: ignore[arg-type]


# ── Registration Tests ───────────────────────────────────────────────


# ── Config Validation via Registry ───────────────────────────────────


# ── Scope Enforcement Tests ──────────────────────────────────────────


# ── Executor Tests ───────────────────────────────────────────────────


class TestRLMProcessExecutor:
    """Executor adapter contract tests."""

    @pytest.mark.asyncio
    async def test_missing_model_raises_error(self):
        from contextunity.core.exceptions import PlatformServiceError

        from contextunity.router.cortex.compiler.platform_tools.rlm import (
            RLMProcessConfig,
            _router_rlm_process_executor,
        )

        config = RLMProcessConfig()
        state: dict = {"rlm_prompt": "match these products"}

        with pytest.raises(PlatformServiceError, match="requires a model"):
            await _router_rlm_process_executor(state, config)

    @pytest.mark.asyncio
    async def test_executor_state_output_and_usage(self, monkeypatch):
        """Executor writes result to config.output_key and returns usage stats."""
        from contextunity.router.cortex.compiler.platform_tools.rlm import (
            RLMProcessConfig,
            _router_rlm_process_executor,
        )
        from contextunity.router.modules.models.types import (
            ModelCapabilities,
            ModelRequest,
            ModelResponse,
            ProviderInfo,
            UsageStats,
        )

        class _StubRLMLLM:
            @property
            def capabilities(self):
                return ModelCapabilities(
                    supports_text=True, supports_image=False, supports_audio=False
                )

            async def generate(self, request: ModelRequest, **kwargs) -> ModelResponse:
                self._last_request = request
                self._last_kwargs = kwargs
                return ModelResponse(
                    text='{"matches": [1, 2]}',
                    raw_provider=ProviderInfo(
                        provider="test", model_name="stub", model_key="rlm/stub"
                    ),
                    usage=UsageStats(input_tokens=500, output_tokens=1000, total_tokens=1500),
                )

        stub = _StubRLMLLM()
        create_kwargs = {}

        def _create_llm(_self, _key, **kwargs):
            create_kwargs.update(kwargs)
            return stub

        monkeypatch.setattr(
            "contextunity.router.cortex.compiler.platform_tools.rlm.model_registry",
            type("R", (), {"create_llm": _create_llm})(),
        )

        config = RLMProcessConfig(model="rlm/gpt-5-mini", output_key="my_result")
        state = {"rlm_prompt": "match", "rlm_data": {"products": [{"id": 1}]}}

        result = await _router_rlm_process_executor(state, config)

        # Output written to configured key
        assert "my_result" in result
        assert result["my_result"] == '{"matches": [1, 2]}'
        # Usage stats extracted
        assert result["rlm_usage"]["input_tokens"] == 500
        assert result["rlm_usage"]["output_tokens"] == 1000
        assert result["rlm_usage"]["total_tokens"] == 1500
        # Custom tools passed through
        assert stub._last_kwargs.get("custom_tools") == {"products": [{"id": 1}]}
        assert create_kwargs["environment"] == "docker"

    @pytest.mark.asyncio
    async def test_executor_system_prompt_and_temperature(self, monkeypatch):
        """System prompt and temperature propagate to ModelRequest."""
        from contextunity.router.cortex.compiler.platform_tools.rlm import (
            RLMProcessConfig,
            _router_rlm_process_executor,
        )
        from contextunity.router.modules.models.types import (
            ModelRequest,
            ModelResponse,
            ProviderInfo,
            UsageStats,
        )

        captured = {}

        class _CaptureLLM:
            async def generate(self, request: ModelRequest, **kwargs) -> ModelResponse:
                captured["system"] = request.system
                captured["temperature"] = request.temperature
                return ModelResponse(
                    text="ok",
                    raw_provider=ProviderInfo(provider="t", model_name="s", model_key="rlm/s"),
                    usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
                )

        monkeypatch.setattr(
            "contextunity.router.cortex.compiler.platform_tools.rlm.model_registry",
            type("R", (), {"create_llm": lambda self, key, **kw: _CaptureLLM()})(),
        )

        config = RLMProcessConfig(
            model="rlm/gpt-5-mini",
            system_prompt="You are a matcher",
            temperature=0.1,
        )
        await _router_rlm_process_executor({"rlm_prompt": "go", "rlm_data": {}}, config)

        assert captured["system"] == "You are a matcher"
        assert captured["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_executor_wraps_unexpected_errors(self, monkeypatch):
        """Non-PlatformServiceError exceptions get wrapped, not leaked."""
        from contextunity.core.exceptions import PlatformServiceError

        from contextunity.router.cortex.compiler.platform_tools.rlm import (
            RLMProcessConfig,
            _router_rlm_process_executor,
        )

        def _explode(key, **kw):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(
            "contextunity.router.cortex.compiler.platform_tools.rlm.model_registry",
            type("R", (), {"create_llm": _explode})(),
        )

        config = RLMProcessConfig(model="rlm/gpt-5-mini")
        with pytest.raises(PlatformServiceError, match="execution failed"):
            await _router_rlm_process_executor({"rlm_prompt": "go", "rlm_data": {}}, config)
