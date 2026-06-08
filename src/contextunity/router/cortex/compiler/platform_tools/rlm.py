"""Router RLM Platform Tool — universal deep-context LLM processing.

Registers `router_rlm_process` wrapping model_registry.create_llm("rlm/...")
for massive-context tasks (50k+ items). Zero domain imports.

The RLM (Recursive Language Model) wrapper creates a Python REPL environment
where large datasets are passed as variables rather than prompt text, enabling
context sizes that would exceed any model's token window.

Architecture:
    Executor(state, config) → reads prompt + data → RLM call → writes result

Security:
    - Requires `router:execute` scope
    - Config schema: frozen=True, extra=forbid, bounded fields
    - No bare `except Exception` — typed PlatformServiceError

Model Resolution:
    Per-node model → Graph defaults.model → CU_ROUTER_DEFAULT_LLM
    RLM models use `rlm/<base_model>` prefix (e.g., `rlm/gpt-5-mini`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError
from contextunity.core.types import is_json_dict
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.modules.models import model_registry
from contextunity.router.modules.models.types import ModelRequest, TextPart

from .helpers.contracts import PlatformResult, PlatformState
from .helpers.registration import ToolRegistrationSpec, register_tool_specs
from .helpers.state import as_text

if TYPE_CHECKING:
    from ..platform_registry import PlatformToolRegistry

logger = get_contextunit_logger(__name__)


# ── Config Schema ───────────────────────────────────────────────────


class RLMProcessConfig(BaseModel, frozen=True):
    """Config for router_rlm_process tool.

    Security invariants:
    - frozen=True: no mutation after creation
    - extra=forbid: no unknown fields accepted
    - All numeric fields have bounds
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(
        default="",
        description=(
            "RLM model key (e.g., 'rlm/gpt-5-mini'). "
            "Falls back to state['model_key'] or graph defaults."
        ),
    )
    system_prompt: str = Field(
        default="",
        description=(
            "System prompt for the RLM session. Overridden by state['system_prompt'] if present."
        ),
    )
    input_key: str = Field(
        default="rlm_prompt",
        description="State key holding the prompt/instructions for RLM.",
    )
    output_key: str = Field(
        default="rlm_result",
        description="State key where RLM output text is written.",
    )
    custom_tools_key: str = Field(
        default="rlm_data",
        description=(
            "State key holding dict of REPL variables (large data). "
            "Passed as custom_tools to model.generate()."
        ),
    )
    max_output_tokens: int = Field(
        default=50000,
        ge=64,
        le=200000,
        description="Max tokens for RLM output.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for result filtering.",
    )
    reasoning_effort: Literal["none", "low", "medium", "high"] = Field(
        default="none",
        description="Reasoning effort level for RLM backend.",
    )


# ── Executor Adapter ────────────────────────────────────────────────


async def _router_rlm_process_executor(
    state: PlatformState, config: RLMProcessConfig
) -> PlatformResult:
    """Execute RLM deep-context processing.

    Reads prompt from state[config.input_key] and large data from
    state[config.custom_tools_key]. Passes data as REPL variables
    (not prompt text) to avoid context degradation.

    Returns:
        Dict with config.output_key (response text) and rlm_usage (token stats).
    """
    try:
        model_key = config.model or as_text(state.get("model_key", ""))
        if not model_key:
            raise PlatformServiceError(
                message=(
                    "router_rlm_process requires a model key "
                    "(config.model or state.model_key). "
                    "Use rlm/<base_model> format, e.g. 'rlm/gpt-5-mini'."
                ),
                tool_binding="router_rlm_process",
            )

        prompt_text = as_text(state.get(config.input_key, ""))
        system = config.system_prompt or as_text(state.get("system_prompt", ""))
        custom_tools_raw = state.get(config.custom_tools_key)
        custom_tools: dict[str, object] | None = (
            dict(custom_tools_raw) if is_json_dict(custom_tools_raw) else None
        )

        reasoning = config.reasoning_effort
        if reasoning and reasoning != "none":
            llm = model_registry.create_llm(
                model_key,
                reasoning_effort=reasoning,
                environment="docker",
            )
        else:
            llm = model_registry.create_llm(model_key, environment="docker")

        request = ModelRequest(
            system=system or None,
            parts=[TextPart(text=prompt_text)],
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

        if custom_tools is not None:
            response = await llm.generate(request, custom_tools=custom_tools)
        else:
            response = await llm.generate(request)

        usage = response.usage
        input_tokens = usage.input_tokens if usage and usage.input_tokens is not None else 0
        output_tokens = usage.output_tokens if usage and usage.output_tokens is not None else 0
        total_tokens = usage.total_tokens if usage and usage.total_tokens is not None else 0
        usage_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or (input_tokens + output_tokens),
        }

        return {
            config.output_key: response.text,
            "rlm_usage": usage_stats,
        }
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_rlm_process execution failed",
            tool_binding="router_rlm_process",
        ) from exc


# ── Registration ────────────────────────────────────────────────────


def register_router_rlm_tools(registry: PlatformToolRegistry) -> None:
    """Register RLM platform tool into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="router_rlm_process",
                executor=_router_rlm_process_executor,
                config_schema=RLMProcessConfig,
                required_scopes=["router:execute"],
            )
        ],
    )


__all__ = [
    "register_router_rlm_tools",
    "RLMProcessConfig",
    "_router_rlm_process_executor",
]
