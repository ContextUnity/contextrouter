"""Router Content Platform Tools — universal LLM capabilities as compiled graph nodes.

Registers 6 generic content-oriented tools wrapping pure LLM capabilities.
These tools have ZERO domain imports (no commerce/, news_engine/).
They accept generic GraphState and produce StateUpdate-compatible output.

Architecture:
    Executor(state, config) → reads state keys → LLM call → writes state keys

Security:
    - All tools require `router:execute` scope
    - Config schemas: frozen=True, extra=forbid, bounded fields
    - No bare `except Exception` — typed PlatformServiceError

Tools:
    router_classify          — Taxonomy/intent classification
    router_generate_content  — Structured content generation
    router_review_content    — Quality review + correction
    router_filter_content    — Content filtering/validation
    router_plan_content      — Editorial/batch planning
    router_match_semantic    — Semantic similarity matching
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError
from pydantic import BaseModel, ConfigDict, Field

from .helpers.content import run_text_generation
from .helpers.contracts import PlatformResult, PlatformState
from .helpers.registration import ToolRegistrationSpec, register_tool_specs
from .helpers.state import get_text_from_state

if TYPE_CHECKING:
    from ..platform_registry import PlatformToolRegistry

logger = get_contextunit_logger(__name__)


# ── Config Schemas ──────────────────────────────────────────────────


class ClassifyConfig(BaseModel, frozen=True):
    """Config for router_classify tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model key override")
    taxonomy_key: str = Field(default="taxonomy", description="State key holding taxonomy/schema")
    input_key: str = Field(default="input_text", description="State key holding input to classify")
    output_key: str = Field(default="classification_result", description="State key for output")
    response_format: Literal["text", "json"] = "json"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GenerateContentConfig(BaseModel, frozen=True):
    """Config for router_generate_content tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model key override")
    input_key: str = Field(
        default="content_input", description="State key holding generation context"
    )
    output_key: str = Field(default="generated_content", description="State key for output")
    language: str = Field(default="en", description="Target language code")
    response_format: Literal["text", "json"] = "text"
    max_tokens: int = Field(default=4096, ge=64, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ReviewContentConfig(BaseModel, frozen=True):
    """Config for router_review_content tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model key override")
    input_key: str = Field(default="content_to_review", description="State key holding content")
    output_key: str = Field(default="reviewed_content", description="State key for output")
    strict_mode: bool = Field(default=True, description="Reject content below quality threshold")
    language: str = Field(default="en", description="Language for grammar checks")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class FilterContentConfig(BaseModel, frozen=True):
    """Config for router_filter_content tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model key override")
    input_key: str = Field(default="items_to_filter", description="State key holding items")
    output_key: str = Field(default="filtered_items", description="State key for output")
    criteria_key: str = Field(
        default="filter_criteria",
        description="State key holding filtering criteria prompt",
    )
    pass_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class PlanContentConfig(BaseModel, frozen=True):
    """Config for router_plan_content tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model key override")
    input_key: str = Field(default="items_to_plan", description="State key holding source items")
    output_key: str = Field(default="content_plan", description="State key for output plan")
    max_items: int = Field(default=20, ge=1, le=500)
    strategy: Literal["editorial", "chronological", "priority"] = "editorial"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class MatchSemanticConfig(BaseModel, frozen=True):
    """Config for router_match_semantic tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model key override")
    input_key: str = Field(default="match_candidates", description="State key holding candidates")
    output_key: str = Field(default="match_results", description="State key for output")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_candidates: int = Field(default=50, ge=1, le=1000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# ── Executor Adapters ───────────────────────────────────────────────


async def _router_classify_executor(state: PlatformState, config: ClassifyConfig) -> PlatformResult:
    """Classify input against a taxonomy/schema using LLM.

    Reads taxonomy from state[config.taxonomy_key] and input from
    state[config.input_key]. Produces classification in state[config.output_key].
    """
    try:
        taxonomy = get_text_from_state(state, config.taxonomy_key)
        input_text = get_text_from_state(state, config.input_key)
        system_prompt = get_text_from_state(state, "system_prompt")

        prompt = f"{system_prompt}\n\nSchema:\n{taxonomy}\n\nInput:\n{input_text}".strip()
        generated = await run_text_generation(
            state=state,
            tool_binding="router_classify",
            model_override=config.model,
            prompt=prompt,
            temperature=config.temperature,
        )
        return {config.output_key: generated}
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_classify execution failed",
            tool_binding="router_classify",
        ) from exc


async def _router_generate_content_executor(
    state: PlatformState, config: GenerateContentConfig
) -> PlatformResult:
    """Generate structured content from prompt + context.

    Reads context from state[config.input_key], produces output in
    state[config.output_key].
    """
    try:
        content_input = get_text_from_state(state, config.input_key)
        system_prompt = get_text_from_state(state, "system_prompt")
        language = config.language

        prompt = f"{system_prompt}\n\nLanguage: {language}\n\n{content_input}".strip()
        generated = await run_text_generation(
            state=state,
            tool_binding="router_generate_content",
            model_override=config.model,
            prompt=prompt,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        return {config.output_key: generated}
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_generate_content execution failed",
            tool_binding="router_generate_content",
        ) from exc


async def _router_review_content_executor(
    state: PlatformState, config: ReviewContentConfig
) -> PlatformResult:
    """Review content quality and optionally correct issues.

    Reads content from state[config.input_key], produces reviewed content
    in state[config.output_key].
    """
    try:
        content = get_text_from_state(state, config.input_key)
        system_prompt = get_text_from_state(state, "system_prompt")
        mode = "strict" if config.strict_mode else "lenient"

        prompt = (
            f"{system_prompt}\n\n"
            f"Review mode: {mode}\nLanguage: {config.language}\n\n"
            f"Content to review:\n{content}"
        ).strip()

        generated = await run_text_generation(
            state=state,
            tool_binding="router_review_content",
            model_override=config.model,
            prompt=prompt,
            temperature=config.temperature,
        )
        return {config.output_key: generated}
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_review_content execution failed",
            tool_binding="router_review_content",
        ) from exc


async def _router_filter_content_executor(
    state: PlatformState, config: FilterContentConfig
) -> PlatformResult:
    """Filter/validate content items against criteria.

    Reads items from state[config.input_key] and criteria from
    state[config.criteria_key]. Produces filtered items in state[config.output_key].
    """
    try:
        items = get_text_from_state(state, config.input_key)
        criteria = get_text_from_state(state, config.criteria_key)
        system_prompt = get_text_from_state(state, "system_prompt")

        prompt = (
            f"{system_prompt}\n\nFilter criteria:\n{criteria}\n\nItems to evaluate:\n{items}"
        ).strip()

        generated = await run_text_generation(
            state=state,
            tool_binding="router_filter_content",
            model_override=config.model,
            prompt=prompt,
            temperature=config.temperature,
        )
        return {config.output_key: generated}
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_filter_content execution failed",
            tool_binding="router_filter_content",
        ) from exc


async def _router_plan_content_executor(
    state: PlatformState, config: PlanContentConfig
) -> PlatformResult:
    """Plan editorial/batch content organisation from a set of items.

    Reads source items from state[config.input_key], produces a plan
    in state[config.output_key].
    """
    try:
        items = get_text_from_state(state, config.input_key)
        system_prompt = get_text_from_state(state, "system_prompt")

        prompt = (
            f"{system_prompt}\n\n"
            f"Strategy: {config.strategy}\n"
            f"Max items: {config.max_items}\n\n"
            f"Source items:\n{items}"
        ).strip()

        generated = await run_text_generation(
            state=state,
            tool_binding="router_plan_content",
            model_override=config.model,
            prompt=prompt,
            temperature=config.temperature,
        )
        return {config.output_key: generated}
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_plan_content execution failed",
            tool_binding="router_plan_content",
        ) from exc


async def _router_match_semantic_executor(
    state: PlatformState, config: MatchSemanticConfig
) -> PlatformResult:
    """Semantic similarity matching and reranking via LLM.

    Reads candidates from state[config.input_key], produces ranked matches
    in state[config.output_key].
    """
    try:
        candidates = get_text_from_state(state, config.input_key)
        query = get_text_from_state(state, "match_query", fallback_key="user_query")
        system_prompt = get_text_from_state(state, "system_prompt")

        prompt = (
            f"{system_prompt}\n\n"
            f"Threshold: {config.threshold}\n"
            f"Max candidates: {config.max_candidates}\n\n"
            f"Query:\n{query}\n\n"
            f"Candidates:\n{candidates}"
        ).strip()

        generated = await run_text_generation(
            state=state,
            tool_binding="router_match_semantic",
            model_override=config.model,
            prompt=prompt,
            temperature=config.temperature,
        )
        return {config.output_key: generated}
    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_match_semantic execution failed",
            tool_binding="router_match_semantic",
        ) from exc


# ── Registration ────────────────────────────────────────────────────


def register_router_content_tools(registry: PlatformToolRegistry) -> None:
    """Register all universal content LLM tools into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="router_classify",
                executor=_router_classify_executor,
                config_schema=ClassifyConfig,
                required_scopes=["router:execute"],
            ),
            ToolRegistrationSpec(
                binding="router_generate_content",
                executor=_router_generate_content_executor,
                config_schema=GenerateContentConfig,
                required_scopes=["router:execute"],
            ),
            ToolRegistrationSpec(
                binding="router_review_content",
                executor=_router_review_content_executor,
                config_schema=ReviewContentConfig,
                required_scopes=["router:execute"],
            ),
            ToolRegistrationSpec(
                binding="router_filter_content",
                executor=_router_filter_content_executor,
                config_schema=FilterContentConfig,
                required_scopes=["router:execute"],
            ),
            ToolRegistrationSpec(
                binding="router_plan_content",
                executor=_router_plan_content_executor,
                config_schema=PlanContentConfig,
                required_scopes=["router:execute"],
            ),
            ToolRegistrationSpec(
                binding="router_match_semantic",
                executor=_router_match_semantic_executor,
                config_schema=MatchSemanticConfig,
                required_scopes=["router:execute"],
            ),
        ],
    )


__all__ = [
    "register_router_content_tools",
    "ClassifyConfig",
    "GenerateContentConfig",
    "ReviewContentConfig",
    "FilterContentConfig",
    "PlanContentConfig",
    "MatchSemanticConfig",
]
