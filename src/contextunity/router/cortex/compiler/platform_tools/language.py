"""Language Tool — local text quality tools for compiled graph nodes.

Registers language_tool into PlatformToolRegistry.
No gRPC, no token scope — this is a local utility tool.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError
from pydantic import BaseModel, ConfigDict, Field

from .helpers.contracts import PlatformResult, PlatformState
from .helpers.registration import PlatformRegistry, ToolRegistrationSpec, register_tool_specs
from .helpers.state import get_text_from_state

logger = get_contextunit_logger(__name__)


# ── Config Schema ───────────────────────────────────────────────────


class LanguageToolConfig(BaseModel, frozen=True):
    """Config for language_tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    language: Literal["uk", "en", "de", "fr", "es", "pl", "pt", "it", "nl"] = "uk"
    categories: list[str] = Field(default_factory=lambda: ["grammar", "spelling", "typography"])
    max_suggestions: int = Field(default=5, ge=1, le=20)


# ── Executor ────────────────────────────────────────────────────────


async def _language_tool_executor(
    state: PlatformState, config: LanguageToolConfig
) -> PlatformResult:
    """Check text for grammar, spelling, typography errors.

    Uses LanguageTool API (local or remote). Falls back to
    a no-op result if LanguageTool is unavailable.
    """
    content = get_text_from_state(state, "final_output")

    try:
        import language_tool_python

        tool = language_tool_python.LanguageToolPublicAPI(config.language)
        matches = tool.check(str(content))

        suggestions: list[dict[str, object]] = []
        for match in matches[: config.max_suggestions]:
            error_length = getattr(match, "errorLength", 0)
            rule_id = getattr(match, "ruleId", "")
            suggestions.append(
                {
                    "message": match.message,
                    "offset": match.offset,
                    "length": int(error_length) if isinstance(error_length, int | float) else 0,
                    "replacements": match.replacements[:3],
                    "category": match.category,
                    "rule_id": str(rule_id),
                }
            )
        return {"suggestions": suggestions, "error_count": len(matches)}

    except ImportError:
        logger.warning("language-tool-python not installed; returning empty suggestions")
        return {"suggestions": [], "error_count": 0, "warning": "LanguageTool not available"}
    except Exception:  # graceful-degrade: tool failure returns empty result
        logger.warning("LanguageTool check failed on node execution", exc_info=True)
        raise PlatformServiceError(
            message="LanguageTool check unavailable",
            tool_binding="language_tool",
        )


# ── Registration ────────────────────────────────────────────────────


def register_language_tools(registry: PlatformRegistry) -> None:
    """Register language tool into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="language_tool",
                executor=_language_tool_executor,
                config_schema=LanguageToolConfig,
                required_scopes=[
                    "router:execute"
                ],  # Self-hosted but still requires scope enforcement
            )
        ],
    )


__all__ = ["register_language_tools", "LanguageToolConfig"]
