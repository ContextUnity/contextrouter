"""Prompt Resolver for the Graph Compiler (Phase 4.D).
Resolves ``prompt_ref`` keys from template nodes into final prompt text.
Supports:
- Registry-based lookup (prompts registered at boot time)
- ``str.format_map()`` variable substitution with ``ctx_`` prefix convention
- Discovery of existing prompts from ``cortex/prompting/`` module
Security by Construction:
- Variable names MUST start with ``ctx_`` — prevents JSON key collisions
  and limits attack surface for prompt injection via variable substitution.
- Missing variables → ConfigurationError (fail-closed, not silent empty string).
- Registry is module-level dict — not user-modifiable at runtime in production.
This is a stepping stone: Phase 5 will migrate prompts to YAML resource files
and deprecate the Python import path.
"""

from __future__ import annotations

import logging
import re

from contextunity.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# ── Prompt Registry ───────────────────────────────────────────────

# Module-level registry: prompt_ref → prompt text
_PROMPT_REGISTRY: dict[str, str] = {}

# ctx_ variable pattern for validation
_CTX_VAR_RE = re.compile(r"\{(ctx_[a-z0-9_]+)\}")
_ANY_VAR_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def register_prompt(ref: str, text: str) -> None:
    """Register a prompt text under a given ref name.

    Args:
        ref: Prompt reference key (e.g., 'rag_intent').
        text: Prompt template text (may contain {ctx_*} placeholders).

    Raises:
        ConfigurationError: If ref is empty or text is empty.
    """
    if not ref or not ref.strip():
        raise ConfigurationError(message="Prompt ref cannot be empty")
    if not text or not text.strip():
        raise ConfigurationError(message=f"Prompt text for '{ref}' cannot be empty")

    _PROMPT_REGISTRY[ref] = text
    logger.debug("📝 Registered prompt '%s' (%d chars)", ref, len(text))


def get_prompt_text(ref: str) -> str:
    """Get raw prompt text by ref. No variable substitution.

    Args:
        ref: Prompt reference key.

    Returns:
        Raw prompt text.

    Raises:
        ConfigurationError: If ref not found in registry.
    """
    if ref not in _PROMPT_REGISTRY:
        raise ConfigurationError(
            message=(
                f"Prompt ref '{ref}' not found in registry. "
                f"Available prompts: {sorted(_PROMPT_REGISTRY.keys())}"
            ),
        )
    return _PROMPT_REGISTRY[ref]


def _validate_variable_safety(text: str, ref: str) -> None:
    """Validate that all variables use ctx_ prefix."""
    all_vars = set(_ANY_VAR_RE.findall(text))
    ctx_vars = set(_CTX_VAR_RE.findall(text))
    unsafe_vars = all_vars - ctx_vars

    if unsafe_vars:
        raise ConfigurationError(
            message=(
                f"Prompt '{ref}' contains variables without 'ctx_' prefix: "
                f"{sorted(unsafe_vars)}. All template variables must use "
                f"'ctx_' prefix to prevent injection and key collisions."
            ),
        )


def _validate_all_variables_provided(
    text: str,
    ref: str,
    ctx_vars: dict[str, object],
) -> None:
    """Raise ``ConfigurationError`` if any ``{ctx_*}`` placeholder in *text* has no matching key in *ctx_vars*."""
    required = set(_CTX_VAR_RE.findall(text))
    provided = set(ctx_vars.keys())
    missing = required - provided

    if missing:
        raise ConfigurationError(
            message=(
                f"Prompt '{ref}' requires variables {sorted(missing)} "
                f"but they were not provided. "
                f"Pass them via ctx_vars dict."
            ),
        )


class _SafeFormatMap(dict[str, object]):
    """Format map that raises on missing keys instead of returning empty string.

    Used by str.format_map() to fail-closed rather than silently dropping vars.
    """

    _ref: str

    def __init__(self, data: dict[str, object], ref: str) -> None:
        """Wrap *data* and remember the prompt *ref* name for error messages."""
        super().__init__(data)
        self._ref = ref

    def __missing__(self, key: str) -> str:
        """Raise ``ConfigurationError`` instead of returning an empty string for undefined keys."""
        raise ConfigurationError(
            message=f"Prompt '{self._ref}' references {{{{key}}}} but no value provided",
        )


def resolve_prompt_ref(
    ref: str,
    *,
    ctx_vars: dict[str, object] | None = None,
) -> str:
    """Resolve a prompt_ref into final prompt text.

    Resolution pipeline:
    1. Lookup ref in registry → raw text
    2. Validate all variables use ctx_ prefix
    3. If ctx_vars provided, substitute via str.format_map()
    4. If no variable placeholders, return raw text

    Args:
        ref: Prompt reference key (e.g., 'rag_intent').
        ctx_vars: Variable values for substitution. Keys must start with 'ctx_'.

    Returns:
        Final prompt text with variables resolved.

    Raises:
        ConfigurationError: Ref not found, unsafe vars, or missing values.
    """
    text = get_prompt_text(ref)

    # Check variable safety
    _validate_variable_safety(text, ref)

    # If no variables in prompt, return raw text
    required = set(_CTX_VAR_RE.findall(text))
    if not required:
        return text

    # Variables found — need ctx_vars
    effective_vars = ctx_vars or {}
    _validate_all_variables_provided(text, ref, effective_vars)

    # Substitute using format_map (fail-closed via _SafeFormatMap)
    safe_map = _SafeFormatMap(effective_vars, ref)
    return text.format_map(safe_map)


# ── Boot-time Discovery ──────────────────────────────────────────


def _auto_discover_prompts() -> None:
    """Auto-discover prompts from cortex/prompting module.

    Called once at module load time. Seeds the registry with
    existing Python-defined prompts from cortex/prompting/*.py.
    """
    try:
        from contextunity.router.cortex.compiler.platform_tools.prompts import (
            IDENTITY_PROMPT,
            IDENTITY_RESPONSE,
            IDENTITY_SUGGESTIONS_PROMPT,
            INTENT_DETECTION_PROMPT,
            NO_RESULTS_PROMPT,
            NO_RESULTS_RESPONSE,
            RAG_SYSTEM_PROMPT,
            SEARCH_SUGGESTIONS_PROMPT,
            SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT,
        )

        _BUILTIN_PROMPTS: dict[str, str] = {
            "rag_intent": INTENT_DETECTION_PROMPT,
            "rag_generate": RAG_SYSTEM_PROMPT,
            "rag_no_results": NO_RESULTS_PROMPT,
            "rag_no_results_response": NO_RESULTS_RESPONSE,
            "rag_suggest": SEARCH_SUGGESTIONS_PROMPT,
            "rag_suggest_with_context": SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT,
            "identity": IDENTITY_PROMPT,
            "identity_response": IDENTITY_RESPONSE,
            "identity_suggestions": IDENTITY_SUGGESTIONS_PROMPT,
        }

        for ref_name, prompt_text in _BUILTIN_PROMPTS.items():
            if ref_name not in _PROMPT_REGISTRY and prompt_text:
                _PROMPT_REGISTRY[ref_name] = prompt_text

        logger.debug(
            "📝 Auto-discovered %d built-in prompts",
            len(_BUILTIN_PROMPTS),
        )
    except ImportError:
        logger.debug("prompting module not available — skipping auto-discovery")


# Run auto-discovery on module load
_auto_discover_prompts()

__all__ = [
    "register_prompt",
    "get_prompt_text",
    "resolve_prompt_ref",
]
