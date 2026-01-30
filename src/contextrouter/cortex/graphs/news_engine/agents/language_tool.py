"""
LanguageTool integration for grammar and spell checking.

Manages the lifecycle of LanguageTool server - init before batch,
apply to each text, close after batch.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# LanguageTool instance (managed per-batch lifecycle)
_language_tool_instance = None


def init_language_tool(lang: str = "uk"):
    """Initialize LanguageTool for a batch of corrections.

    Call this at the start of post generation, then close_language_tool() at the end.

    Args:
        lang: Language code (e.g., 'uk', 'en', 'de')
    """
    global _language_tool_instance
    if _language_tool_instance is not None:
        return  # Already initialized

    try:
        import language_tool_python

        logger.info(f"Starting LanguageTool server for language: {lang}")
        _language_tool_instance = language_tool_python.LanguageTool(lang)
    except ImportError:
        logger.warning("language-tool-python not installed")
    except Exception as e:
        logger.error(f"Failed to start LanguageTool: {e}")


def close_language_tool():
    """Close LanguageTool after batch is complete."""
    global _language_tool_instance
    if _language_tool_instance is not None:
        try:
            logger.info("Closing LanguageTool server")
            _language_tool_instance.close()
        except Exception as e:
            logger.warning(f"Error closing LanguageTool: {e}")
        finally:
            _language_tool_instance = None


async def apply_language_tool(text: str, auto_correct: bool = True) -> str:
    """Apply LanguageTool grammar/spell check to text.

    Requires init_language_tool() to be called first.

    Args:
        text: Text to check
        auto_correct: If True, return corrected text; if False, just log errors

    Returns:
        Corrected text if auto_correct=True, else original text
    """
    global _language_tool_instance

    if _language_tool_instance is None:
        return text  # Not initialized, skip

    try:
        import asyncio

        def check_and_correct():
            matches = _language_tool_instance.check(text)

            if matches:
                logger.info(f"LanguageTool found {len(matches)} issues in text")
                for match in matches[:5]:  # Log first 5 issues
                    logger.debug(
                        f"  - {match.rule_issue_type}: '{match.matched_text}' -> {match.replacements[:3]}"
                    )

                if auto_correct:
                    return _language_tool_instance.correct(text)

            return text

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, check_and_correct)

    except Exception as e:
        logger.warning(f"LanguageTool error: {e}")
        return text


__all__ = [
    "init_language_tool",
    "close_language_tool",
    "apply_language_tool",
]
