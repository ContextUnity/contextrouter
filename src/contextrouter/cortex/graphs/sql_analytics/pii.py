"""PII helpers — anonymize/deanonymize text around LLM calls.

These are thin wrappers around ContextZero tools that ensure:
- Graceful degradation (original text returned on failure)
- Consistent logging for debugging PII flow
- Proper handling of both dict and str responses from tools
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def pii_anonymize(
    text: str,
    *,
    tool: BaseTool | None,
    session_id: str,
) -> str:
    """Anonymize text via Zero tool. Returns original on failure or if tool is None."""
    if not tool or not text or not session_id:
        return text
    try:
        result = await tool.ainvoke({"text": text, "session_id": session_id})
        if isinstance(result, dict):
            anonymized = result.get("anonymized_text", text)
            entities = result.get("entities_masked", 0)
            if entities:
                logger.debug(
                    "pii_anonymize: masked %d entities (session=%s, len=%d→%d)",
                    entities,
                    session_id,
                    len(text),
                    len(anonymized),
                )
            return anonymized
        # Tool might return str directly (unlikely for anonymize)
        if isinstance(result, str):
            return result
    except Exception as e:
        logger.warning("pii_anonymize failed, using original: %s", e)
    return text


async def pii_deanonymize(
    text: str,
    *,
    tool: BaseTool | None,
    session_id: str,
) -> str:
    """Deanonymize text via Zero tool. Returns original on failure or if tool is None."""
    if not tool or not text or not session_id:
        return text
    try:
        restored = await tool.ainvoke({"text": text, "session_id": session_id})
        # Tool returns str (local and RPC modes)
        if isinstance(restored, str):
            if restored.strip():
                return restored
            logger.warning(
                "pii_deanonymize: tool returned empty string (session=%s)",
                session_id,
            )
            return text
        # Unexpected: tool returned dict (error response from gRPC wrapper?)
        if isinstance(restored, dict):
            logger.warning(
                "pii_deanonymize: got dict instead of str: %s (session=%s)",
                {k: str(v)[:80] for k, v in restored.items()},
                session_id,
            )
            # Try to extract restored text from a dict response
            if "restored_text" in restored:
                return restored["restored_text"]
            if "error" in restored:
                logger.error(
                    "pii_deanonymize: Zero returned error: %s (session=%s)",
                    restored["error"],
                    session_id,
                )
            return text
    except Exception as e:
        logger.warning("pii_deanonymize failed, using original: %s", e)
    return text


__all__ = ["pii_anonymize", "pii_deanonymize"]
