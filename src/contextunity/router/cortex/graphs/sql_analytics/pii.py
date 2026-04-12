"""PII helpers — anonymize/deanonymize text around LLM calls.

These are thin wrappers around cu.zero tools that ensure:
- Graceful degradation (original text returned on failure)
- Consistent logging for debugging PII flow
- Proper handling of both dict and str responses from tools
"""

from __future__ import annotations

from typing import Any

from contextunity.core import get_contextunit_logger
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

logger = get_contextunit_logger(__name__)


async def pii_anonymize(
    text: str,
    *,
    tool: BaseTool | None,
    session_id: str,
    config: Any | None = None,
) -> str:
    """Anonymize text via Zero tool. Returns original on failure or if tool is None."""
    if not tool or not text or not session_id:
        return text
    try:
        result = await tool.ainvoke({"text": text, "session_id": session_id}, config=config)
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
        logger.exception("pii_anonymize FAILED: %s", e)
        raise RuntimeError(f"contextunity.zero anonymization failed: {e}") from e

    raise RuntimeError("contextunity.zero anonymization failed: tool returned unexpected format")


async def pii_deanonymize(
    text: str,
    *,
    tool: BaseTool | None,
    session_id: str,
    config: Any | None = None,
) -> str:
    """Deanonymize text via Zero tool. Returns original on failure or if tool is None."""
    if not tool or not text or not session_id:
        return text
    try:
        restored = await tool.ainvoke({"text": text, "session_id": session_id}, config=config)
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
                logger.exception(
                    "pii_deanonymize: Zero returned error: %s (session=%s)",
                    restored["error"],
                    session_id,
                )
            raise RuntimeError(
                f"contextunity.zero deanonymization failed: {restored.get('error', 'unknown error')}"
            )
    except Exception as e:
        logger.exception("pii_deanonymize FAILED: %s", e)
        raise RuntimeError(f"contextunity.zero deanonymization failed: {e}") from e

    raise RuntimeError("contextunity.zero deanonymization failed: tool returned unexpected format")


__all__ = ["pii_anonymize", "pii_deanonymize", "PiiSession"]


class PiiSession:
    """Smart context manager for handling PII anonymization and deanonymization.

    It abstracts away the tracing and allows recursive structure traversal
    for automatic deanonymization of dictionaries and lists.

    Note: PII tool calls are captured by BrainAutoTracer via LangChain callbacks.
    We do NOT record them in _steps to avoid duplicate Graph Journey entries.
    """

    def __init__(
        self,
        sub_steps: list[dict],
        session_id: str,
        anonymize_tool: BaseTool | None,
        deanonymize_tool: BaseTool | None,
        config: Any | None = None,
    ):
        self.sub_steps = sub_steps
        self.session_id = session_id
        self.anonymize_tool = anonymize_tool
        self.deanonymize_tool = deanonymize_tool
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def hide(self, data: Any) -> Any:
        """Anonymize a string, or human messages in a list."""
        if not self.anonymize_tool or not self.session_id:
            return data

        if isinstance(data, list):
            res = []
            for item in data:
                if isinstance(item, HumanMessage):
                    masked = await pii_anonymize(
                        item.content,
                        tool=self.anonymize_tool,
                        session_id=self.session_id,
                        config=self.config,
                    )
                    res.append(HumanMessage(content=masked))
                elif isinstance(item, BaseMessage):
                    res.append(item)
                elif isinstance(item, str):
                    masked = await pii_anonymize(
                        item,
                        tool=self.anonymize_tool,
                        session_id=self.session_id,
                        config=self.config,
                    )
                    res.append(masked)
                else:
                    res.append(item)
            return res
        elif isinstance(data, str):
            return await pii_anonymize(
                data,
                tool=self.anonymize_tool,
                session_id=self.session_id,
                config=self.config,
            )
        return data

    async def reveal(self, data: Any) -> Any:
        """Deanonymize PII tokens in a data structure with a single tool call.

        Instead of recursively traversing and calling deanonymize per string
        (which caused 100+ gRPC calls for complex structures), we serialize
        the entire structure to JSON, make ONE deanonymize call, then parse back.
        PII tokens (PER_xxxxx, ORG_xxxxx etc.) are simple string substitutions,
        so deanonymizing the whole JSON blob is equivalent.
        """
        if not self.deanonymize_tool or not self.session_id:
            return data

        import json

        # If data is already a string, no need to JSON round-trip it.
        # This avoids issues where the original text contains quotes/newlines
        # that get incorrectly escaped or unescaped during replacement.
        if isinstance(data, str):
            try:
                return await pii_deanonymize(
                    data,
                    tool=self.deanonymize_tool,
                    session_id=self.session_id,
                    config=self.config,
                )
            except Exception as e:
                logger.warning("reveal: string deanonymize failed: %s", e)
                return data

        # For complex structures, serialize them first.
        # Note: If the replaced values contain quotes, this may break standard JSON!
        # We try our best to round-trip it.
        try:
            raw = json.dumps(data, ensure_ascii=False, default=str)
            restored = await pii_deanonymize(
                raw,
                tool=self.deanonymize_tool,
                session_id=self.session_id,
                config=self.config,
            )
            return json.loads(restored)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("reveal: JSON round-trip failed, falling back to original: %s", e)
            return data
        except Exception as e:
            logger.warning("reveal: complex deanonymize failed: %s", e)
            return data


__all__ = ["pii_anonymize", "pii_deanonymize", "PiiSession"]
