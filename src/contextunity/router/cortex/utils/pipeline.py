"""Small utilities shared by LangGraph nodes.

These helpers keep node implementations concise and avoid duplicated code.
- `safe_preview`: compact logging preview for potentially large objects
- `pipeline_log`: structured debug logging gated by `DEBUG_PIPELINE=1`
"""

from __future__ import annotations

import logging

from contextunity.core import get_contextunit_logger

from contextunity.router.core import get_bool_env

logger = get_contextunit_logger("contextunity.router")


def safe_preview(val: object, limit: int = 240) -> str:
    """Create a single-line preview of a value for logs, truncating if it exceeds the limit.

    Args:
        val: The object/value to convert to a string and preview.
        limit: The maximum length of the output preview string. Defaults to 240.

    Returns:
        A normalized single-line preview string, truncated with an ellipsis if necessary.
    """
    if val is None:
        return ""
    s = val if isinstance(val, str) else str(val)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def pipeline_log(event: str, **fields: object) -> None:
    """Log a structured pipeline event when debug logging is enabled.

    Logs the event string and key-value fields at the INFO level when either the
    `DEBUG_PIPELINE` environment variable is active or the logger is set to DEBUG.

    Args:
        event: The name or identifier of the pipeline event.
        **fields: Keyword arguments containing values to include in the log fields.
    """
    debug_env = bool(get_bool_env("DEBUG_PIPELINE"))
    if not debug_env and not logger.isEnabledFor(logging.DEBUG):
        return
    safe_fields = {k: safe_preview(v, 220) for k, v in fields.items()}
    logger.info("PIPELINE %s | %s", event, safe_fields)
