"""Small utilities shared by LangGraph nodes.

These helpers keep node implementations concise and avoid duplicated code.

- `safe_preview`: compact logging preview for potentially large objects
- `pipeline_log`: structured debug logging gated by `DEBUG_PIPELINE=1`
"""

from __future__ import annotations

import logging

from contextrouter.core import get_bool_env

logger = logging.getLogger("contextrouter")


def safe_preview(val: object, limit: int = 240) -> str:
    """Create a single-line preview of a value for logs."""
    if val is None:
        return ""
    s = val if isinstance(val, str) else str(val)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def pipeline_log(event: str, **fields: object) -> None:
    """Log a structured pipeline event when debug logging is enabled."""
    debug_env = bool(get_bool_env("DEBUG_PIPELINE"))
    if not debug_env and not logger.isEnabledFor(logging.DEBUG):
        return
    safe_fields = {k: safe_preview(v, 220) for k, v in fields.items()}
    logger.info("PIPELINE %s | %s", event, safe_fields)
