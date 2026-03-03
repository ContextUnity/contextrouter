"""Planner node — generates SQL from user question."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    acc_tokens,
    extract_json,
    invoke_model,
)
from contextrouter.cortex.graphs.sql_analytics.pii import (
    PiiSession,
)
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.models import model_registry

logger = logging.getLogger(__name__)


def make_planner_node(
    *,
    planner_prompt: str,
    default_model_key: str | None,
    fallback_keys: list[str] | None = None,
    pii_masking: bool,
    anonymize_tool: BaseTool | None,
    deanonymize_tool: BaseTool | None = None,
):
    """Create the planner node closure.

    Atomic PII flow (when pii_masking=True):
        anonymize user messages → LLM (sees PER_xxxx) → deanonymize SQL output.
        The SQL that leaves this node always contains real names.

    Tracing: All tool/LLM calls go through LangChain and are captured
    automatically by BrainAutoTracer callbacks — no manual _steps needed.
    """

    async def planner_node(state: SqlAnalyticsState):
        # Record pipeline start time on first entry
        updates: dict[str, Any] = {}
        if not state.get("_start_ts"):
            import time

            updates["_start_ts"] = time.monotonic()

        messages = state["messages"]
        llm = model_registry.get_llm_with_fallback(default_model_key, fallback_keys=fallback_keys)

        # Increment retry count if returning from error or failed validation
        current_retry = state.get("retry_count", 0)
        validation = state.get("validation") or {}
        if state.get("error") or (validation and not validation.get("valid", True)):
            current_retry += 1

        # Prepare context
        logger.info(
            "planner_node: planner_prompt len=%d, first_100=%s",
            len(planner_prompt),
            planner_prompt[:100],
        )
        sys_msg = SystemMessage(content=planner_prompt)
        history = list(messages)

        # ── PII session: generate unique ID per graph execution ──
        # On first planner call, create a new session. On retries, reuse it.
        metadata = state.get("metadata") or {}
        session_id = metadata.get("session_id", "")
        if pii_masking and not session_id:
            session_id = f"pii-{uuid.uuid4().hex[:12]}"
            metadata = {**metadata, "session_id": session_id}
            updates["metadata"] = metadata

        async with PiiSession(
            sub_steps=[],  # unused — callbacks handle tracing
            session_id=session_id,
            anonymize_tool=anonymize_tool if pii_masking else None,
            deanonymize_tool=deanonymize_tool if pii_masking else None,
        ) as pii:
            history = await pii.hide(history)

            if state.get("error") and state.get("sql"):
                history.append(
                    HumanMessage(
                        content=(
                            f"Previous SQL failed:\nSQL: {state['sql']}\nError: {state['error']}\n\n"
                            "IMPORTANT: Use ONLY views from the schema above: "
                            "vw_medical_analytics, vw_alarms_analytics. Fix the SQL."
                        )
                    )
                )
            elif validation and not validation.get("valid", True) and state.get("sql"):
                issues = validation.get("issues", [])
                issues_text = "\n".join(f"- {i}" for i in issues[:5])
                history.append(
                    HumanMessage(
                        content=(
                            f"SQL verifier found issues:\n{issues_text}\n\n"
                            f"Previous SQL:\n{state['sql']}\n\nFix the SQL."
                        )
                    )
                )

            try:
                response, usage = await invoke_model(llm, [sys_msg, *history])
                raw_content = response.content or ""
                logger.info(
                    "planner_node: raw LLM response len=%d, first_500=%s",
                    len(raw_content),
                    raw_content[:500],
                )
                data = extract_json(raw_content)
                logger.info(
                    "planner_node: parsed keys=%s, sql_len=%d",
                    list(data.keys())[:5],
                    len(data.get("sql", "")),
                )

                token_acc = acc_tokens(state, usage)

                if not data:
                    logger.error(
                        "planner_node: FAILED to parse JSON. Full content (%d chars): %s",
                        len(raw_content),
                        raw_content[:2000],
                    )
                    return {
                        **updates,
                        "error": "Failed to parse JSON response from planner",
                        "sql": "",
                        "retry_count": current_retry,
                        "iteration": current_retry + 1,
                        "_token_usage": token_acc,
                    }

                clean_data = await pii.reveal(data)
                generated_sql = clean_data.get("sql", "")
                purpose = clean_data.get("purpose", "")

                return {
                    **updates,
                    "sql": generated_sql,
                    "purpose": purpose,
                    "format": data.get("format", "table"),
                    "error": "",
                    "messages": [response],
                    "retry_count": current_retry,
                    "iteration": current_retry + 1,
                    "_token_usage": token_acc,
                }
            except Exception as e:
                return {
                    **updates,
                    "error": str(e),
                    "retry_count": current_retry,
                }

    return planner_node


__all__ = ["make_planner_node"]
