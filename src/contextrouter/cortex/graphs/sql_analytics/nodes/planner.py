"""Planner node — generates SQL from user question."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    StepTimer,
    acc_tokens,
    extract_json,
    invoke_model,
    is_debug,
    step,
)
from contextrouter.cortex.graphs.sql_analytics.pii import pii_anonymize, pii_deanonymize
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.models import model_registry

logger = logging.getLogger(__name__)


def make_planner_node(
    *,
    planner_prompt: str,
    default_model_key: str | None,
    pii_masking: bool,
    anonymize_tool: BaseTool | None,
    deanonymize_tool: BaseTool | None = None,
):
    """Create the planner node closure.

    Atomic PII flow (when pii_masking=True):
        anonymize user messages → LLM (sees PER_xxxx) → deanonymize SQL output.
        The SQL that leaves this node always contains real names.
    """

    async def planner_node(state: SqlAnalyticsState):
        # Record pipeline start time on first entry
        updates: dict[str, Any] = {}
        if not state.get("_start_ts"):
            import time

            updates["_start_ts"] = time.monotonic()

        messages = state["messages"]
        metadata = state.get("metadata") or {}
        model_key = metadata.get("model_key") or default_model_key
        llm = model_registry.get_llm_with_fallback(key=model_key)

        # Increment retry count if returning from error or failed validation
        current_retry = state.get("retry_count", 0)
        validation = state.get("validation") or {}
        if state.get("error") or (validation and not validation.get("valid", True)):
            current_retry += 1

        # Prepare context
        sys_msg = SystemMessage(content=planner_prompt)
        history = list(messages)

        # Extract original user question for trace (before anonymization)
        user_q = ""
        for m in reversed(history):
            if isinstance(m, HumanMessage):
                user_q = m.content
                break

        # ── PII session: generate unique ID per graph execution ──
        # On first planner call, create a new session. On retries, reuse it.
        session_id = metadata.get("session_id", "")
        if pii_masking and not session_id:
            session_id = f"pii-{uuid.uuid4().hex[:12]}"
            metadata = {**metadata, "session_id": session_id}
            updates["metadata"] = metadata

        sub_steps: list[dict] = []

        if pii_masking and anonymize_tool:
            anon_timer = StepTimer()
            masked_history = []
            original_texts: list[str] = []
            masked_texts: list[str] = []
            with anon_timer:
                for msg in history:
                    if isinstance(msg, HumanMessage):
                        original_texts.append(msg.content[:200])
                        masked_text = await pii_anonymize(
                            msg.content,
                            tool=anonymize_tool,
                            session_id=session_id,
                        )
                        masked_texts.append(masked_text[:200])
                        masked_history.append(HumanMessage(content=masked_text))
                    else:
                        masked_history.append(msg)
            history = masked_history
            sub_steps.append(
                step(
                    "pii_anonymize",
                    timer=anon_timer,
                    request={"messages": len(original_texts), "sample": original_texts[:2]},
                    result={"masked_sample": masked_texts[:2]},
                )
            )

        if state.get("error") and state.get("sql"):
            history.append(
                HumanMessage(
                    content=f"Previous SQL failed:\nSQL: {state['sql']}\nError: {state['error']}\nFix it."
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

        llm_timer = StepTimer()
        try:
            with llm_timer:
                response, usage = await invoke_model(llm, [sys_msg, *history])
            data = extract_json(response.content)
            logger.info(
                "planner_node: keys=%s, sql_len=%d",
                list(data.keys())[:5],
                len(data.get("sql", "")),
            )
            if is_debug():
                logger.debug("planner_node: LLM content[:500]=%s", response.content[:500])

            token_acc = acc_tokens(state, usage)

            sub_steps.append(
                step(
                    "planner_llm",
                    timer=llm_timer,
                    request={"question": user_q[:1000], "model": str(model_key)},
                    result={
                        "keys": list(data.keys())[:5] if data else [],
                        "sql_len": len(data.get("sql", "")) if data else 0,
                        "raw_preview": response.content[:300],
                    },
                    tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    cost_usd=usage.get("total_cost", 0),
                )
            )

            if not data:
                sub_steps.append(step("planner", status="error", reason="json_parse_failed"))
                return {
                    **updates,
                    "error": "Failed to parse JSON response from planner",
                    "sql": "",
                    "retry_count": current_retry,
                    "iteration": current_retry + 1,
                    "_token_usage": token_acc,
                    "_steps": sub_steps,
                }

            generated_sql = data.get("sql", "")
            purpose = data.get("purpose", "")

            # ── Deanonymize SQL output — restore PER_xxxx → real names ──
            if pii_masking and deanonymize_tool and generated_sql:
                deanon_timer = StepTimer()
                sql_before = generated_sql
                with deanon_timer:
                    generated_sql = await pii_deanonymize(
                        generated_sql,
                        tool=deanonymize_tool,
                        session_id=session_id,
                    )
                sql_changed = sql_before != generated_sql
                logger.info(
                    "planner deanon SQL: changed=%s, before[50:]=%r, after[50:]=%r",
                    sql_changed,
                    sql_before[-50:],
                    generated_sql[-50:],
                )
                sub_steps.append(
                    step(
                        "pii_deanonymize_sql",
                        timer=deanon_timer,
                        request={"sql_preview": sql_before[:200]},
                        result={"changed": sql_changed, "sql_preview": generated_sql[:200]},
                    )
                )

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
                "_steps": sub_steps,
            }
        except Exception as e:
            sub_steps.append(
                step(
                    "planner_llm",
                    status="error",
                    timer=llm_timer if "llm_timer" in dir() else None,
                    request={"question": user_q[:1000]},
                )
            )
            return {
                **updates,
                "error": str(e),
                "retry_count": current_retry,
                "_steps": sub_steps,
            }

    return planner_node


__all__ = ["make_planner_node"]
