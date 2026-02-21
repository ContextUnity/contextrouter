"""Verifier node — validates SQL correctness using LLM."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    StepTimer,
    acc_tokens,
    extract_json,
    invoke_model,
    step,
)
from contextrouter.cortex.graphs.sql_analytics.pii import pii_anonymize, pii_deanonymize
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.models import model_registry

logger = logging.getLogger(__name__)


def make_verifier_node(
    *,
    verifier_prompt: str | None,
    default_model_key: str | None,
    pii_masking: bool = False,
    anonymize_tool: BaseTool | None = None,
    deanonymize_tool: BaseTool | None = None,
):
    """Create the verifier node closure.

    Atomic PII: anonymize prompt → LLM → deanonymize output.
    """

    async def verifier_node(state: SqlAnalyticsState):
        if not verifier_prompt:
            return {"validation": {"valid": True}, "_steps": [step("verifier", status="skipped")]}

        sql = state.get("sql")
        sql_result = state.get("sql_result") or {}
        if not sql or not sql_result:
            return {"validation": {"valid": True}, "_steps": [step("verifier", status="skipped")]}

        user_q = ""
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                user_q = m.content
                break
            elif isinstance(m, dict) and m.get("role") == "user":
                user_q = m.get("content", "")
                break

        columns = sql_result.get("columns", []) if isinstance(sql_result, dict) else []
        rows = sql_result.get("rows", [])[:5] if isinstance(sql_result, dict) else []
        row_count = sql_result.get("row_count", len(rows)) if isinstance(sql_result, dict) else 0

        prompt = (
            f"Запит користувача: {user_q}\n"
            f"SQL: {sql}\n"
            f"Колонки: {columns}\n"
            f"Перші рядки ({len(rows)}):\n{json.dumps(rows, ensure_ascii=False, default=str)}\n"
            f"Всього рядків: {row_count}"
        )

        # ── Anonymize → LLM → Deanonymize ──
        metadata = state.get("metadata") or {}
        session_id = metadata.get("session_id", "")

        if pii_masking:
            prompt = await pii_anonymize(prompt, tool=anonymize_tool, session_id=session_id)

        messages = [SystemMessage(content=verifier_prompt), HumanMessage(content=prompt)]

        model_key = metadata.get("model_key") or default_model_key
        llm = model_registry.get_llm_with_fallback(key=model_key)

        timer = StepTimer()
        try:
            with timer:
                response, usage = await invoke_model(llm, messages)

            content = response.content
            if pii_masking:
                content = await pii_deanonymize(
                    content, tool=deanonymize_tool, session_id=session_id
                )

            data = extract_json(content)
            issues = data.get("issues", [])
            logger.info("verifier_node: valid=%s, issues=%d", data.get("valid"), len(issues))
            if issues and not data.get("valid", True):
                logger.warning("verifier_node: issues=%s", issues[:3])
            return {
                "validation": data,
                "_token_usage": acc_tokens(state, usage),
                "_steps": [
                    step(
                        "verifier",
                        timer=timer,
                        request={"sql": sql[:500], "question": user_q[:500]},
                        result={"valid": data.get("valid"), "issues": issues[:5]},
                        valid=data.get("valid"),
                        issues=len(issues),
                    )
                ],
            }
        except Exception as e:
            logger.warning("Verifier failed (treating as valid): %s", e)
            return {
                "validation": {"valid": True, "warning": str(e)},
                "_steps": [
                    step("verifier", status="error", timer=timer, request={"sql": sql[:500]}),
                ],
            }

    return verifier_node


__all__ = ["make_verifier_node"]
