"""Verifier node — validates SQL correctness using LLM."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    acc_tokens,
    extract_json,
    invoke_model,
)
from contextrouter.cortex.graphs.sql_analytics.pii import PiiSession
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.models import model_registry

logger = logging.getLogger(__name__)


def make_verifier_node(
    *,
    verifier_prompt: str | None,
    default_model_key: str | None,
    fallback_keys: list[str] | None = None,
    pii_masking: bool = False,
    anonymize_tool: BaseTool | None = None,
    deanonymize_tool: BaseTool | None = None,
):
    """Create the verifier node closure.

    Atomic PII: anonymize prompt → LLM → deanonymize output.

    Tracing: All tool/LLM calls go through LangChain and are captured
    automatically by BrainAutoTracer callbacks — no manual _steps needed.
    """

    async def verifier_node(state: SqlAnalyticsState):
        if not verifier_prompt:
            return {"validation": {"valid": True}}

        sql = state.get("sql")
        sql_result = state.get("sql_result") or {}
        if not sql or not sql_result:
            return {"validation": {"valid": True}}

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

        metadata = state.get("metadata") or {}
        session_id = metadata.get("session_id", "")

        async with PiiSession(
            sub_steps=[],  # unused — callbacks handle tracing
            session_id=session_id,
            anonymize_tool=anonymize_tool if pii_masking else None,
            deanonymize_tool=deanonymize_tool if pii_masking else None,
        ) as pii:
            prompt = await pii.hide(prompt)

            messages = [SystemMessage(content=verifier_prompt), HumanMessage(content=prompt)]

            llm = model_registry.get_llm_with_fallback(
                default_model_key, fallback_keys=fallback_keys
            )

            try:
                response, usage = await invoke_model(llm, messages)
                content = response.content
                content = await pii.reveal(content)

                data = extract_json(content)
                issues = data.get("issues", [])
                logger.info("verifier_node: valid=%s, issues=%d", data.get("valid"), len(issues))
                if issues and not data.get("valid", True):
                    logger.warning("verifier_node: issues=%s", issues[:3])

                return {
                    "validation": data,
                    "_token_usage": acc_tokens(state, usage),
                }
            except Exception as e:
                logger.warning("Verifier failed (treating as valid): %s", e)
                return {
                    "validation": {"valid": True, "warning": str(e)},
                }

    return verifier_node


__all__ = ["make_verifier_node"]
