"""Verifier node — validates SQL correctness using LLM.

Pure domain logic: builds verification prompt from SQL + results, parses LLM
validation response. Infrastructure concerns (PII, token scoping) are handled
by make_secure_node at the graph compilation layer.
"""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ConfigDict

from contextunity.router.core.exceptions import RouterLLMError
from contextunity.router.cortex.compiler.platform_tools.helpers.sql import (
    JsonMap,
    ValidationDict,
    acc_tokens,
    extract_json,
    invoke_model,
)
from contextunity.router.cortex.compiler.state_routing import read_state_input, read_state_str
from contextunity.router.cortex.types import GraphState, StateUpdate, extract_message_content
from contextunity.router.modules.models import model_registry
from contextunity.router.modules.models.types import ModelError


class SqlVerifierConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    strict_mode: bool = True


logger = get_contextunit_logger(__name__)


async def verify_sql(state: GraphState, config: RunnableConfig) -> StateUpdate:
    """Validate SQL correctness using LLM.

    Reads all configuration from graph state keys:
        - verifier_prompt: system prompt (skips verification if missing)
        - model_key: LLM model identifier
        - sql: the generated SQL to verify
        - sql_result: execution results (rows, columns, row_count)
        - messages: conversation history (to extract user question)
        - metadata.project_config: project-level overrides
    """
    verifier_prompt = read_state_input(state, "verifier_prompt")
    if not verifier_prompt:
        return {"dynamic": {"validation": {"valid": True}}}

    sql = read_state_input(state, "sql")
    sql_result: object = read_state_input(state, "sql_result", default={}) or {}
    if not sql or not sql_result:
        return {"dynamic": {"validation": {"valid": True}}}

    # Extract user question from messages
    user_q = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_q = extract_message_content(m)
            break
        elif is_object_dict(m) and m.get("role") == "user":
            user_q = str(m.get("content", ""))
            break

    result: JsonMap = (
        {str(key): value for key, value in sql_result.items()} if is_object_dict(sql_result) else {}
    )
    raw_columns = result.get("columns", [])
    raw_rows = result.get("rows", [])
    columns = [str(column) for column in raw_columns] if is_object_list(raw_columns) else []
    rows: list[JsonMap] = (
        [
            {str(key): value for key, value in row.items()}
            for row in raw_rows[:5]
            if is_object_dict(row)
        ]
        if is_object_list(raw_rows)
        else []
    )
    row_count_raw = result.get("row_count", len(rows))
    row_count = row_count_raw if isinstance(row_count_raw, int) else len(rows)

    prompt = (
        f"Запит користувача: {user_q}\n"
        f"SQL: {sql}\n"
        f"Колонки: {columns}\n"
        f"Перші рядки ({len(rows)}):\n{json_dumps(rows, ensure_ascii=False, default=str)}\n"
        f"Всього рядків: {row_count}"
    )

    model_key = read_state_str(state, "model_key")
    llm = model_registry.get_llm_with_fallback(model_key)

    try:
        messages = [SystemMessage(content=str(verifier_prompt)), HumanMessage(content=prompt)]
        response, usage = await invoke_model(
            llm, messages, config=config, node_name="verifier", state=state
        )
        response_content = extract_message_content(response)

        data = extract_json(response_content)
        issues_raw: object = data.get("issues", [])
        issues = issues_raw if is_object_list(issues_raw) else []
        logger.info("verify_sql: valid=%s, issues=%d", data.get("valid"), len(issues))
        if issues and not data.get("valid", True):
            logger.warning("verify_sql: issues=%s", issues[:3])

        validation: ValidationDict = {
            "valid": bool(data.get("valid", True)),
        }
        reason = data.get("reason")
        if isinstance(reason, str):
            validation["reason"] = reason
        hints = data.get("hints")
        if is_object_list(hints):
            validation["hints"] = [str(h) for h in hints]
        if is_object_list(issues):
            validation["issues"] = [str(i) for i in issues]

        return {
            "dynamic": {"validation": validation},
            "_token_usage": acc_tokens(state, usage),
        }
    except (RouterLLMError, ModelError) as e:
        logger.warning("Verifier LLM call failed (treating as valid): %s", e)
        return {
            "dynamic": {"validation": {"valid": True, "warning": "Verifier unavailable"}},
        }
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Verifier response parsing failed (treating as valid): %s", e)
        return {
            "dynamic": {"validation": {"valid": True, "warning": "Verifier parse error"}},
        }


__all__ = ["SqlVerifierConfig", "verify_sql"]
