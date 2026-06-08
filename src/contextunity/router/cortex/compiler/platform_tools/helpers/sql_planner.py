"""Planner node — generates SQL from user question.

Pure domain logic: builds prompts, parses LLM JSON response, tracks retries.
Infrastructure concerns (PII, token scoping, model secrets) are handled by
make_secure_node at the graph compilation layer.
"""

from __future__ import annotations

import time
from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_list
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.compiler.state_routing import (
    read_state_dict,
    read_state_input,
    read_state_int,
    read_state_str,
)
from contextunity.router.cortex.types import GraphState, StateUpdate, extract_message_content
from contextunity.router.modules.models import model_registry

from .sql import acc_tokens, extract_json, invoke_model


class SqlPlannerConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_retries: int = Field(default=2, ge=0, le=10)
    pii_masking: bool = False


logger = get_contextunit_logger(__name__)


async def plan_sql(state: GraphState, config: RunnableConfig) -> StateUpdate:
    """Generate SQL from user question.

    Reads all configuration from graph state keys:
        - planner_prompt: system prompt for SQL generation
        - model_key: LLM model identifier
        - messages: conversation history
        - error / validation / sql: retry context
        - metadata.project_config: project-level overrides
    """
    base_update: StateUpdate = {}
    if not state.get("_start_ts"):
        base_update["_start_ts"] = time.monotonic()

    planner_prompt = (
        read_state_str(state, "planner_prompt", default="You are a helpful analyst.") or ""
    )
    messages = state["messages"]
    llm = model_registry.get_llm_with_fallback(read_state_str(state, "model_key"))

    # Retry context
    current_retry = read_state_int(state, "retry_count")
    validation = read_state_dict(state, "validation")
    error = read_state_input(state, "error")
    prev_sql = read_state_input(state, "sql")
    if error or (validation and not validation.get("valid", True)):
        current_retry += 1

    sys_msg = SystemMessage(content=planner_prompt)
    history = list(messages)

    # Build retry feedback from previous errors
    if error and prev_sql:
        history.append(
            HumanMessage(
                content=(f"Previous SQL failed:\nSQL: {prev_sql}\nError: {error}\n\nFix the SQL.")
            )
        )
    elif validation and not validation.get("valid", True) and prev_sql:
        raw_issues = validation.get("issues", [])
        issues = [str(issue) for issue in raw_issues] if is_object_list(raw_issues) else []
        issues_text = "\n".join(f"- {i}" for i in issues[:5])
        history.append(
            HumanMessage(
                content=(
                    f"SQL verifier found issues:\n{issues_text}\n\n"
                    f"Previous SQL:\n{prev_sql}\n\nFix the SQL."
                )
            )
        )

    try:
        response, usage = await invoke_model(
            llm, [sys_msg, *history], config=config, node_name="planner", state=state
        )

        raw_content = extract_message_content(response)
        logger.info(
            "plan_sql: LLM response len=%d, first_500=%s",
            len(raw_content),
            raw_content[:500],
        )
        data = extract_json(raw_content)
        token_acc = acc_tokens(state, usage)

        if not data:
            logger.error("plan_sql: failed to parse JSON from LLM response")
            return {
                **base_update,
                "dynamic": {
                    "error": "Failed to parse JSON response from planner",
                    "sql": "",
                    "retry_count": current_retry,
                    "iteration": current_retry + 1,
                },
                "_token_usage": token_acc,
            }

        generated_sql = data.get("sql", "")
        generated_sql = generated_sql if isinstance(generated_sql, str) else ""
        purpose = data.get("purpose", "")
        purpose = purpose if isinstance(purpose, str) else ""
        output_format = data.get("format", "table")
        output_format = output_format if isinstance(output_format, str) else "table"

        return {
            **base_update,
            "dynamic": {
                "sql": generated_sql,
                "purpose": purpose,
                "format": output_format,
                "error": "",
                "retry_count": current_retry,
                "iteration": current_retry + 1,
            },
            "messages": [response],
            "_token_usage": token_acc,
        }
    except Exception as e:  # graceful-degrade: SQL error returns empty result
        return {
            **base_update,
            "dynamic": {
                "error": str(e),
                "retry_count": current_retry,
            },
        }


__all__ = ["SqlPlannerConfig", "plan_sql"]
