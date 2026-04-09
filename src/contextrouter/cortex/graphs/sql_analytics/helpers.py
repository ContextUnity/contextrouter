"""Shared helpers for SQL analytics graph nodes."""

from __future__ import annotations

import json
import re
from typing import Any

from contextcore import get_context_unit_logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from contextrouter.core import get_core_config
from contextrouter.modules.models.types import ModelRequest, TextPart

logger = get_context_unit_logger(__name__)

# Verbose graph-level logging — lazy to avoid import-time config access
_dbg_cache: bool | None = None


def is_debug() -> bool:
    """Check if graph-level debug logging is enabled."""
    global _dbg_cache
    if _dbg_cache is None:
        try:
            _dbg_cache = get_core_config().debug_graph_messages
        except Exception:
            _dbg_cache = False
    return _dbg_cache


def extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM response text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Last resort: bracket-counting extraction
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


async def invoke_model(
    model: Any,
    messages: list[BaseMessage],
    *,
    config: Any = None,
    prompt_version: str | None = None,
) -> tuple[AIMessage, dict]:
    """Bridge LangChain messages to Router model.generate() API.

    Extracts system prompt from SystemMessage, concatenates remaining
    messages into a single TextPart, and returns result as AIMessage.
    """
    system_parts: list[str] = []
    parts_text: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            if isinstance(msg.content, str):
                system_parts.append(msg.content.strip())
            elif isinstance(msg.content, list):
                # Handle edge cases where content might be a list
                system_parts.append(str(msg.content).strip())
        elif isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            parts_text.append(f"{role}: {msg.content}")
        else:
            parts_text.append(str(msg.content))

    system = "\n\n".join(system_parts) if system_parts else None

    user_text = "\n\n".join(parts_text)
    logger.info(
        "invoke_model: system_len=%s, user_text_len=%d",
        len(system) if system else 0,
        len(user_text),
    )

    request = ModelRequest(
        parts=[TextPart(text=user_text)],
        system=system,
        response_format="json_object",
    )
    logger.info("invoke_model: request.system len=%s", len(request.system) if request.system else 0)

    # Standard LangChain LLM tracing via config callbacks
    # This automatically notifies Langfuse, BrainAutoTracer, and any other handlers
    from langchain_core.runnables.config import get_async_callback_manager_for_config

    cb_manager = get_async_callback_manager_for_config(config or {})

    # Extract actual model name instead of FallbackModel wrapper class
    llm_name = getattr(model, "name", None)
    if not llm_name:
        candidates = getattr(model, "_candidate_keys", None)
        if candidates and isinstance(candidates, list) and candidates:
            llm_name = candidates[0]
    llm_name = llm_name or getattr(model.__class__, "__name__", "LLM")

    rms = await cb_manager.on_chat_model_start(
        serialized={"name": llm_name},
        messages=[messages],
    )
    run_manager = rms[0] if rms else None

    if prompt_version:
        from contextrouter.cortex.runtime_context import append_provenance

        append_provenance(f"prompt:{llm_name}:{prompt_version}")

    response = await model.generate(request)

    if run_manager:
        try:
            from langchain_core.outputs import ChatGeneration, LLMResult

            gen = ChatGeneration(message=AIMessage(content=response.text), text=response.text)

            usage = response.usage
            llm_output = {}
            if usage:
                llm_output["token_usage"] = {
                    "prompt_tokens": usage.input_tokens or 0,
                    "completion_tokens": usage.output_tokens or 0,
                    "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0),
                    "total_cost": usage.total_cost or 0.0,
                }

            # Use actual model name from response (what the provider returned)
            if response.raw_provider:
                llm_output["model_name"] = response.raw_provider.model_name
            else:
                candidate_keys = getattr(model, "_candidate_keys", None)
                if candidate_keys:
                    llm_output["model_name"] = candidate_keys[0]

            res = LLMResult(generations=[[gen]], llm_output=llm_output)
            await run_manager.on_llm_end(res)
        except Exception as e:
            logger.warning("Failed to end callback manager run: %s", e)

    usage = response.usage
    if usage:
        usage_dict = {
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
            "total_cost": usage.total_cost or 0.0,
        }
    else:
        usage_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0,
        }
    return AIMessage(
        content=response.text,
        usage_metadata={
            "input_tokens": usage_dict["input_tokens"],
            "output_tokens": usage_dict["output_tokens"],
            "total_tokens": usage_dict["input_tokens"] + usage_dict["output_tokens"],
        },
        response_metadata={
            "model_name": response.raw_provider.model_name
            if getattr(response, "raw_provider", None)
            else "unknown",
            "total_cost": usage_dict["total_cost"],
        },
    ), usage_dict


def acc_tokens(state: dict, usage: dict) -> dict:
    """Merge new usage into accumulated _token_usage."""
    prev = state.get("_token_usage") or {}
    in_tok = prev.get("input_tokens", 0) + usage.get("input_tokens", 0)
    out_tok = prev.get("output_tokens", 0) + usage.get("output_tokens", 0)
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "total_cost": prev.get("total_cost", 0.0) + usage.get("total_cost", 0.0),
    }


def validate_sql_syntax(sql: str) -> str | None:
    """Quick pre-validation for obvious SQL syntax errors.

    Returns error message if invalid, None if OK.
    """
    if re.search(r"\bIN\s*\(\s*\)", sql, re.IGNORECASE):
        return "SQL contains empty IN() clause. Use a valid list or remove the condition."

    if sql.count("(") != sql.count(")"):
        return f"Unbalanced parentheses: {sql.count('(')} opening vs {sql.count(')')} closing."

    return None


__all__ = [
    "acc_tokens",
    "extract_json",
    "invoke_model",
    "is_debug",
    "validate_sql_syntax",
]
