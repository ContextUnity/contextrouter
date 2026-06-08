"""LLM-node model invocation with atomic prompt privacy."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TypedDict

from contextunity.core.manifest.router import RetryPolicy
from contextunity.core.types import is_object_dict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from contextunity.router.core.exceptions import RouterLLMError
from contextunity.router.cortex.types import GraphState, extract_message_content
from contextunity.router.modules.models.base import BaseLLM
from contextunity.router.modules.models.types import ModelRequest, ModelResponse, TextPart

from .telemetry import model_telemetry


class TokenUsageDict(TypedDict, total=False):
    """Accumulated token usage tracking."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float


def _session_id_for_node(state: GraphState | None, node_name: str | None) -> str:
    if state:
        metadata = state.get("metadata")
        raw = metadata.get("session_id") if is_object_dict(metadata) else None
        if raw:
            return str(raw)
    return f"pii-{node_name or 'llm'}-{uuid.uuid4().hex[:12]}"


def _build_request_from_messages(messages: Sequence[BaseMessage]) -> ModelRequest:
    system_parts: list[str] = []
    parts_text: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(extract_message_content(msg).strip())
        elif isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            parts_text.append(f"{role}: {extract_message_content(msg)}")
        else:
            parts_text.append(extract_message_content(msg))

    return ModelRequest(
        parts=[TextPart(text="\n\n".join(parts_text))],
        system="\n\n".join(system_parts) if system_parts else None,
        response_format="json_object",
    )


async def generate_with_node_privacy(
    llm: BaseLLM,
    request: ModelRequest,
    config: RunnableConfig | None,
    *,
    node_name: str | None,
    state: GraphState | None,
    prompt_version: str | None = None,
    fallback_model_name: str = "",
    retry_policy: RetryPolicy | None = None,
) -> ModelResponse:
    """Run one LLM call using the old atomic privacy contract.

    Flow: anonymize prompt -> trace/provider sees masked prompt -> deanonymize
    model response. Telemetry records the same request passed to the provider.
    """
    request_for_call = request
    pii_session = None

    from contextunity.router.cortex.utils.pii import PiiSession, should_apply_prompt_pii

    if should_apply_prompt_pii():
        pii_session = PiiSession(session_id=_session_id_for_node(state, node_name))
        try:
            request_for_call = pii_session.hide_model_request(request)
            pii_session.emit_tool_result(
                "anonymize_text",
                {"entities_masked": pii_session.entities_masked_total()},
                node_name=node_name,
            )
        except Exception as exc:  # wraps-to-domain: re-raises as typed exception
            pii_session.destroy()
            raise RouterLLMError(
                message=f"Node '{node_name or 'llm'}' prompt anonymization failed",
                node_name=node_name or "llm",
            ) from exc

    try:
        response = await model_telemetry(
            llm,
            request_for_call,
            config,
            prompt_version=prompt_version,
            node_name=node_name,
            state=state,
            fallback_model_name=fallback_model_name,
            retry_policy=retry_policy,
        )
        if pii_session:
            try:
                response = response.model_copy(
                    update={"text": pii_session.reveal_text(response.text)}
                )
                pii_session.emit_tool_result(
                    "deanonymize_text",
                    {"text_restored": response.text != ""},
                    node_name=node_name,
                )
            except Exception as exc:  # wraps-to-domain: re-raises as typed exception
                raise RouterLLMError(
                    message=f"Node '{node_name or 'llm'}' response deanonymization failed",
                    node_name=node_name or "llm",
                ) from exc
        return response
    finally:
        if pii_session:
            pii_session.destroy()


async def invoke_messages_with_node_privacy(
    llm: BaseLLM,
    messages: Sequence[BaseMessage],
    *,
    config: RunnableConfig | None = None,
    prompt_version: str | None = None,
    node_name: str | None = None,
    state: GraphState | None = None,
    fallback_model_name: str = "",
    retry_policy: RetryPolicy | None = None,
) -> tuple[AIMessage, TokenUsageDict]:
    """Bridge LangChain messages to the LLM-node privacy invocation path."""
    response = await generate_with_node_privacy(
        llm,
        _build_request_from_messages(messages),
        config,
        prompt_version=prompt_version,
        node_name=node_name,
        state=state,
        fallback_model_name=fallback_model_name,
        retry_policy=retry_policy,
    )

    usage = response.usage
    usage_dict: TokenUsageDict
    if usage:
        usage_dict = {
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
            "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0),
            "total_cost": usage.total_cost or 0.0,
        }
    else:
        usage_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    return AIMessage(
        content=response.text,
        usage_metadata={
            "input_tokens": usage_dict["input_tokens"],
            "output_tokens": usage_dict["output_tokens"],
            "total_tokens": usage_dict["total_tokens"],
        },
        response_metadata={
            "model_name": response.raw_provider.model_name
            if getattr(response, "raw_provider", None)
            else "unknown",
            "total_cost": usage_dict["total_cost"],
        },
    ), usage_dict


__all__ = [
    "TokenUsageDict",
    "generate_with_node_privacy",
    "invoke_messages_with_node_privacy",
]
