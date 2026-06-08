"""Final generation step (pure function).
This keeps the direct-mode graph free of registry registration side effects.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import ClassVar, Protocol

from contextunity.core import get_contextunit_logger
from contextunity.core.narrowing import as_str
from contextunity.core.types import is_object_list
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from contextunity.router.cortex.compiler.platform_tools.prompts import (
    NO_RESULTS_RESPONSE,
    RAG_SYSTEM_PROMPT,
)
from contextunity.router.cortex.config_resolution import get_node_manifest_config
from contextunity.router.cortex.types import GraphState, StateUpdate, extract_message_content
from contextunity.router.cortex.utils.messages import format_conversation_history
from contextunity.router.cortex.utils.pipeline import pipeline_log
from contextunity.router.modules.models import model_registry
from contextunity.router.modules.models.types import ModelRequest, ModelStreamEvent, TextPart
from contextunity.router.modules.retrieval.rag.models import RetrievedDoc

from .no_results import no_results_response


class GenerateConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=1, le=128000)


logger = get_contextunit_logger(__name__)


def _dynamic_str(dyn: dict[str, object], key: str, *, default: str = "") -> str:
    return as_str(dyn.get(key, default), default=default)


def _graph_facts_from_dynamic(dyn: dict[str, object]) -> list[str]:
    raw: object = dyn.get("graph_facts", [])
    if isinstance(raw, str):
        return [raw] if raw else []
    if is_object_list(raw):
        graph_facts: list[str] = []
        for raw_item in raw:
            if isinstance(raw_item, str):
                graph_facts.append(raw_item)
        return graph_facts
    return []


def _format_context(docs: Sequence[RetrievedDoc]) -> str:
    """Join retrieved document contents into a ``---``-delimited context block.

    Args:
        docs: Sequence of retrieved documents from the RAG pipeline.

    Returns:
        Concatenated context string, or empty string if no documents.
    """
    if not docs:
        return ""

    parts: list[str] = []
    for doc in docs:
        content = doc.content
        if not content:
            content = ""
        parts.append(f"---\n{content}")

    return "\n\n".join(parts)


def _build_rag_prompt(
    messages: Sequence[BaseMessage],
    retrieved_docs: list[RetrievedDoc],
    user_query: str,
    platform: str = "api",
    *,
    style_prompt: str = "",
    no_results_message: str = "",
    rag_system_prompt_override: str = "",
    graph_facts: Sequence[str] | str = "",
) -> list[BaseMessage]:
    """Build the prompt for RAG generation.

    Args:
        messages: Conversation history
        retrieved_docs: Retrieved documents from storage providers
        user_query: Current user query
        platform: Platform source (used for tracing only)
        style_prompt: Optional persona/style instructions
        no_results_message: Custom no-results message
        rag_system_prompt_override: Custom RAG system prompt
        graph_facts: Knowledge graph facts

    Returns:
        List of messages to send to the LLM
    """
    context = _format_context(retrieved_docs)

    # Add graph facts into the context payload under a dedicated section.
    graph_facts_text = ""
    if isinstance(graph_facts, str):
        graph_facts_text = graph_facts.strip()
    else:
        parts = [graph_fact.strip() for graph_fact in graph_facts if graph_fact.strip()]
        graph_facts_text = "\n".join(parts).strip()

    if context and graph_facts_text:
        context = (
            context.rstrip()
            + "\n\n=== GRAPH FACTS (Use for Logic/Reasoning) ===\n"
            + graph_facts_text
        )
    elif not context and graph_facts_text:
        context = "=== GRAPH FACTS (Use for Logic/Reasoning) ===\n" + graph_facts_text

    if not context:
        override = no_results_message.strip()
        system_content = override if override else NO_RESULTS_RESPONSE
    else:
        tmpl = rag_system_prompt_override.strip() or RAG_SYSTEM_PROMPT
        system_content = tmpl.format(
            query=user_query,
            context=context,
            graph_facts=graph_facts_text,
        )

    if logger.isEnabledFor(logging.DEBUG):
        logger.info(
            "build_rag_prompt platform=%s has_context=%s style_len=%d system_preview=%r",
            platform,
            bool(context),
            len(style_prompt.strip()),
            (system_content or "")[:300],
        )

    system_content = (system_content or "").rstrip()
    style = style_prompt.strip()
    if style:
        system_content = f"{system_content}\n\n{style}"

    prompt_messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    prompt_messages.extend(messages)

    return prompt_messages


def _last_nonempty_assistant_text(messages: list[BaseMessage]) -> str:
    """Scan messages in reverse for the last non-empty ``AIMessage`` content.

    Args:
        messages: Conversation history to scan.

    Returns:
        Stripped text of the last assistant message, or empty string.
    """
    for msg in reversed(messages or []):
        if not isinstance(msg, AIMessage):
            continue
        content = extract_message_content(msg)
        content = (content or "").strip()
        if content:
            return content
    return ""


class SupportsStream(Protocol):
    """Structural type for model instances that expose async streaming."""

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Yield model stream events for the given request.

        Args:
            request: Model request payload.

        Returns:
            Async iterator of stream events.
        """
        ...


async def _run_generation(model_instance: SupportsStream, request: ModelRequest) -> str:
    """Stream LLM output and accumulate into a complete response string.

    Handles ``text_delta``, ``final_text``, and ``error`` event types.
    Prefers ``final_text`` if it is at least as long as accumulated deltas.

    Args:
        model_instance: Model with streaming capability.
        request: Prepared model request.

    Returns:
        Full generated text content.
    """
    full_content = ""
    async for event in model_instance.stream(request):
        event_type = getattr(event, "event_type", None)
        if event_type == "text_delta":
            delta_raw: object = getattr(event, "delta", "")
            full_content += delta_raw if isinstance(delta_raw, str) else str(delta_raw)
        elif event_type == "final_text":
            text_value_raw: object = getattr(event, "text", "")
            final_text = text_value_raw if isinstance(text_value_raw, str) else str(text_value_raw)
            if len(final_text) >= len(full_content):
                full_content = final_text
        elif event_type == "error":
            logger.error("Generation error: %s", getattr(event, "error", "unknown"))

    return full_content


def _build_model_request(
    messages: Sequence[BaseMessage], merge_system: bool = False
) -> ModelRequest:
    """Convert LangChain messages into a provider-agnostic ``ModelRequest``.

    Separates ``SystemMessage`` content into ``request.system`` (unless
    ``merge_system`` is ``True``) and concatenates remaining messages
    into ``TextPart`` list.

    Args:
        messages: Ordered LangChain messages.
        merge_system: If ``True``, treat system messages as regular content.

    Returns:
        A ``ModelRequest`` ready for LLM invocation.
    """
    system_parts: list[str] = []
    other_parts: list[str] = []

    for msg in messages:
        content = extract_message_content(msg)
        if not content.strip():
            continue

        if isinstance(msg, SystemMessage) and not merge_system:
            system_parts.append(content)
        else:
            other_parts.append(content)

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    user_prompt = "\n\n".join(other_parts)

    # Some providers require at least one user message/part even when a system prompt is present.
    # Ensure we always send at least one TextPart to avoid provider-side validation errors.
    if not user_prompt:
        user_prompt = ""

    return ModelRequest(
        system=system_prompt,
        parts=[TextPart(text=user_prompt)],
    )


class IntentStrategy(Protocol):
    """Protocol for intent-specific generation strategies."""

    async def generate(self, state: GraphState) -> StateUpdate:
        """Generate a response for the given graph state.

        Args:
            state: Graph execution state with intent and context.

        Returns:
            State update containing generated messages and metadata.
        """
        ...


class RAGStrategy(IntentStrategy):
    """Generation strategy backed by retrieved context (RAG or web search)."""

    @override
    async def generate(self, state: GraphState) -> StateUpdate:
        """Generate a response using retrieved documents and graph facts.

        Falls back to ``no_results_response`` when no documents are
        available. Uses manifest ``generate`` node config for model selection.

        Args:
            state: Graph state with ``retrieved_docs`` in ``dynamic``.

        Returns:
            State update with generated ``AIMessage`` and citations.
        """
        dyn = state.get("dynamic", {})
        messages = state.get("messages", [])
        retrieved_docs_raw = dyn.get("retrieved_docs", [])
        retrieved_docs: list[RetrievedDoc] = []
        if is_object_list(retrieved_docs_raw):
            for raw_doc in retrieved_docs_raw:
                if isinstance(raw_doc, RetrievedDoc):
                    retrieved_docs.append(raw_doc)
        user_query_raw = dyn.get("user_query", "")
        user_query = user_query_raw if isinstance(user_query_raw, str) else ""
        platform_raw = dyn.get("platform", "api")
        platform = platform_raw if isinstance(platform_raw, str) else "api"
        intent_text_raw = dyn.get("intent_text", user_query)
        intent_text = intent_text_raw if isinstance(intent_text_raw, str) else user_query

        logger.debug(
            "RAGStrategy: docs=%d messages=%d query=%s",
            len(retrieved_docs),
            len(messages),
            str(user_query)[:80],
        )

        if not retrieved_docs:
            conversation_history = format_conversation_history(messages)
            no_results_msg = await no_results_response(
                user_query=str(intent_text or user_query or ""),
                conversation_history=conversation_history,
                state=state,
                prompt_override=str(state.get("no_results_prompt") or ""),
            )
            return {
                "messages": [no_results_msg],
                "dynamic": {
                    "citations": [],
                    "generation_complete": True,
                    "should_retrieve": False,
                },
            }

        prompt_messages = _build_rag_prompt(
            messages=messages,
            retrieved_docs=retrieved_docs,
            user_query=intent_text,
            platform=platform,
            style_prompt=str(dyn.get("style_prompt") or "").strip(),
            rag_system_prompt_override=str(dyn.get("rag_system_prompt_override") or "").strip(),
            graph_facts=_graph_facts_from_dynamic(dyn),
        )

        node_config = get_node_manifest_config(state, "generate")
        model_key = node_config.get("model") or "vertex/gemini-2.5-pro"

        llm = model_registry.create_llm(model_key)

        request = _build_model_request(prompt_messages, merge_system=True)
        full_content = await _run_generation(llm, request)

        if not full_content.strip():
            full_content = "We apologize, but we encountered an issue generating a response. Please try again later."

        return {
            "messages": [AIMessage(content=full_content)],
            "dynamic": {
                "citations": dyn.get("citations", []),
                "generation_complete": True,
            },
        }


class IdentityStrategy(IntentStrategy):
    """Generation strategy for identity / greeting intents (no retrieval)."""

    @override
    async def generate(self, state: GraphState) -> StateUpdate:
        """Generate an identity response using the persona prompt.

        Args:
            state: Graph state with ``intent_text`` in ``dynamic``.

        Returns:
            State update with generated ``AIMessage``.
        """
        from contextunity.router.cortex.compiler.platform_tools.prompts import IDENTITY_PROMPT

        dyn = state.get("dynamic", {})
        intent_text = _dynamic_str(dyn, "intent_text", default=_dynamic_str(dyn, "user_query"))
        style_prompt = str(dyn.get("style_prompt") or "").strip()
        style_context = f"## PERSONA/STYLE CONTEXT\n{style_prompt}" if style_prompt else ""

        system_content = IDENTITY_PROMPT.format(style_context=style_context, query=intent_text)
        prompt_messages = [SystemMessage(content=system_content), HumanMessage(content=intent_text)]

        node_config = get_node_manifest_config(state, "generate")
        model_key = node_config.get("model") or "vertex/gemini-2.5-flash-lite"

        llm = model_registry.create_llm(model_key)

        request = _build_model_request(prompt_messages, merge_system=True)
        full_content = await _run_generation(llm, request)

        # Suggestions for identity are handled by suggest node; keep empty here.
        return {
            "messages": [AIMessage(content=full_content)],
            "dynamic": {
                "citations": [],
                "generation_complete": True,
                "should_retrieve": False,
            },
        }


class TransformStrategy(IntentStrategy):
    """Generation strategy for text transformation (translate / summarize / rewrite)."""

    @override
    async def generate(self, state: GraphState) -> StateUpdate:
        """Transform the last assistant message according to user intent.

        Applies translate, summarize, or rewrite instructions to the
        most recent non-empty ``AIMessage`` in the conversation history.

        Args:
            state: Graph state with ``intent`` type in ``dynamic``.

        Returns:
            State update with transformed ``AIMessage``.
        """
        dyn = state.get("dynamic", {})
        intent_raw = dyn.get("intent", "rewrite")
        intent = intent_raw if isinstance(intent_raw, str) else "rewrite"
        intent_text = _dynamic_str(dyn, "intent_text", default=_dynamic_str(dyn, "user_query"))
        messages = state.get("messages", [])

        last_assistant = _last_nonempty_assistant_text(list(messages or []))
        if not last_assistant:
            return {
                "messages": [
                    AIMessage(content="I don't have a previous assistant message to transform.")
                ],
                "dynamic": {
                    "citations": [],
                    "generation_complete": True,
                    "should_retrieve": False,
                },
            }

        instructions = {
            "translate": "Translate the text below. Preserve meaning. Keep formatting (markdown).",
            "summarize": "Summarize the text below in a concise way.",
            "rewrite": "Rewrite the text below according to the user's instruction. Improve clarity.",
        }
        instruction = instructions.get(intent, instructions["rewrite"])
        prompt_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content=f"User instruction: {intent_text}\n\nTask: {instruction}\n\nTEXT:\n{last_assistant}"
            ),
        ]

        node_config = get_node_manifest_config(state, "generate")
        model_key = node_config.get("model") or "vertex/gemini-2.5-flash-lite"

        llm = model_registry.create_llm(model_key)

        request = _build_model_request(prompt_messages, merge_system=True)
        full_content = await _run_generation(llm, request)

        return {
            "messages": [AIMessage(content=full_content)],
            "dynamic": {
                "citations": [],
                "generation_complete": True,
                "should_retrieve": False,
            },
        }


async def generate_response(state: GraphState) -> StateUpdate:
    """Dispatch generation to the appropriate intent strategy.

    Routes to ``RAGStrategy``, ``IdentityStrategy``, or
    ``TransformStrategy`` based on ``state.dynamic.intent``.
    Falls back to ``RAGStrategy`` for unknown intents.

    Args:
        state: Graph execution state with ``dynamic.intent``.

    Returns:
        State update from the selected strategy.
    """
    dyn = state.get("dynamic", {})
    intent_raw = dyn.get("intent", "rag")
    intent = intent_raw if isinstance(intent_raw, str) else "rag"

    strategies: dict[str, IntentStrategy] = {
        "rag": RAGStrategy(),
        "web_search": RAGStrategy(),  # Both retrieve external context
        "identity": IdentityStrategy(),
        "translate": TransformStrategy(),
        "summarize": TransformStrategy(),
        "rewrite": TransformStrategy(),
    }

    strategy = strategies.get(intent, strategies["rag"])
    logger.debug("Generate: intent=%s strategy=%s", intent, type(strategy).__name__)
    pipeline_log("generate.dispatch", intent=intent)

    try:
        return await strategy.generate(state)
    except Exception:  # graceful-degrade: tool failure returns empty result
        logger.exception("Generation failed for intent: %s", intent)
        return {
            "messages": [
                AIMessage(content="I apologize, but I encountered an error. Please try again.")
            ],
            "dynamic": {
                "citations": dyn.get("citations", []),
                "generation_complete": True,
            },
        }


__all__ = ["generate_response"]
