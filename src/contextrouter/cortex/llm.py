"""Prompt building helpers for the cortex.

This module must be side-effect free (no env loading, no model initialization).
Model selection + instantiation lives in `modules/models/*` and is driven by
`contextrouter.core.config.Config`.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.messages import BaseMessage, SystemMessage

from .models import RetrievedDoc
from .prompting import (
    IDENTITY_RESPONSE,
    NO_RESULTS_RESPONSE,
    RAG_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def build_rag_prompt(
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
        retrieved_docs: Retrieved documents from Vertex AI Search
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
        parts = [str(x).strip() for x in graph_facts if isinstance(x, str) and str(x).strip()]
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


def _format_context(docs: list[RetrievedDoc]) -> str:
    """Format retrieved documents into clean context string."""
    if not docs:
        return ""

    parts = []
    for doc in docs:
        content = doc.content if hasattr(doc, "content") else ""
        if not content:
            content = doc.text if hasattr(doc, "text") else ""
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"---\n{content}")

    return "\n\n".join(parts)


def get_identity_response() -> str:
    """Get the identity response for 'who are you' questions."""
    return IDENTITY_RESPONSE


def get_no_results_response() -> str:
    """Get the fallback response when no results are found."""
    return NO_RESULTS_RESPONSE
