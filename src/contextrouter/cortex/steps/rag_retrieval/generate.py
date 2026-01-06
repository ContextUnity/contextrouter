"""Final generation step (pure function).

This keeps the direct-mode graph free of registry registration side effects.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from contextrouter.core import get_core_config
from contextrouter.cortex import AgentState
from contextrouter.modules.models import model_registry

from ...llm import build_rag_prompt
from ...nodes.utils import pipeline_log
from .no_results import no_results_response

logger = logging.getLogger(__name__)


def _last_nonempty_assistant_text(messages: list[Any]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        content = (content or "").strip()
        if content:
            return content
    return ""


class IntentStrategy(ABC):
    @abstractmethod
    async def generate(self, state: AgentState) -> dict[str, Any]:
        raise NotImplementedError


class RAGStrategy(IntentStrategy):
    async def generate(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        retrieved_docs = state.get("retrieved_docs", [])
        user_query = state.get("user_query", "")
        platform = state.get("platform", "api")
        intent_text = state.get("intent_text", user_query)

        logger.debug(
            "RAGStrategy: docs=%d messages=%d query=%s",
            len(retrieved_docs),
            len(messages),
            str(user_query)[:80],
        )

        if not retrieved_docs:
            conversation_history = self._format_history(messages)
            no_results_msg = await no_results_response(
                user_query=str(intent_text or user_query or ""),
                conversation_history=conversation_history,
                prompt_override=str(state.get("no_results_prompt") or ""),
            )
            return {
                "messages": [no_results_msg],
                "citations": [],
                "generation_complete": True,
                "should_retrieve": False,
            }

        prompt_messages = build_rag_prompt(
            messages=messages,
            retrieved_docs=retrieved_docs,
            user_query=intent_text,
            platform=platform,
            style_prompt=str(state.get("style_prompt") or ""),
            rag_system_prompt_override=str(state.get("rag_system_prompt_override") or ""),
            graph_facts=state.get("graph_facts", []) or [],
        )

        core_cfg = get_core_config()
        model_key = (
            core_cfg.models.generation_llm
            if isinstance(core_cfg.models.generation_llm, str)
            and core_cfg.models.generation_llm.strip()
            else core_cfg.models.default_llm
        )
        llm = model_registry.create_llm(model_key).as_chat_model()
        full_content = ""
        async for chunk in llm.astream(prompt_messages):
            if chunk and hasattr(chunk, "content"):
                full_content += str(chunk.content)

        if not full_content.strip():
            full_content = "We apologize, but we encountered an issue generating a response. Please try again later."

        return {
            "messages": [AIMessage(content=full_content)],
            "citations": state.get("citations", []),
            "generation_complete": True,
        }

    def _format_history(self, messages: List[Any]) -> str:
        parts: list[str] = []
        for msg in list(messages)[-10:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = " ".join((content or "").split())
            if content:
                parts.append(f"{role}: {content[:500]}")
        return "\n".join(parts)


class IdentityStrategy(IntentStrategy):
    async def generate(self, state: AgentState) -> dict[str, Any]:
        from contextrouter.cortex.prompting import IDENTITY_PROMPT

        intent_text = state.get("intent_text") or state.get("user_query") or ""
        style_prompt = str(state.get("style_prompt") or "").strip()
        style_context = f"## PERSONA/STYLE CONTEXT\n{style_prompt}" if style_prompt else ""

        system_content = IDENTITY_PROMPT.format(style_context=style_context, query=intent_text)
        prompt_messages = [SystemMessage(content=system_content), HumanMessage(content=intent_text)]

        core_cfg = get_core_config()
        model_key = (
            core_cfg.models.generation_llm
            if isinstance(core_cfg.models.generation_llm, str)
            and core_cfg.models.generation_llm.strip()
            else core_cfg.models.default_llm
        )
        llm = model_registry.create_llm(model_key).as_chat_model()
        full_content = ""
        async for chunk in llm.astream(prompt_messages):
            if chunk and hasattr(chunk, "content"):
                full_content += str(chunk.content)

        # Suggestions for identity are handled by suggest node; keep empty here.
        return {
            "messages": [AIMessage(content=full_content)],
            "citations": [],
            "generation_complete": True,
            "should_retrieve": False,
        }


class TransformStrategy(IntentStrategy):
    async def generate(self, state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", "rewrite")
        intent_text = state.get("intent_text") or state.get("user_query") or ""
        messages = state.get("messages", [])

        last_assistant = _last_nonempty_assistant_text(list(messages or []))
        if not last_assistant:
            return {
                "messages": [
                    AIMessage(content="I don't have a previous assistant message to transform.")
                ],
                "citations": [],
                "generation_complete": True,
                "should_retrieve": False,
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

        core_cfg = get_core_config()
        model_key = (
            core_cfg.models.generation_llm
            if isinstance(core_cfg.models.generation_llm, str)
            and core_cfg.models.generation_llm.strip()
            else core_cfg.models.default_llm
        )
        llm = model_registry.create_llm(model_key).as_chat_model()
        full_content = ""
        async for chunk in llm.astream(prompt_messages):
            if chunk and hasattr(chunk, "content"):
                full_content += str(chunk.content)

        return {
            "messages": [AIMessage(content=full_content)],
            "citations": [],
            "generation_complete": True,
            "should_retrieve": False,
        }


async def generate_response(state: AgentState) -> dict[str, Any]:
    intent = state.get("intent", "rag_and_web")

    strategies: dict[str, IntentStrategy] = {
        "rag_and_web": RAGStrategy(),
        "identity": IdentityStrategy(),
        "translate": TransformStrategy(),
        "summarize": TransformStrategy(),
        "rewrite": TransformStrategy(),
    }

    strategy = strategies.get(intent, strategies["rag_and_web"])
    logger.debug("Generate: intent=%s strategy=%s", intent, type(strategy).__name__)
    pipeline_log("generate.dispatch", intent=intent)

    try:
        return await strategy.generate(state)
    except Exception:
        logger.exception("Generation failed for intent: %s", intent)
        return {
            "messages": [
                AIMessage(content="I apologize, but I encountered an error. Please try again.")
            ],
            "citations": state.get("citations", []),
            "generation_complete": True,
        }


__all__ = ["generate_response"]
