"""No-results response helper (pure function).

Lives under `cortex/steps` so direct-mode doesn't import `cortex/nodes`.
"""

from __future__ import annotations

import re
from typing import ClassVar

from contextunity.core import get_contextunit_logger
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

from contextunity.router.cortex.compiler.platform_tools.prompts import NO_RESULTS_RESPONSE
from contextunity.router.cortex.config_resolution import get_node_manifest_config
from contextunity.router.cortex.types import GraphState
from contextunity.router.cortex.utils.json import strip_json_fence
from contextunity.router.modules.observability.langfuse import retrieval_span


class NoResultsConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str | None = None


logger = get_contextunit_logger(__name__)


def _strip_leading_translation_json(text: str) -> str:
    raw = strip_json_fence(text)

    if not raw.startswith("{"):
        return text

    if '"english_query"' not in raw[:120]:
        return text

    m = re.match(r"\{.*?\}\s*", raw, flags=re.DOTALL)
    if not m:
        return text

    stripped = raw[m.end() :].lstrip("\n\r\t -")
    return stripped or text


async def no_results_response(
    *,
    user_query: str,
    conversation_history: str,
    state: GraphState,
    prompt_override: str = "",
) -> AIMessage:
    """Generate a verbose no-results response using an LLM."""
    with retrieval_span(
        name="no_results_response",
        input_data={"query": user_query[:200]},
    ) as span_ctx:
        try:
            from contextunity.router.modules.models.registry import model_registry
            from contextunity.router.modules.models.types import ModelRequest, TextPart

            node_config = get_node_manifest_config(state, "generate")
            model_key = node_config.get("model") or "vertex/gemini-2.5-flash-lite"

            llm = model_registry.create_llm(model_key)

            from contextunity.router.cortex.compiler.platform_tools.prompts import NO_RESULTS_PROMPT

            template = prompt_override.strip() or NO_RESULTS_PROMPT
            system_prompt = template.format(
                query=user_query,
                conversation_history=conversation_history,
            )

            # Build prompt from system and user messages
            full_prompt = f"{system_prompt}\n\n{user_query}"

            request = ModelRequest(
                parts=[TextPart(text=full_prompt)],
                temperature=0.0,
                max_output_tokens=512,
            )

            full_content = ""
            async for event in llm.stream(request):
                event_type: object = getattr(event, "event_type", None)
                if event_type == "text_delta":
                    delta: object = getattr(event, "delta", "")
                    full_content += delta if isinstance(delta, str) else str(delta)
                elif event_type == "final_text":
                    text_value: object = getattr(event, "text", "")
                    full_content = text_value if isinstance(text_value, str) else str(text_value)

            out = _strip_leading_translation_json(full_content)
            span_ctx["output"] = {"response_len": len(out)}
            return AIMessage(content=out)
        except Exception:  # graceful-degrade: tool failure returns empty result
            logger.exception("No-results LLM generation failed")
            span_ctx["output"] = {"error": "generation_failed"}
            return AIMessage(content=NO_RESULTS_RESPONSE)


__all__ = ["no_results_response", "NoResultsConfig"]
