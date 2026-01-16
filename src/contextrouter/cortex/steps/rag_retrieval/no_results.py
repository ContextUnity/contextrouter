"""No-results response helper (pure function).

Lives under `cortex/steps` so direct-mode doesn't import `cortex/nodes`.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage

from contextrouter.core import get_core_config
from contextrouter.modules.observability.langfuse import retrieval_span

from ...llm import get_no_results_response
from ...utils.json import strip_json_fence

logger = logging.getLogger(__name__)


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
    prompt_override: str = "",
) -> AIMessage:
    """Generate a verbose no-results response using an LLM."""
    with retrieval_span(
        name="no_results_response",
        input_data={"query": user_query[:200]},
    ) as span_ctx:
        try:
            core_cfg = get_core_config()
            from contextrouter.modules.models.registry import model_registry
            from contextrouter.modules.models.types import ModelRequest, TextPart

            # Get no_results model with fallback support
            # TODO: Update to use new config structure once breaking changes are applied
            no_results_cfg = core_cfg.models.rag.no_results
            no_results_model_key = no_results_cfg.model or core_cfg.models.default_llm
            fallback_keys = list(no_results_cfg.fallback or [])
            strategy = no_results_cfg.strategy or "fallback"

            model = model_registry.get_llm_with_fallback(
                key=no_results_model_key,
                fallback_keys=fallback_keys,
                strategy=strategy,
                config=core_cfg,
            )

            # Direct model usage with new multimodal interface
            llm = model

            from contextrouter.cortex.prompting import NO_RESULTS_PROMPT

            template = prompt_override.strip() or NO_RESULTS_PROMPT
            system_prompt = template.format(
                query=user_query,
                conversation_history=conversation_history,
            )

            # Build prompt from system and user messages
            full_prompt = f"{system_prompt}\n\n{user_query}"

            request = ModelRequest(
                parts=[TextPart(text=full_prompt)],
                temperature=core_cfg.llm.temperature,
                max_output_tokens=512,
            )

            full_content = ""
            async for event in llm.stream(request):
                if getattr(event, "event_type", None) == "text_delta":
                    full_content += event.delta
                elif getattr(event, "event_type", None) == "final_text":
                    full_content = event.text

            out = _strip_leading_translation_json(full_content)
            span_ctx["output"] = {"response_len": len(out)}
            return AIMessage(content=out)
        except Exception:
            logger.exception("No-results LLM generation failed")
            span_ctx["output"] = {"error": "generation_failed"}
            return AIMessage(content=get_no_results_response())


__all__ = ["no_results_response"]
