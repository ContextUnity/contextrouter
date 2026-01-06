"""No-results response helper (pure function).

Lives under `cortex/steps` so direct-mode doesn't import `cortex/nodes`.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from contextrouter.core.config import get_core_config
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

            # Use configured model key directly (no parsing / provider-specific hacks).
            model_key = (
                core_cfg.models.no_results_llm
                if isinstance(getattr(core_cfg.models, "no_results_llm", None), str)
                and core_cfg.models.no_results_llm.strip()
                else core_cfg.models.default_llm
            )
            llm = model_registry.create_llm(
                model_key,
                temperature=core_cfg.llm.temperature,
                max_output_tokens=512,
                streaming=True,
            ).as_chat_model()

            from contextrouter.cortex.prompting import NO_RESULTS_PROMPT

            template = prompt_override.strip() or NO_RESULTS_PROMPT
            system_prompt = template.format(
                query=user_query,
                conversation_history=conversation_history,
            )

            full_content = ""
            async for chunk in llm.astream(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_query),
                ]
            ):
                if chunk and hasattr(chunk, "content"):
                    full_content += str(chunk.content)

            out = _strip_leading_translation_json(full_content)
            span_ctx["output"] = {"response_len": len(out)}
            return AIMessage(content=out)
        except Exception:
            logger.exception("No-results LLM generation failed")
            span_ctx["output"] = {"error": "generation_failed"}
            return AIMessage(content=get_no_results_response())


__all__ = ["no_results_response"]
