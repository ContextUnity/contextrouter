"""
Writer node implementations.
"""

from __future__ import annotations

from contextcore import get_context_unit_logger
from langchain_core.messages import HumanMessage, SystemMessage

from contextrouter.cortex.graphs.config_resolution import get_node_manifest_config

from .state import WriterState

logger = get_context_unit_logger(__name__)


def make_generate_descriptions():
    async def generate_descriptions(state: WriterState) -> dict:
        from contextrouter.modules.models.registry import model_registry

        node_config = get_node_manifest_config(state, "generate")
        model_name = node_config.get("model", "openai/gpt-4o-mini")

        tenant_id = state.get("tenant_id", "traverse")

        llm = model_registry.create_llm(
            model_name, tenant_id=tenant_id, shield_key_name="generation_llm"
        )

        title = state.get("title", "")
        product_type = state.get("product_type", "")
        brand = state.get("brand", "")
        model = state.get("model", "")
        extra = state.get("extra", "")

        prompt = (
            f"Product: {title}\n"
            f"Type: {product_type}\n"
            f"Brand: {brand}\n"
            f"Model: {model}\n"
            f"Extra characteristics: {extra}\n"
        )

        sys_uk = SystemMessage(
            content="You are an expert e-commerce copywriter. Write a clean, concise product description in Ukrainian using HTML formatting (paragraphs, lists). Do not wrap in markdown tags like ```html."
        )
        sys_en = SystemMessage(
            content="You are an expert e-commerce copywriter. Write a clean, concise product description in English using HTML formatting (paragraphs, lists). Do not wrap in markdown tags like ```html."
        )

        try:
            res_uk = await llm.ainvoke([sys_uk, HumanMessage(content=prompt)])
            uk_content = res_uk.content if hasattr(res_uk, "content") else str(res_uk)
            uk_content = uk_content.replace("```html", "").replace("```", "").strip()

            # Apply grammar and spell check for uk
            from contextrouter.cortex.graphs.news_engine.agents.language_tool import (
                apply_language_tool,
                close_language_tool,
                init_language_tool,
            )

            init_language_tool(lang="uk")
            try:
                uk_content = await apply_language_tool(uk_content, auto_correct=True)
            finally:
                close_language_tool()

        except Exception as e:
            logger.error("Error generating UK description: %s", e)
            uk_content = f"<p>Generation failed: {e}</p>"

        try:
            res_en = await llm.ainvoke([sys_en, HumanMessage(content=prompt)])
            en_content = res_en.content if hasattr(res_en, "content") else str(res_en)
            en_content = en_content.replace("```html", "").replace("```", "").strip()

            # Apply grammar and spell check for en
            from contextrouter.cortex.graphs.news_engine.agents.language_tool import (
                apply_language_tool,
                close_language_tool,
                init_language_tool,
            )

            init_language_tool(lang="en-US")
            try:
                en_content = await apply_language_tool(en_content, auto_correct=True)
            finally:
                close_language_tool()

        except Exception as e:
            logger.error("Error generating EN description: %s", e)
            en_content = f"<p>Generation failed: {e}</p>"

        return {
            "descriptions": {
                "uk": uk_content,
                "en": en_content,
            }
        }

    return generate_descriptions
