"""
Enricher node implementations.

Real LLM calls using model_registry pattern from contextunity.router.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from contextunity.core import get_contextunit_logger

from contextunity.router.cortex.graphs.config_resolution import get_node_manifest_config

from .bidi import EnricherBiDi
from .state import ProductEnricherState

logger = get_contextunit_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found: {path}")


def make_init_credentials():
    async def init_credentials(state: ProductEnricherState) -> dict:
        start = time.time()
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        trace_id = state.get("trace_id", uuid.uuid4().hex[:12])

        # Credentials are resolved at graph execution time via Shield
        # The model_registry handles API key injection internally
        step_trace = {
            "step": "init_credentials",
            "tenant_id": tenant_id,
            "duration_ms": int((time.time() - start) * 1000),
        }

        return {
            "trace_id": trace_id,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return init_credentials


def make_normalize_raw():
    async def normalize_raw(state: ProductEnricherState) -> dict:
        start = time.time()
        products = state.get("dealer_products", [])
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")

        if not products:
            return {
                "normalized_data": [],
                "step_traces": state.get("step_traces", [])
                + [{"step": "normalize_raw", "total": 0, "duration_ms": 0}],
            }

        from contextunity.router.modules.models.registry import model_registry

        node_config = get_node_manifest_config(state, "normalize_raw")
        model_name = node_config.get("model", "openai/gpt-4o-mini")

        llm = model_registry.create_llm(
            model_name,
            tenant_id=tenant_id,
            shield_key_name="normalizer_llm",
        )

        # Build payload for Gardener normalizer
        from contextunity.router.cortex.graphs.commerce.gardener.normalizer import (
            prepare_llm_payload,
        )

        # Minimal taxonomy — in production would be fetched from BiDi
        taxonomy = {"categories": [], "colors": [], "sizes": []}
        system_prompt, user_msg = prepare_llm_payload(
            products=products,
            taxonomy=taxonomy,
            examples=[],
            deterministic_results=[],
        )

        from contextunity.router.modules.models.types import ModelRequest, TextPart

        try:
            response = await llm.generate(
                ModelRequest(system=system_prompt, parts=[TextPart(text=user_msg)])
            )
            content = response.text

            from contextunity.router.cortex.graphs.commerce.gardener.normalizer import (
                _parse_json_response,
            )

            normalized_data = _parse_json_response(content)
        except Exception as e:
            logger.error("normalize_raw LLM error: %s", e)
            normalized_data = []

        step_trace = {
            "step": "normalize_raw",
            "total": len(products),
            "parsed": len(normalized_data),
            "duration_ms": int((time.time() - start) * 1000),
        }

        return {
            "normalized_data": normalized_data,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return normalize_raw


def make_search_images():
    async def search_images(state: ProductEnricherState) -> dict:
        start = time.time()

        urls = {}
        from contextunity.router.core.config.base import get_env

        api_key = get_env("SERPAPI_API_KEY") or ""

        for p in state.get("dealer_products", []):
            product_id = p.get("id", 0)
            query = f"{p.get('brand', '')} {p.get('name', '')}".strip()

            if api_key and query:
                try:
                    from serpapi import GoogleSearch

                    search = GoogleSearch({"q": query, "tbm": "isch", "api_key": api_key, "num": 5})
                    results = search.get_dict()
                    images = results.get("images_results", [])
                    urls[product_id] = [
                        img.get("original") for img in images if img.get("original")
                    ][:5]
                except Exception as e:
                    logger.error("SerpApi error for %s: %s", product_id, e)
                    urls[product_id] = []
            else:
                urls[product_id] = []

        step_trace = {
            "step": "search_images",
            "products": len(urls),
            "duration_ms": int((time.time() - start) * 1000),
        }
        return {
            "google_search_urls": urls,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return search_images


def make_generate_description():
    async def generate_description(state: ProductEnricherState) -> dict:
        start = time.time()
        products = state.get("dealer_products", [])
        normalized = state.get("normalized_data", [])
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")

        from contextunity.router.modules.models.registry import model_registry

        node_config = get_node_manifest_config(state, "generate_description")
        model_name = node_config.get("model", "perplexity/sonar")

        llm = model_registry.create_llm(
            model_name,
            tenant_id=tenant_id,
            shield_key_name="generation_llm",
        )

        # Build normalized lookup
        norm_map = {n.get("id"): n for n in normalized}

        descriptions = {}
        seo = {}

        for p in products:
            pid = p.get("id", 0)
            norm = norm_map.get(pid, {})

            product_info = (
                f"Product: {p.get('name', '')}\n"
                f"Brand: {norm.get('brand') or p.get('brand', '')}\n"
                f"Type: {norm.get('product_type', '')}\n"
                f"Model: {norm.get('model_name', '')}\n"
                f"Extra: {norm.get('extra', '')}\n"
            )

            from contextunity.router.modules.models.types import ModelRequest, TextPart

            try:
                # Ukrainian description
                res_uk = await llm.generate(
                    ModelRequest(
                        system="You are an expert e-commerce copywriter. Write a product description "
                        "in Ukrainian using HTML (paragraphs, lists). Do not wrap in markdown code fences.",
                        parts=[TextPart(text=product_info)],
                    )
                )
                uk_text = res_uk.text.replace("```html", "").replace("```", "").strip()

                # English description
                res_en = await llm.generate(
                    ModelRequest(
                        system="You are an expert e-commerce copywriter. Write a product description "
                        "in English using HTML (paragraphs, lists). Do not wrap in markdown code fences.",
                        parts=[TextPart(text=product_info)],
                    )
                )
                en_text = res_en.text.replace("```html", "").replace("```", "").strip()

                # Apply grammar check
                try:
                    from contextunity.router.cortex.graphs.news_engine.agents.language_tool import (
                        apply_language_tool,
                        close_language_tool,
                        init_language_tool,
                    )

                    init_language_tool(lang="uk")
                    uk_text = await apply_language_tool(uk_text, auto_correct=True)
                    close_language_tool()

                    init_language_tool(lang="en-US")
                    en_text = await apply_language_tool(en_text, auto_correct=True)
                    close_language_tool()
                except Exception as lt_err:
                    logger.warning("Language tool unavailable: %s", lt_err)

                descriptions[pid] = {"uk": uk_text, "en": en_text}

                # Generate SEO metadata
                seo_res = await llm.generate(
                    ModelRequest(
                        system="Generate SEO metadata for a product page. Return JSON: "
                        '{"meta_title": "...", "meta_description": "...", "slug": "..."}',
                        parts=[TextPart(text=product_info)],
                    )
                )
                try:
                    seo_data = json.loads(
                        seo_res.text.replace("```json", "").replace("```", "").strip()
                    )
                    seo[pid] = seo_data
                except (json.JSONDecodeError, Exception):
                    seo[pid] = {"meta_title": "", "meta_description": "", "slug": ""}

            except Exception as e:
                logger.error("generate_description error for %s: %s", pid, e)
                descriptions[pid] = {"uk": f"<p>Error: {e}</p>", "en": f"<p>Error: {e}</p>"}
                seo[pid] = {"meta_title": "", "meta_description": "", "slug": ""}

        step_trace = {
            "step": "generate_description",
            "products": len(products),
            "duration_ms": int((time.time() - start) * 1000),
        }
        return {
            "descriptions": descriptions,
            "seo_metadata": seo,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return generate_description


def make_ner_technologies():
    async def ner_technologies(state: ProductEnricherState) -> dict:
        start = time.time()
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        descriptions = state.get("descriptions", {})

        from contextunity.router.modules.models.registry import model_registry

        node_config = get_node_manifest_config(state, "ner_technologies")
        model_name = node_config.get("model", "openai/gpt-4o-mini")

        llm = model_registry.create_llm(
            model_name,
            tenant_id=tenant_id,
            shield_key_name="normalizer_llm",
        )

        # Combine all descriptions for NER extraction
        combined_text = " ".join(
            f"{d.get('uk', '')} {d.get('en', '')}" for d in descriptions.values()
        )

        prompt = _load_prompt("ner_technologies.txt")

        from contextunity.router.modules.models.types import ModelRequest, TextPart

        try:
            res = await llm.generate(
                ModelRequest(system=prompt, parts=[TextPart(text=combined_text)])
            )
            content = res.text.strip()

            # Parse JSON array
            import re

            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                extracted = json.loads(match.group())
            else:
                extracted = []
        except Exception as e:
            logger.error("ner_technologies error: %s", e)
            extracted = []

        step_trace = {
            "step": "ner_technologies",
            "extracted": len(extracted),
            "duration_ms": int((time.time() - start) * 1000),
        }
        return {
            "extracted_technologies_names": extracted,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return ner_technologies


def make_verify_technologies_bidi():
    async def verify_technologies_bidi(state: ProductEnricherState) -> dict:
        start = time.time()
        extracted = state.get("extracted_technologies_names", [])
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")

        if not extracted:
            return {
                "missing_technologies": [],
                "step_traces": state.get("step_traces", [])
                + [{"step": "verify_technologies_bidi", "missing": 0, "duration_ms": 0}],
            }

        bidi = EnricherBiDi(state.get("trace_id", "t"), tenant_id=tenant_id)
        try:
            missing = await bidi.verify_technologies(extracted)
        except Exception as e:
            logger.error("verify_technologies_bidi error: %s", e)
            missing = extracted  # Assume all missing if BiDi fails

        step_trace = {
            "step": "verify_technologies_bidi",
            "checked": len(extracted),
            "missing": len(missing),
            "duration_ms": int((time.time() - start) * 1000),
        }
        return {
            "missing_technologies": missing,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return verify_technologies_bidi


def should_create_technologies(state: ProductEnricherState) -> str:
    """Conditional edge routing based on missing technologies."""
    missing = state.get("missing_technologies", [])
    if missing:
        return "create"
    return "skip"


def make_create_missing_technology_articles():
    async def create_missing_technology_articles(state: ProductEnricherState) -> dict:
        start = time.time()
        missing = state.get("missing_technologies", [])
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        bidi = EnricherBiDi(state.get("trace_id", "t"), tenant_id=tenant_id)

        from contextunity.router.modules.models.registry import model_registry

        node_config = get_node_manifest_config(state, "create_missing_technology_articles")
        model_name = node_config.get("model", "openai/gpt-4o-mini")

        llm = model_registry.create_llm(
            model_name,
            tenant_id=tenant_id,
            shield_key_name="normalizer_llm",
        )

        prompt_template = _load_prompt("technology_creator.txt")
        created_ids = []

        from contextunity.router.modules.models.types import ModelRequest, TextPart

        for t_name in missing:
            try:
                prompt = prompt_template.replace("{technology_name}", t_name)
                res = await llm.generate(
                    ModelRequest(
                        system="You generate technology encyclopedia entries. Return valid JSON.",
                        parts=[TextPart(text=prompt)],
                    )
                )

                content = res.text.replace("```json", "").replace("```", "").strip()
                tech_data = json.loads(content)
                tech_data["name"] = t_name

                created_id = await bidi.create_wagtail_technology(tech_data)
                created_ids.append(created_id)
            except Exception as e:
                logger.error("create_technology error for %s: %s", t_name, e)

        step_trace = {
            "step": "create_missing_technology_articles",
            "attempted": len(missing),
            "created": len(created_ids),
            "duration_ms": int((time.time() - start) * 1000),
        }
        return {
            "created_technologies_ids": created_ids,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return create_missing_technology_articles


def make_map_attributes():
    async def map_attributes(state: ProductEnricherState) -> dict:
        start = time.time()
        tenant_id = state.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        products = state.get("dealer_products", [])
        normalized = state.get("normalized_data", [])

        from contextunity.router.modules.models.registry import model_registry

        node_config = get_node_manifest_config(state, "map_attributes")
        model_name = node_config.get("model", "openai/gpt-4o-mini")

        llm = model_registry.create_llm(
            model_name,
            tenant_id=tenant_id,
            shield_key_name="normalizer_llm",
        )

        norm_map = {n.get("id"): n for n in normalized}
        mapped = {}

        for p in products:
            pid = p.get("id", 0)
            norm = norm_map.get(pid, {})

            product_info = json.dumps(
                {
                    "name": p.get("name", ""),
                    "brand": norm.get("brand", p.get("brand", "")),
                    "type": norm.get("product_type", ""),
                    "extra": norm.get("extra", ""),
                },
                ensure_ascii=False,
            )

            from contextunity.router.modules.models.types import ModelRequest, TextPart

            try:
                res = await llm.generate(
                    ModelRequest(
                        system="Extract product attributes (Material, Weight, Waterproof rating, Season, etc.) "
                        'as JSON array: [{"name": "Material", "value": "..."}, ...]',
                        parts=[TextPart(text=product_info)],
                    )
                )
                content = res.text.replace("```json", "").replace("```", "").strip()
                attrs = json.loads(content)
                mapped[pid] = attrs if isinstance(attrs, list) else []
            except Exception as e:
                logger.error("map_attributes error for %s: %s", pid, e)
                mapped[pid] = []

        step_trace = {
            "step": "map_attributes",
            "products": len(products),
            "duration_ms": int((time.time() - start) * 1000),
        }
        return {
            "mapped_attributes": mapped,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return map_attributes
