"""Intent detection step (pure function).

This is a copy of the intent logic without agent registration side effects.
"""

from __future__ import annotations

import json
import logging
import re
import time

from contextrouter.cortex import AgentState, get_graph_service, get_last_user_query
from contextrouter.modules.observability import retrieval_span

from ...nodes.utils import pipeline_log
from ...utils.json import strip_json_fence
from ...utils.taxonomy_loader import (
    get_taxonomy_canonical_map,
    get_taxonomy_top_level_categories,
)

logger = logging.getLogger(__name__)


def _build_taxonomy_context() -> str:
    try:
        cats_line = get_taxonomy_top_level_categories()
        canonical_map = get_taxonomy_canonical_map()

        if not cats_line and not canonical_map:
            logger.warning(
                "Taxonomy not available. Taxonomy enrichment disabled in intent detection."
            )
            return ""

        parts: list[str] = []
        if cats_line:
            parts.append(cats_line)
        if canonical_map:
            examples = list(canonical_map.items())[:10]
            mappings = [f"'{k}' -> '{v}'" for k, v in examples if k.lower() != v.lower()]
            if mappings:
                parts.append(f"Example synonyms: {', '.join(mappings[:5])}")

        return ("\n\n".join(parts) + "\n") if parts else ""
    except Exception as e:
        logger.warning("Failed to build taxonomy context: %s. Taxonomy enrichment disabled.", e)
        return ""


def _extract_taxonomy_concepts(
    query: str, retrieval_queries: list[str]
) -> tuple[list[str], list[str]]:
    categories: set[str] = set()
    concepts: set[str] = set()

    try:
        graph_service = get_graph_service()
        canonical_map = graph_service.get_canonical_map() or get_taxonomy_canonical_map()
        if not canonical_map:
            return [], []

        all_text = " ".join([query] + retrieval_queries).lower()
        for synonym, canonical in canonical_map.items():
            if synonym in all_text or canonical.lower() in all_text:
                concepts.add(canonical)
                category = graph_service.get_category_for_concept(canonical)
                if category:
                    categories.add(category)
    except Exception as e:
        logger.warning("Failed to extract taxonomy concepts: %s. Taxonomy extraction disabled.", e)

    return list(categories)[:5], list(concepts)[:10]


async def detect_intent(state: AgentState) -> dict[str, object]:
    """Detect intent and derive retrieval queries for the current user message."""
    user_query = (
        get_last_user_query(state.get("messages") or [])
        or ((state.get("user_query") or "").strip()[-500:])
    )
    pipeline_log("detect_intent.in", user_query=user_query)

    if not user_query:
        return {
            "intent": "rag_and_web",
            "intent_text": "",
            "user_language": "",
            "ignore_history": False,
            "retrieval_queries": [],
            "should_retrieve": False,
            "retrieved_docs": [],
            "citations": [],
            "search_suggestions": [],
        }

    ignore_history_hint = user_query.lower().startswith("new topic")

    from contextrouter.core import get_core_config
    from contextrouter.modules.models import model_registry
    from contextrouter.modules.models.types import ModelRequest, TextPart

    core_cfg = get_core_config()

    intent_cfg = core_cfg.models.rag.intent
    intent_model_key = intent_cfg.model or "vertex/gemini-2.5-flash-lite"
    fallback_keys = list(intent_cfg.fallback or [])
    strategy = intent_cfg.strategy or "fallback"

    model = model_registry.get_llm_with_fallback(
        key=intent_model_key,
        fallback_keys=fallback_keys,
        strategy=strategy,
        config=core_cfg,
    )

    # Direct model usage with new multimodal interface
    llm = model

    from contextrouter.cortex.prompting import INTENT_DETECTION_PROMPT

    taxonomy_context = _build_taxonomy_context()
    system_prompt = (
        f"{INTENT_DETECTION_PROMPT}\n\n{taxonomy_context}"
        if taxonomy_context
        else INTENT_DETECTION_PROMPT
    )

    with retrieval_span(name="detect_intent", input_data={"query": user_query[:200]}) as span_ctx:
        t0 = time.perf_counter()

        # Build prompt from system and user messages
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        prompt_parts.append(user_query)
        full_prompt = "\n\n".join(prompt_parts)

        request = ModelRequest(
            parts=[TextPart(text=full_prompt)],
            temperature=0.0,
            max_output_tokens=256,
        )

        resp = await llm.generate(request)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        pipeline_log("detect_intent.llm", duration_ms=elapsed_ms)
        span_ctx["output"] = {"elapsed_ms": elapsed_ms}

    text = resp.text
    raw = strip_json_fence(text)
    pipeline_log("detect_intent.raw", text=raw[:200])

    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError("intent_detection_invalid_json") from e
    if not isinstance(data, dict):
        raise ValueError("intent_detection_invalid_payload")

    intent = data.get("intent")
    ignore_history = data.get("ignore_history")
    cleaned = data.get("cleaned_query")
    retrieval_queries = data.get("retrieval_queries")
    user_language = data.get("user_language")
    taxonomy_concepts_llm = data.get("taxonomy_concepts")

    if isinstance(intent, str):
        intent = intent.lower().strip()
        if intent in {"rag", "search", "rag_and_web"}:
            intent = "rag_and_web"
        elif intent in {"translate", "translation"}:
            intent = "translate"
        elif intent in {"summarize", "summary", "sum"}:
            intent = "summarize"
        elif intent in {"rewrite", "edit", "fix"}:
            intent = "rewrite"
        elif intent in {"identity", "meta", "about", "self"}:
            intent = "identity"
    if intent not in {"rag_and_web", "translate", "summarize", "rewrite", "identity"}:
        logger.warning("Invalid intent from LLM: %s. Defaulting to rag_and_web.", intent)
        intent = "rag_and_web"

    if not isinstance(ignore_history, bool):
        ignore_history = bool(ignore_history_hint)
    if not isinstance(cleaned, str) or not cleaned.strip():
        cleaned = user_query

    lang_out = ""
    if isinstance(user_language, str):
        lang_out = "".join(user_language.strip().lower().split())
    if not lang_out or len(lang_out) > 8 or not re.fullmatch(r"[a-z]{2,8}", lang_out):
        lang_out = ""

    rq_out: list[str] = []
    if isinstance(retrieval_queries, list):
        for q in retrieval_queries:
            if isinstance(q, str):
                q2 = " ".join(q.split())
                if q2:
                    rq_out.append(q2[:200])
    rq_out = rq_out[:3]

    categories, concepts = _extract_taxonomy_concepts(cleaned, rq_out)
    if isinstance(taxonomy_concepts_llm, list):
        for c in taxonomy_concepts_llm:
            if isinstance(c, str) and c.strip():
                concepts.append(c.strip())
    concepts = list(dict.fromkeys(concepts))[:10]

    # Back-compat with pre-split `packages/contextrouter`:
    # strengthen retrieval queries with deterministic taxonomy concepts.
    # This ensures we don't rely purely on the LLM for domain-term retrieval and
    # helps when the cleaned query is broad but taxonomy concepts are specific.
    if intent == "rag_and_web" and concepts:
        if len(rq_out) < 3:
            concept_terms = [c for c in concepts[:3] if isinstance(c, str) and c.strip()]
            concept_query = " ".join(concept_terms).strip()
            if concept_query:
                concept_query = concept_query[:180]
                existing_lower = " ".join(rq_out).lower()
                if all(c.lower() not in existing_lower for c in concept_terms):
                    rq_out.append(concept_query)
                    rq_out = rq_out[:3]

    should = bool(cleaned) and intent == "rag_and_web"
    pipeline_log(
        "detect_intent.out",
        intent=intent,
        cleaned_query=cleaned[:200],
        retrieval_queries=rq_out,
        taxonomy_concepts=concepts[:10],
        should_retrieve=should,
    )
    return {
        "intent": intent,
        "intent_text": cleaned,
        "user_language": lang_out,
        "ignore_history": ignore_history,
        "retrieval_queries": rq_out,
        "should_retrieve": should,
        "taxonomy_categories": categories,
        "taxonomy_concepts": concepts,
    }


__all__ = ["detect_intent"]
