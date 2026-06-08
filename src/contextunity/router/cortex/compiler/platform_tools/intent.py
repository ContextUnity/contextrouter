"""Intent detection step (pure function).

This is a copy of the intent logic without agent registration side effects.
"""

from __future__ import annotations

import re
import time
from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.types import is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.compiler.types import CompilerDataSourceSpec
from contextunity.router.cortex.core_state import get_last_user_query
from contextunity.router.cortex.services import get_graph_service
from contextunity.router.cortex.types import GraphState, StateUpdate
from contextunity.router.cortex.utils.json import strip_json_fence
from contextunity.router.cortex.utils.pipeline import pipeline_log
from contextunity.router.cortex.utils.taxonomy_loader import (
    get_taxonomy_canonical_map,
    get_taxonomy_top_level_categories,
)
from contextunity.router.modules.observability import retrieval_span


class DetectIntentConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


logger = get_contextunit_logger(__name__)


def _data_sources_from_state(state: GraphState) -> list[CompilerDataSourceSpec]:
    """Extract configured data sources from graph state."""
    raw_state_config: object = state.get("config")
    if not is_object_dict(raw_state_config):
        return []
    raw_sources: object = raw_state_config.get("data_sources", [])
    if not is_object_list(raw_sources):
        return []

    sources: list[CompilerDataSourceSpec] = []
    for raw_source in raw_sources:
        if not is_object_dict(raw_source):
            continue
        source: CompilerDataSourceSpec = {}
        source_type = raw_source.get("type")
        if source_type == "vector":
            source["type"] = "vector"
        elif source_type == "sql":
            source["type"] = "sql"
        elif source_type == "federated":
            source["type"] = "federated"
        elif source_type == "web":
            source["type"] = "web"
        binding = raw_source.get("binding")
        if isinstance(binding, str):
            source["binding"] = binding
        description = raw_source.get("description")
        if isinstance(description, str):
            source["description"] = description
        if source:
            sources.append(source)
    return sources


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
    except Exception as e:  # graceful-degrade: tool failure returns empty result
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
    except Exception as e:  # graceful-degrade: tool failure returns empty result
        logger.warning("Failed to extract taxonomy concepts: %s. Taxonomy extraction disabled.", e)

    return list(categories)[:5], list(concepts)[:10]


async def detect_intent(state: GraphState) -> StateUpdate:
    """Detect intent and derive retrieval queries for the current user message."""
    user_query = (
        get_last_user_query(state.get("messages") or [])
        or ((state.get("user_query") or "").strip()[-500:])
    )
    pipeline_log("detect_intent.in", user_query=user_query)

    if not user_query:
        return {
            "intent": "rag",
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

    from contextunity.router.cortex.config_resolution import get_node_manifest_config
    from contextunity.router.modules.models import model_registry
    from contextunity.router.modules.models.types import ModelRequest, TextPart

    node_config = get_node_manifest_config(state, "detect_intent")
    intent_model_key = node_config.get("model") or "vertex/gemini-2.5-flash-lite"
    model = model_registry.create_llm(intent_model_key)

    # Direct model usage with new multimodal interface
    llm = model

    from contextunity.router.cortex.compiler.platform_tools.prompts import INTENT_DETECTION_PROMPT

    taxonomy_context = _build_taxonomy_context()

    data_sources = _data_sources_from_state(state)
    ds_text = ""
    if data_sources:
        ds_lines = ["\nDATA SOURCES:"]
        for ds in data_sources:
            binding = ds.get("binding", "unknown")
            ds_type = ds.get("type", "unknown")
            desc = ds.get("description", "")
            ds_lines.append(f"- {binding} ({ds_type}): {desc}")
        ds_text = "\n".join(ds_lines)

    system_prompt = INTENT_DETECTION_PROMPT
    if taxonomy_context:
        system_prompt += f"\n\n{taxonomy_context}"
    if ds_text:
        system_prompt += f"\n{ds_text}"

    with retrieval_span(name="detect_intent", input_data={"query": user_query[:200]}) as span_ctx:
        t0 = time.perf_counter()

        # Build prompt from system and user messages
        prompt_parts: list[str] = []
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
        payload = json_loads(raw)
    except Exception as e:  # graceful-degrade: tool failure returns empty result
        from contextunity.router.core.exceptions import RouterIntentDetectionError

        raise RouterIntentDetectionError("Invalid JSON in intent detection response.") from e
    if not is_object_dict(payload):
        from contextunity.router.core.exceptions import RouterIntentDetectionError

        raise RouterIntentDetectionError("Invalid intent detection payload structure.")
    data = payload

    intent = data.get("intent")
    ignore_history = data.get("ignore_history")
    cleaned = data.get("cleaned_query")
    retrieval_queries = data.get("retrieval_queries")
    user_language = data.get("user_language")
    taxonomy_concepts_llm = data.get("taxonomy_concepts")
    selected_sources = data.get("selected_sources")

    selected_sources_out: list[str] = []
    if is_object_list(selected_sources):
        for source in selected_sources:
            if isinstance(source, str) and source.strip():
                selected_sources_out.append(source)
    selected_sources = selected_sources_out

    if isinstance(intent, str):
        intent = intent.lower().strip()
        if intent in {"rag", "search", "rag_and_web", "web_search", "web"}:
            intent = "rag"  # Historically rag_and_web, now defaults to rag and relies on selected_sources
        elif intent in {"sql", "analytics", "sql_analytics", "database"}:
            intent = "sql"
        elif intent in {"translate", "translation"}:
            intent = "translate"
        elif intent in {"summarize", "summary", "sum"}:
            intent = "summarize"
        elif intent in {"rewrite", "edit", "fix"}:
            intent = "rewrite"
        elif intent in {"identity", "meta", "about", "self"}:
            intent = "identity"
    if intent not in {"rag", "translate", "summarize", "rewrite", "identity", "sql"}:
        logger.warning("Invalid intent from LLM: %s. Defaulting to rag.", intent)
        intent = "rag"

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
    if is_object_list(retrieval_queries):
        for q in retrieval_queries:
            if isinstance(q, str):
                q2 = " ".join(q.split())
                if q2:
                    rq_out.append(q2[:200])
    rq_out = rq_out[:3]

    categories, concepts = _extract_taxonomy_concepts(cleaned, rq_out)
    if is_object_list(taxonomy_concepts_llm):
        for c in taxonomy_concepts_llm:
            if isinstance(c, str) and c.strip():
                concepts.append(c.strip())
    concepts = list(dict.fromkeys(concepts))[:10]

    # Back-compat with pre-split `packages/contextunity.router`:
    # strengthen retrieval queries with deterministic taxonomy concepts.
    # This ensures we don't rely purely on the LLM for domain-term retrieval and
    # helps when the cleaned query is broad but taxonomy concepts are specific.
    if intent == "rag" and concepts:
        if len(rq_out) < 3:
            concept_terms = [c for c in concepts[:3] if c.strip()]
            concept_query = " ".join(concept_terms).strip()
            if concept_query:
                concept_query = concept_query[:180]
                existing_lower = " ".join(rq_out).lower()
                if all(c.lower() not in existing_lower for c in concept_terms):
                    rq_out.append(concept_query)
                    rq_out = rq_out[:3]

    should = bool(cleaned) and (intent == "rag")

    # Compute intent_route for LangGraph branching
    data_sources = _data_sources_from_state(state)
    has_sql = any(ds.get("type") == "sql" for ds in data_sources)
    has_vector = any(ds.get("type") == "vector" for ds in data_sources)

    if selected_sources:
        intent_route = "selected_sources_fanout"
    elif intent == "sql":
        intent_route = "sql_analytics" if has_sql else "no_results"
    elif intent == "rag":
        # If legacy state or no data_sources given, assume vector RAG by default
        intent_route = "retrieve" if (has_vector or not data_sources) else "no_results"
    else:
        intent_route = "skip_retrieve"

    pipeline_log(
        "detect_intent.out",
        intent=intent,
        intent_route=intent_route,
        cleaned_query=cleaned[:200],
        retrieval_queries=rq_out,
        taxonomy_concepts=concepts[:10],
        should_retrieve=should,
    )
    return {
        "dynamic": {
            "intent": intent,
            "intent_route": intent_route,
            "intent_text": cleaned,
            "user_language": lang_out,
            "ignore_history": ignore_history,
            "retrieval_queries": rq_out,
            "should_retrieve": should,
            "taxonomy_categories": categories,
            "taxonomy_concepts": concepts,
            "selected_sources": selected_sources,
        }
    }


__all__ = ["detect_intent"]
