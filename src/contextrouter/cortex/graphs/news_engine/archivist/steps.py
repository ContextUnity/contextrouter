"""
Archivist subgraph steps.

Pipeline:
1. filter_node - Apply positive filter, reject banned content
2. dedupe_node - Vector search for duplicates via Brain
3. store_node - Store valid facts in Brain
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from contextrouter.core import get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from ..state import NewsEngineState
from .filters import BANNED_KEYWORDS, DEFAULT_ARCHIVIST_PROMPT, SIMILARITY_THRESHOLD
from .json_utils import extract_json_from_response

logger = logging.getLogger(__name__)


async def filter_node(state: NewsEngineState) -> Dict[str, Any]:
    """Apply positive filter to raw items using keyword matching."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info("[%s] Filtering %s items", tenant_id, len(raw_items))

    filtered = []
    rejected = 0

    for item in raw_items:
        text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()

        # Check for banned keywords
        if any(keyword in text for keyword in BANNED_KEYWORDS):
            rejected += 1
            continue

        filtered.append(item)

    logger.info("[%s] Keyword filter: %s passed, %s rejected", tenant_id, len(filtered), rejected)

    return {
        "raw_items": filtered,
        "rejected_count": rejected,
    }


async def validate_node(state: NewsEngineState) -> Dict[str, Any]:
    """LLM validation for quality and greenwashing detection.

    Note: This node is currently not used in the pipeline (Showrunner handles selection).
    Kept for potential future use.
    """
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    if not raw_items:
        return {}

    config = get_core_config()

    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("archivist", DEFAULT_ARCHIVIST_PROMPT)

    logger.info("[%s] LLM validating %s items", tenant_id, len(raw_items))

    try:
        model = model_registry.get_llm_with_fallback(
            key=config.models.default_llm,
            fallback_keys=config.models.fallback_llms,
            strategy="fallback",
            config=config,
        )

        validated = []
        rejected = state.get("rejected_count", 0)

        for item in raw_items:
            headline = item.get("headline", "")[:60]
            user_prompt = f"""Validate this news item:

Headline: {item.get("headline", "")}
Summary: {item.get("summary", "")}
Source: {item.get("url", "")}"""

            request = ModelRequest(
                system=system_prompt,
                parts=[TextPart(text=user_prompt)],
                temperature=0.2,
                max_output_tokens=4000,  # Extra for reasoning models
                response_format="json_object",  # Enforce JSON output
            )

            try:
                response = await model.generate(request)

                # Parse response using robust extractor
                result = extract_json_from_response(response.text)

                if result:
                    if result.get("verdict") == "accept":
                        item["category"] = result.get("category", item.get("category", "unknown"))
                        item["significance_score"] = result.get("significance_score", 5)
                        item["suggested_agents"] = result.get("suggested_agents", [])
                        item["validation_reason"] = result.get("reason", "")
                        validated.append(item)
                    else:
                        rejected += 1
                        logger.debug(
                            "Rejected: %s - %s", item.get("headline"), result.get("reason")
                        )
                else:
                    # Log first 200 chars of response for debugging
                    preview = response.text[:200].replace("\n", " ") if response.text else "(empty)"
                    logger.warning(
                        "[No JSON Found] '%s...' - Could not extract JSON from response. Preview: %s... Accepting item by default.",
                        headline,
                        preview,
                    )
                    validated.append(item)

            except asyncio.CancelledError:
                logger.warning(
                    "[Connection Timeout] '%s...' - OpenAI request was cancelled ", headline
                )
                validated.append(item)

            except Exception as e:
                error_type = type(e).__name__
                logger.warning("[%s] '%s...' - Validation failed: %s. ", error_type, headline, e)
                validated.append(item)

        logger.info(
            "[%s] LLM validation: %s accepted, %s total rejected",
            tenant_id,
            len(validated),
            rejected,
        )

        return {
            "raw_items": validated,
            "rejected_count": rejected,
        }

    except Exception as e:
        logger.error("LLM validation failed: %s", e)
        return {}


async def dedupe_node(state: NewsEngineState) -> Dict[str, Any]:
    """Check for duplicates via semantic similarity (facts-based).

    Note: URL-based deduplication was removed because:
    - Perplexity often returns generic homepage URLs
    - One URL can contain multiple different articles
    - Semantic similarity is more reliable for detecting duplicate content
    """
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    if not raw_items:
        return {"duplicate_count": 0}

    logger.info("[%s] Deduplicating %s items (semantic only)", tenant_id, len(raw_items))

    config = get_core_config()

    unique_items = []
    duplicates = 0
    seen_headlines: set[str] = set()

    for item in raw_items:
        headline = item.get("headline", "")
        summary = item.get("summary", "")

        # 1. Local headline deduplication (within current batch)
        # This catches exact duplicates in the same harvest
        headline_key = headline.strip().lower()
        if headline_key in seen_headlines:
            duplicates += 1
            logger.debug("Exact headline duplicate in batch: '%s...'", headline[:50])
            continue
        seen_headlines.add(headline_key)

        # 2. Semantic similarity search via Brain
        is_duplicate = False
        try:
            from contextcore import BrainClient

            from contextrouter.core.brain_token import get_brain_service_token

            client = BrainClient(host=config.brain.grpc_endpoint, token=get_brain_service_token())

            search_text = f"{headline} {summary}"[:500]
            similar = await client.search(
                tenant_id=tenant_id,
                query_text=search_text,
                source_types=["news_fact", "news_post"],
                limit=5,
            )

            for s in similar:
                if s.score >= SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    score_str = format(s.score, ".2f")
                    logger.info(
                        "Semantic duplicate (score=%s): '%s...' matches existing: '%s...'",
                        score_str,
                        headline[:50],
                        s.content[:50],
                    )
                    duplicates += 1
                    break
        except Exception as e:
            logger.debug("Semantic search unavailable: %s", e)

        if not is_duplicate:
            unique_items.append(item)

    logger.info(
        "[%s] Dedupe: %s unique, %s duplicates (threshold=%s)",
        tenant_id,
        len(unique_items),
        duplicates,
        SIMILARITY_THRESHOLD,
    )

    return {
        "raw_items": unique_items,
        "duplicate_count": duplicates,
    }


async def store_node(state: NewsEngineState) -> Dict[str, Any]:
    """Convert raw items to facts and store in Brain via gRPC."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info("[%s] Storing %s facts to Brain", tenant_id, len(raw_items))

    facts = []
    config = get_core_config()
    stored_count = 0

    try:
        import uuid

        from contextcore import BrainClient

        from contextrouter.core.brain_token import get_brain_service_token

        client = BrainClient(host=config.brain.grpc_endpoint, token=get_brain_service_token())

        for item in raw_items:
            fact = {
                "id": str(uuid.uuid4()),
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "category": item.get("category", "unknown"),
                "significance_score": item.get("significance_score", 5),
                "suggested_agents": item.get("suggested_agents", []),
                "source": item.get("source", "unknown"),
                "raw_id": item.get("brain_id"),  # Link to raw item if available
            }

            try:
                fact_id = await client.upsert_news_item(
                    tenant_id=tenant_id,
                    url=fact["url"],
                    headline=fact["headline"],
                    summary=fact["summary"],
                    item_type="fact",
                    category=fact["category"],
                    metadata={
                        "significance_score": str(fact["significance_score"]),
                        "source": fact["source"],
                        "suggested_agents": ",".join(fact["suggested_agents"])
                        if fact["suggested_agents"]
                        else "",
                    },
                )

                if fact_id:
                    stored_count += 1
                    fact["brain_id"] = fact_id

            except Exception as e:
                logger.warning("Failed to store fact to Brain: %s", e)

            facts.append(fact)

        logger.info("[%s] Stored %s/%s facts to Brain", tenant_id, stored_count, len(facts))

    except ImportError:
        logger.warning("contextcore not available, facts won't be stored to Brain")
        # Still create facts for the pipeline
        import uuid

        for item in raw_items:
            fact = {
                "id": str(uuid.uuid4()),
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "category": item.get("category", "unknown"),
                "significance_score": item.get("significance_score", 5),
                "suggested_agents": item.get("suggested_agents", []),
                "source": item.get("source", "unknown"),
            }
            facts.append(fact)
    except Exception as e:
        logger.error("Brain storage error: %s", e)

    return {
        "facts": facts,
        "result": {
            "status": "archived",
            "facts_count": len(facts),
            "stored_count": stored_count,
            "rejected_count": state.get("rejected_count", 0),
            "duplicate_count": state.get("duplicate_count", 0),
        },
    }


def create_archivist_subgraph():
    """Build the archivist subgraph.

    Pipeline: filter -> dedupe -> store

    Note: LLM validation was removed because Showrunner already does
    editorial selection. This speeds up the pipeline by ~4 minutes.
    """
    workflow = StateGraph(NewsEngineState)

    workflow.add_node("filter", filter_node)
    # validate_node removed - Showrunner handles editorial selection
    workflow.add_node("dedupe", dedupe_node)
    workflow.add_node("store", store_node)

    workflow.set_entry_point("filter")
    workflow.add_edge("filter", "dedupe")  # Skip validate, go directly to dedupe
    workflow.add_edge("dedupe", "store")
    workflow.add_edge("store", END)

    return workflow.compile()
