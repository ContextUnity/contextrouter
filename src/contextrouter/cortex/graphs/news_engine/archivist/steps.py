"""
Archivist subgraph steps.

Pipeline:
1. filter_node - Apply positive filter, reject banned content
2. dedupe_node - Vector search for duplicates via Brain
3. store_node - Store valid facts in Brain
"""

from __future__ import annotations

from typing import Any, Dict

from contextcore import get_context_unit_logger
from langgraph.graph import END, StateGraph

from contextrouter.core import get_core_config

from ..state import NewsEngineState
from .filters import BANNED_KEYWORDS, SIMILARITY_THRESHOLD

logger = get_context_unit_logger(__name__)


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
    """Convert raw items to facts and store in Brain via generic Vector UPSERT."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info(
        "[%s] Storing %s facts to Brain (Generic Vector Storage)", tenant_id, len(raw_items)
    )

    import uuid

    facts = []
    stored_count = 0
    config = get_core_config()

    try:
        from contextcore import BrainClient

        from contextrouter.core.brain_token import get_brain_service_token

        client = BrainClient(host=config.brain.grpc_endpoint, token=get_brain_service_token())

        for item in raw_items:
            fact_id = str(uuid.uuid4())
            fact = {
                "id": fact_id,
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "category": item.get("category", "unknown"),
                "significance_score": item.get("significance_score", 5),
                "suggested_agents": item.get("suggested_agents", []),
                "source": item.get("source", "unknown"),
            }

            try:
                # Upsert into generic knowledge graph (ContextBrain)
                content_text = f"{fact['headline']}\\n\\n{fact['summary']}"
                await client.upsert(
                    tenant_id=tenant_id,
                    content=content_text,
                    source_type="news_fact",
                    doc_id=fact_id,
                    metadata={
                        "url": fact["url"],
                        "category": fact["category"],
                        "significance_score": str(fact["significance_score"]),
                        "source": fact["source"],
                    },
                )
                stored_count += 1
                fact["brain_id"] = fact_id
            except Exception as e:
                logger.warning("Failed to upsert fact generically: %s", e)

            facts.append(fact)

    except ImportError:
        logger.warning("contextcore not available, facts won't be stored")
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
        logger.error("Brain generic storage error: %s", e)

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
    from contextrouter.cortex.graphs.secure_node import make_secure_node

    workflow = StateGraph(NewsEngineState)

    secure_filter = make_secure_node("filter", filter_node)
    secure_dedupe = make_secure_node("dedupe", dedupe_node)
    secure_store = make_secure_node("store", store_node)

    workflow.add_node("filter", secure_filter)
    # validate_node removed - Showrunner handles editorial selection
    workflow.add_node("dedupe", secure_dedupe)
    workflow.add_node("store", secure_store)

    workflow.set_entry_point("filter")
    workflow.add_edge("filter", "dedupe")  # Skip validate, go directly to dedupe
    workflow.add_edge("dedupe", "store")
    workflow.add_edge("store", END)

    return workflow.compile()
