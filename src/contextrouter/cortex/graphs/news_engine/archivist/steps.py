"""
Archivist subgraph steps.

Pipeline:
1. filter_node - Apply positive filter, reject banned content
2. validate_node - LLM validation for greenwashing/quality
3. dedupe_node - Vector search for duplicates via Brain
4. store_node - Store valid facts in Brain
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from contextrouter.core import get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from ..state import NewsEngineState

logger = logging.getLogger(__name__)

# Banned keywords for content filtering
BANNED_KEYWORDS = {
    "war",
    "війна",
    "russia",
    "росія",
    "belarus",
    "білорусь",
    "putin",
    "путін",
    "zelensky",
    "зеленський",
    "crime",
    "злочин",
    "murder",
    "вбивство",
    "death",
    "смерть",
    "scandal",
    "скандал",
    "tragedy",
    "трагедія",
    "accident",
    "аварія",
    "corruption",
    "корупція",
    "arrest",
    "арешт",
}

# Default archivist prompt
DEFAULT_ARCHIVIST_PROMPT = """You are an editor for a positive news channel.

Your job is to validate news items for quality and authenticity.

REJECT items that are:
- Greenwashing (corporate PR disguised as real progress)
- Vague announcements without concrete results
- Speculation or "plans to" without actual achievement
- Clickbait with misleading headlines
- Anything promoting harmful products/practices

ACCEPT items that are:
- Concrete achievements with measurable impact
- Community-driven initiatives with real outcomes
- Scientific/technological breakthroughs with verification
- Policy changes that have already taken effect

For each item, respond with JSON:
{
  "verdict": "accept" | "reject",
  "reason": "brief explanation",
  "category": "environment|technology|community|urban|nature|energy|innovation",
  "significance_score": 1-10,
  "suggested_agents": ["agent1", "agent2"]
}"""


async def filter_node(state: NewsEngineState) -> Dict[str, Any]:
    """Apply positive filter to raw items using keyword matching."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info(f"[{tenant_id}] Filtering {len(raw_items)} items")

    filtered = []
    rejected = 0

    for item in raw_items:
        text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()

        # Check for banned keywords
        if any(keyword in text for keyword in BANNED_KEYWORDS):
            rejected += 1
            continue

        filtered.append(item)

    logger.info(f"[{tenant_id}] Keyword filter: {len(filtered)} passed, {rejected} rejected")

    return {
        "raw_items": filtered,
        "rejected_count": rejected,
    }


async def validate_node(state: NewsEngineState) -> Dict[str, Any]:
    """LLM validation for quality and greenwashing detection."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    if not raw_items:
        return {}

    config = get_core_config()

    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("archivist", DEFAULT_ARCHIVIST_PROMPT)

    logger.info(f"[{tenant_id}] LLM validating {len(raw_items)} items")

    try:
        model = model_registry.get_llm_with_fallback(
            key=config.models.default_llm,
            fallback_keys=[],
            strategy="fallback",
            config=config,
        )

        validated = []
        rejected = state.get("rejected_count", 0)

        for item in raw_items:
            user_prompt = f"""Validate this news item:

Headline: {item.get("headline", "")}
Summary: {item.get("summary", "")}
Source: {item.get("url", "")}"""

            request = ModelRequest(
                system=system_prompt,
                parts=[TextPart(text=user_prompt)],
                temperature=0.2,
                max_output_tokens=500,
            )

            try:
                response = await model.generate(request)

                # Parse response
                text = response.text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(text[start:end])

                    if result.get("verdict") == "accept":
                        item["category"] = result.get("category", item.get("category", "unknown"))
                        item["significance_score"] = result.get("significance_score", 5)
                        item["suggested_agents"] = result.get("suggested_agents", [])
                        item["validation_reason"] = result.get("reason", "")
                        validated.append(item)
                    else:
                        rejected += 1
                        logger.debug(f"Rejected: {item.get('headline')} - {result.get('reason')}")
                else:
                    # Couldn't parse, accept by default
                    validated.append(item)
            except Exception as e:
                logger.warning(f"Validation failed for item: {e}")
                validated.append(item)  # Accept on error

        logger.info(
            f"[{tenant_id}] LLM validation: {len(validated)} accepted, {rejected} total rejected"
        )

        return {
            "raw_items": validated,
            "rejected_count": rejected,
        }

    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        return {}


async def dedupe_node(state: NewsEngineState) -> Dict[str, Any]:
    """Check for duplicates via Brain vector search using semantic similarity."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    if not raw_items:
        return {"duplicate_count": 0}

    logger.info(f"[{tenant_id}] Deduplicating {len(raw_items)} items via semantic search")

    config = get_core_config()

    # Similarity threshold - items above this are considered duplicates
    # 0.85 = very similar, 0.90 = nearly identical
    SIMILARITY_THRESHOLD = 0.85

    try:
        from contextcore import BrainClient

        client = BrainClient(host=config.brain.grpc_endpoint)

        unique_items = []
        duplicates = 0

        for item in raw_items:
            headline = item.get("headline", "")
            summary = item.get("summary", "")

            # Use headline + summary for better matching
            search_text = f"{headline} {summary}"[:500]

            # Search for similar items in Brain
            similar = await client.search(
                tenant_id=tenant_id,
                query_text=search_text,
                source_types=["news_fact"],
                limit=3,  # Check top 3 matches
            )

            # Check if any result is too similar using similarity score
            is_duplicate = False
            for s in similar:
                # SearchResult now has score field directly
                if s.score >= SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    logger.debug(
                        f"Duplicate (score={s.score:.2f}): '{headline[:50]}...' "
                        f"matches '{s.content[:50]}...'"
                    )
                    break

            if is_duplicate:
                duplicates += 1
            else:
                unique_items.append(item)

        logger.info(
            f"[{tenant_id}] Dedupe: {len(unique_items)} unique, {duplicates} duplicates "
            f"(threshold={SIMILARITY_THRESHOLD})"
        )

        return {
            "raw_items": unique_items,
            "duplicate_count": duplicates,
        }

    except ImportError:
        logger.warning("contextcore not available, skipping dedupe")
        return {"duplicate_count": 0}
    except Exception as e:
        logger.warning(f"Dedupe failed: {e}")
        return {"duplicate_count": 0}


async def store_node(state: NewsEngineState) -> Dict[str, Any]:
    """Convert raw items to facts and store in Brain via gRPC."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info(f"[{tenant_id}] Storing {len(raw_items)} facts to Brain")

    facts = []
    config = get_core_config()
    stored_count = 0

    try:
        # Use gRPC client
        import uuid

        import grpc
        from contextcore import brain_pb2, brain_pb2_grpc

        channel = grpc.aio.insecure_channel(config.brain.grpc_endpoint)
        stub = brain_pb2_grpc.BrainServiceStub(channel)

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
                news_item = brain_pb2.NewsItem(
                    id=fact["id"],
                    tenant_id=tenant_id,
                    url=fact["url"],
                    headline=fact["headline"],
                    summary=fact["summary"],
                    category=fact["category"],
                    metadata={
                        "significance_score": str(fact["significance_score"]),
                        "source": fact["source"],
                        "suggested_agents": ",".join(fact["suggested_agents"])
                        if fact["suggested_agents"]
                        else "",
                    },
                )

                request = brain_pb2.UpsertNewsItemRequest(
                    tenant_id=tenant_id,
                    item=news_item,
                    item_type="fact",
                )

                response = await stub.UpsertNewsItem(request)
                if response.success:
                    stored_count += 1
                    fact["brain_id"] = response.id

            except Exception as e:
                logger.warning(f"Failed to store fact to Brain: {e}")

            facts.append(fact)

        await channel.close()
        logger.info(f"[{tenant_id}] Stored {stored_count}/{len(facts)} facts to Brain")

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
        logger.error(f"Brain storage error: {e}")

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
    """Build the archivist subgraph."""
    workflow = StateGraph(NewsEngineState)

    workflow.add_node("filter", filter_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("dedupe", dedupe_node)
    workflow.add_node("store", store_node)

    workflow.set_entry_point("filter")
    workflow.add_edge("filter", "validate")
    workflow.add_edge("validate", "dedupe")
    workflow.add_edge("dedupe", "store")
    workflow.add_edge("store", END)

    return workflow.compile()
