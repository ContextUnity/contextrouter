"""
Harvest subgraph steps - fetch news using Perplexity with LLM fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from contextrouter.core import get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from ..state import NewsEngineState
from .json_parser import extract_json_array
from .prompts import DEFAULT_HARVESTER_PROMPT

logger = logging.getLogger(__name__)


async def harvest_perplexity_node(state: NewsEngineState) -> dict[str, Any]:
    """Use Perplexity to harvest news with built-in search."""
    tenant_id = state.get("tenant_id", "unknown")
    logger.info("[%s] Harvesting via Perplexity", tenant_id)

    config = get_core_config()

    if not config.perplexity.api_key:
        logger.warning("Perplexity API key not configured, skipping")
        return {"harvest_errors": ["Perplexity API key not configured"]}

    # Get prompt override or use default
    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("harvester", DEFAULT_HARVESTER_PROMPT)

    # Calculate date thresholds for filtering
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    day_before = (now - timedelta(days=2)).strftime("%Y-%m-%d")

    max_retries = 3
    min_items = 10
    all_items = []

    for attempt in range(max_retries):
        try:
            model = model_registry.create_llm(
                "perplexity/sonar",
                config=config,
                search_recency_filter="day",
                return_citations=True,
            )

            # Adjust prompt on retry to encourage more results
            if attempt > 0:
                user_prompt = f"Find today's positive news. Need at least {min_items} stories. Attempt {attempt + 1}. TODAY IS {today}."
            else:
                user_prompt = f"Find today's positive news matching the criteria. TODAY IS {today}."

            request = ModelRequest(
                system=system_prompt,
                parts=[TextPart(text=user_prompt)],
                temperature=0.3 + (attempt * 0.1),  # Slightly higher temp on retry
                max_output_tokens=8000,  # Extra for reasoning models
            )

            logger.info("[%s] Sending Perplexity request (attempt %s)...", tenant_id, attempt + 1)
            response = await model.generate(request)
            logger.info("[%s] Perplexity response received", tenant_id)

            # Log preview
            text = response.text
            if text:
                logger.info("Perplexity response preview: %s...", text[:500])

            # Parse response using shared parser
            items = extract_json_array(text)

            # Filter items by date (keep only last 48 hours)
            fresh_items = []
            stale_count = 0
            for item in items:
                pub_date = item.get("publication_date", "")
                url = item.get("url", "")
                headline = item.get("headline", "")[:50]

                # Check if publication_date is fresh (today, yesterday, or day before)
                is_fresh_by_date = pub_date in (today, yesterday, day_before) if pub_date else False

                # Also check URL for date patterns (e.g., /2026/02/02/)
                is_fresh_by_url = any(
                    date_str in url or date_str.replace("-", "/") in url
                    for date_str in (today, yesterday, day_before)
                )

                if is_fresh_by_date or is_fresh_by_url:
                    fresh_items.append(item)
                else:
                    stale_count += 1
                    logger.info(
                        "Filtered stale news: '%s...' (date: %s)", headline, pub_date or "unknown"
                    )

            if stale_count > 0:
                logger.warning(
                    "[%s] Filtered out %s stale items from Perplexity response",
                    tenant_id,
                    stale_count,
                )

            # Mark source and add to collection (dedupe by headline)
            for item in fresh_items:
                item["source"] = "perplexity"
                # Dedupe by headline
                if not any(
                    existing.get("headline") == item.get("headline") for existing in all_items
                ):
                    all_items.append(item)

            logger.info(
                "[%s] Perplexity attempt %s: got %s items, %s fresh, total unique: %s",
                tenant_id,
                attempt + 1,
                len(items),
                len(fresh_items),
                len(all_items),
            )

            # Check if we have enough
            if len(all_items) >= min_items:
                break

            if attempt < max_retries - 1:
                logger.info(
                    "[%s] Retrying Perplexity - need %s more items",
                    tenant_id,
                    min_items - len(all_items),
                )
                import asyncio

                await asyncio.sleep(1)  # Brief pause between retries

        except Exception as e:
            logger.error("Perplexity harvest attempt %s failed: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                return {
                    "raw_items": all_items,
                    "harvest_source": "perplexity",
                    "harvest_errors": [f"Perplexity error: {str(e)}"],
                }

    logger.info("[%s] Perplexity returned %s total fresh items", tenant_id, len(all_items))

    return {
        "raw_items": all_items,
        "harvest_source": "perplexity",
    }


async def harvest_llm_fallback_node(state: NewsEngineState) -> dict[str, Any]:
    """Fallback to OpenAI with web search if Perplexity failed or returned no results.

    Uses model_registry.create_llm with enable_web_search=True to activate
    OpenAI's Responses API with web_search_preview tool.
    """
    existing_items = state.get("raw_items", [])
    errors = state.get("harvest_errors", [])

    # Skip if we already have items
    if existing_items:
        return {}

    tenant_id = state.get("tenant_id", "unknown")
    config = get_core_config()

    logger.info("[%s] Falling back to OpenAI with web search", tenant_id)

    # Get prompt override or use default
    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("harvester", DEFAULT_HARVESTER_PROMPT)

    try:
        # Use model_registry with enable_web_search=True (architecture-compliant)
        model = model_registry.create_llm(
            "openai/gpt-4o-mini",
            config=config,
            enable_web_search=True,
        )

        # Include today's date for accurate freshness filtering
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        user_prompt = f"""TODAY'S DATE: {today}

TASK: Find REAL, FRESH positive news about Ukraine published in the last 24 hours.

STRICT REQUIREMENTS:
1. Only include news published TODAY ({today}) or yesterday
2. Only use NEWS SOURCES (pravda.com.ua, kyivindependent.com, ukrinform.net, etc.)
3. DO NOT use Wikipedia, general pages, or old articles
4. Each URL must point to an actual news article with a visible publication date
5. Verify news freshness before including - reject anything older than 24 hours

Return 5-10 stories as JSON array with:
- headline: exact headline from the source
- summary: brief 1-2 sentence summary
- url: direct link to the news article
- publication_date: the article's publication date (YYYY-MM-DD format)
- source: the news outlet name
- category: one of [military, diplomacy, economy, culture, technology, humanitarian]
- significance_score: 1-10"""

        request = ModelRequest(
            system=system_prompt,
            parts=[TextPart(text=user_prompt)],
            temperature=0.3,  # Lower temperature for more factual results
            max_output_tokens=8000,
        )

        response = await model.generate(request)

        # Parse JSON from response
        items = extract_json_array(response.text)

        # Filter out old news (keep only items from last 24 hours)
        from datetime import timedelta

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        fresh_items = []
        for item in items:
            pub_date = item.get("publication_date", "")
            url = item.get("url", "")

            # Check if publication_date is fresh
            is_fresh_by_date = pub_date >= yesterday if pub_date else False

            # Also check URL for date patterns (e.g., /2026/01/30/)
            is_fresh_by_url = today in url or yesterday in url.replace("-", "/")

            if is_fresh_by_date or is_fresh_by_url:
                item["source"] = "openai_web_search"
                fresh_items.append(item)
            else:
                logger.debug(
                    "Filtered old news: %s (date: %s)", item.get("headline", "unknown"), pub_date
                )

        logger.info(
            "[%s] OpenAI web search: %s total, %s fresh items",
            tenant_id,
            len(items),
            len(fresh_items),
        )

        if not fresh_items:
            logger.warning("[%s] No fresh news found via web search", tenant_id)

        return {
            "raw_items": fresh_items,
            "harvest_source": "openai_web_search",
        }

    except Exception as e:
        logger.error("OpenAI web search fallback failed: %s", e)
        return {
            "harvest_errors": errors + [f"OpenAI web search error: {str(e)}"],
        }


async def store_raw_to_brain_node(state: NewsEngineState) -> dict[str, Any]:
    """Store raw harvested items to Brain for audit trail."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")
    harvest_source = state.get("harvest_source", "unknown")

    if not raw_items:
        return {}

    logger.info("[%s] Storing %s raw items to Brain", tenant_id, len(raw_items))

    config = get_core_config()

    try:
        from contextcore import BrainClient

        from contextrouter.core.brain_token import get_brain_service_token

        client = BrainClient(host=config.brain.grpc_endpoint, token=get_brain_service_token())

        stored_count = 0
        for item in raw_items:
            try:
                item_id = await client.upsert_news_item(
                    tenant_id=tenant_id,
                    url=item.get("url", ""),
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    item_type="raw",
                    category=item.get("category", ""),
                    source_api=harvest_source,
                    metadata={
                        "significance_score": str(item.get("significance_score", 5)),
                        "source": item.get("source", ""),
                    },
                )

                if item_id:
                    stored_count += 1
                    item["brain_id"] = item_id

            except Exception as e:
                logger.warning("Failed to store raw item to Brain: %s", e)

        logger.info("[%s] Stored %s/%s raw items to Brain", tenant_id, stored_count, len(raw_items))

    except ImportError:
        logger.warning("contextcore not available, skipping Brain storage")
    except Exception as e:
        logger.error("Brain storage error: %s", e)

    return {}


def _should_fallback(state: NewsEngineState) -> str:
    """Decide if we need LLM fallback."""
    items = state.get("raw_items", [])

    if items:
        return "store_to_brain"
    # Fallback if no items OR if there were errors
    return "llm_fallback"


def _after_llm_fallback(state: NewsEngineState) -> str:
    """After LLM fallback, store to Brain if we have items."""
    items = state.get("raw_items", [])
    if items:
        return "store_to_brain"
    return END


def create_harvest_subgraph():
    """Create harvest subgraph with Perplexity + LLM fallback + Brain storage.

    Flow:
        perplexity → (if no items) → llm_fallback → store_to_brain → END
                   → (if items)   → store_to_brain → END
    """
    graph = StateGraph(NewsEngineState)

    graph.add_node("perplexity", harvest_perplexity_node)
    graph.add_node("llm_fallback", harvest_llm_fallback_node)
    graph.add_node("store_to_brain", store_raw_to_brain_node)

    graph.add_edge(START, "perplexity")
    graph.add_conditional_edges(
        "perplexity",
        _should_fallback,
        {
            "llm_fallback": "llm_fallback",
            "store_to_brain": "store_to_brain",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "llm_fallback",
        _after_llm_fallback,
        {
            "store_to_brain": "store_to_brain",
            END: END,
        },
    )
    graph.add_edge("store_to_brain", END)

    return graph.compile()
