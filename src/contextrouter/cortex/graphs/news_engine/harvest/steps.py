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

logger = logging.getLogger(__name__)

# Default harvester system prompt
DEFAULT_HARVESTER_PROMPT = """You are a news researcher for a positive news channel focused on solarpunk themes.

Search for RECENT (today/yesterday) news about:
- Environmental progress and climate solutions
- Technology breakthroughs benefiting humanity
- Community initiatives and local heroes
- Urban development and sustainable cities
- Animal conservation and nature wins
- Renewable energy milestones
- Social innovation and cooperation

STRICT FILTERS - EXCLUDE:
- Wars, conflicts, military news
- Political scandals or corruption
- Crime, violence, accidents
- Celebrity gossip or entertainment drama
- Market crashes or economic doom
- Disease outbreaks (unless cure/solution)

CRITICAL: You MUST respond with ONLY a valid JSON array. No markdown, no explanation, no preamble.
Start your response with [ and end with ].

JSON format:
[
  {
    "headline": "Short catchy headline",
    "summary": "2-3 sentence summary of the positive news",
    "url": "source URL from your search",
    "category": "environment|technology|community|urban|nature|energy|innovation",
    "significance_score": 7
  }
]

Return 5-10 ACTIONABLE, INSPIRING, SOLUTION-ORIENTED stories."""


async def harvest_perplexity_node(state: NewsEngineState) -> dict[str, Any]:
    """Use Perplexity to harvest news with built-in search."""
    tenant_id = state.get("tenant_id", "unknown")
    logger.info(f"[{tenant_id}] Harvesting via Perplexity")

    config = get_core_config()

    if not config.perplexity.api_key:
        logger.warning("Perplexity API key not configured, skipping")
        return {"harvest_errors": ["Perplexity API key not configured"]}

    # Get prompt override or use default
    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("harvester", DEFAULT_HARVESTER_PROMPT)

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
                user_prompt = f"Find today's positive news. Need at least {min_items} stories. Attempt {attempt + 1}."
            else:
                user_prompt = "Find today's positive news matching the criteria."

            request = ModelRequest(
                system=system_prompt,
                parts=[TextPart(text=user_prompt)],
                temperature=0.3 + (attempt * 0.1),  # Slightly higher temp on retry
                max_output_tokens=8000,  # Extra for reasoning models
            )

            response = await model.generate(request)

            # Parse response
            import json
            import re

            try:
                text = response.text
                logger.debug(f"Perplexity raw response length: {len(text)}")
                logger.info(f"Perplexity response preview: {text[:500]}...")

                # Look for JSON array of objects - pattern: [ { ... } ]
                # Find first [ that is followed by {
                match = re.search(r"\[\s*\{", text)
                if match:
                    start = match.start()
                    # Find matching closing ]
                    depth = 0
                    end = start
                    for i, char in enumerate(text[start:]):
                        if char == "[":
                            depth += 1
                        elif char == "]":
                            depth -= 1
                            if depth == 0:
                                end = start + i + 1
                                break

                    json_str = text[start:end]
                    logger.debug(f"Extracted JSON ({len(json_str)} chars): {json_str[:200]}...")
                    items = json.loads(json_str)
                else:
                    logger.warning("No JSON array of objects found in response")
                    # Try to extract from code block
                    code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
                    if code_match:
                        items = json.loads(code_match.group(1))
                    else:
                        items = []
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Perplexity response: {e}")
                items = []

            # Mark source and add to collection
            for item in items:
                item["source"] = "perplexity"
                # Dedupe by headline
                if not any(
                    existing.get("headline") == item.get("headline") for existing in all_items
                ):
                    all_items.append(item)

            logger.info(
                f"[{tenant_id}] Perplexity attempt {attempt + 1}: got {len(items)} items, total unique: {len(all_items)}"
            )

            # Check if we have enough
            if len(all_items) >= min_items:
                break

            if attempt < max_retries - 1:
                logger.info(
                    f"[{tenant_id}] Retrying Perplexity - need {min_items - len(all_items)} more items"
                )
                import asyncio

                await asyncio.sleep(1)  # Brief pause between retries

        except Exception as e:
            logger.error(f"Perplexity harvest attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {
                    "raw_items": all_items,
                    "harvest_source": "perplexity",
                    "harvest_errors": [f"Perplexity error: {str(e)}"],
                }

    logger.info(f"[{tenant_id}] Perplexity returned {len(all_items)} total items")

    return {
        "raw_items": all_items,
        "harvest_source": "perplexity",
    }


async def harvest_llm_fallback_node(state: NewsEngineState) -> dict[str, Any]:
    """Fallback to default LLM if Perplexity failed or returned no results."""
    existing_items = state.get("raw_items", [])
    errors = state.get("harvest_errors", [])

    # Skip if we already have items
    if existing_items:
        return {}

    tenant_id = state.get("tenant_id", "unknown")
    config = get_core_config()

    logger.info(f"[{tenant_id}] Falling back to default LLM ({config.models.default_llm})")

    # Get prompt override or use default
    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("harvester", DEFAULT_HARVESTER_PROMPT)

    try:
        model = model_registry.get_llm_with_fallback(
            key=config.models.default_llm,
            fallback_keys=[],
            strategy="fallback",
            config=config,
        )

        user_prompt = """Find today's positive news matching the criteria.
Focus on recent stories from the last 24 hours.
Return 5-10 actionable, inspiring stories."""

        request = ModelRequest(
            system=system_prompt,
            parts=[TextPart(text=user_prompt)],
            temperature=0.5,
            max_output_tokens=8000,  # Extra for reasoning models
        )

        response = await model.generate(request)

        # Parse response
        import json
        import re

        try:
            text = response.text
            logger.debug(f"LLM fallback raw response length: {len(text)}")

            # Look for JSON array
            match = re.search(r"\[\s*\{", text)
            if match:
                start = match.start()
                depth = 0
                end = start
                for i, char in enumerate(text[start:]):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            end = start + i + 1
                            break

                json_str = text[start:end]
                items = json.loads(json_str)
            else:
                # Try code block extraction
                code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
                if code_match:
                    items = json.loads(code_match.group(1))
                else:
                    items = []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM fallback response: {e}")
            items = []

        # Mark source
        for item in items:
            item["source"] = "llm_fallback"

        logger.info(f"[{tenant_id}] LLM fallback returned {len(items)} items")

        return {
            "raw_items": items,
            "harvest_source": "llm_fallback",
        }

    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return {
            "harvest_errors": errors + [f"LLM fallback error: {str(e)}"],
        }


async def store_raw_to_brain_node(state: NewsEngineState) -> dict[str, Any]:
    """Store raw harvested items to Brain for audit trail."""
    raw_items = state.get("raw_items", [])
    tenant_id = state.get("tenant_id", "default")
    harvest_source = state.get("harvest_source", "unknown")

    if not raw_items:
        return {}

    logger.info(f"[{tenant_id}] Storing {len(raw_items)} raw items to Brain")

    config = get_core_config()

    try:
        # Import Brain client
        import grpc
        from contextcore import brain_pb2, brain_pb2_grpc

        channel = grpc.aio.insecure_channel(config.brain.grpc_endpoint)
        stub = brain_pb2_grpc.BrainServiceStub(channel)

        stored_count = 0
        for item in raw_items:
            try:
                import uuid

                item_id = str(uuid.uuid4())

                news_item = brain_pb2.NewsItem(
                    id=item_id,
                    tenant_id=tenant_id,
                    url=item.get("url", ""),
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    category=item.get("category", ""),
                    source_api=harvest_source,
                    metadata={
                        "significance_score": str(item.get("significance_score", 5)),
                        "source": item.get("source", ""),
                    },
                )

                request = brain_pb2.UpsertNewsItemRequest(
                    tenant_id=tenant_id,
                    item=news_item,
                    item_type="raw",
                )

                response = await stub.UpsertNewsItem(request)
                if response.success:
                    stored_count += 1
                    # Add ID back to item for tracking
                    item["brain_id"] = response.id

            except Exception as e:
                logger.warning(f"Failed to store raw item to Brain: {e}")

        await channel.close()
        logger.info(f"[{tenant_id}] Stored {stored_count}/{len(raw_items)} raw items to Brain")

    except ImportError:
        logger.warning("contextcore not available, skipping Brain storage")
    except Exception as e:
        logger.error(f"Brain storage error: {e}")

    return {}


def _should_fallback(state: NewsEngineState) -> str:
    """Decide if we need LLM fallback."""
    items = state.get("raw_items", [])
    errors = state.get("harvest_errors", [])

    if items:
        return "store_to_brain"
    if errors:
        return "llm_fallback"
    return END


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
