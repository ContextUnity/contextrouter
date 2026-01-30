"""
Agents subgraph - generate posts using personas.

Uses LLM with persona-specific prompts to generate
unique voices for each reporter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from contextrouter.core import get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from ..state import NewsEngineState
from .language_tool import apply_language_tool, close_language_tool, init_language_tool
from .personas import (
    AGENT_EMOJI,
    AGENT_HASHTAGS,
    AGENT_PERSONAS,
    AGENT_RUBRIC_NAME,
    AGENT_SIGNATURE,
    BASE_AGENT_PROMPT,
)

logger = logging.getLogger(__name__)


async def load_context_node(state: NewsEngineState) -> Dict[str, Any]:
    """Load RAG context - similar past posts for each story."""
    stories = state.get("stories", [])
    tenant_id = state.get("tenant_id", "default")

    if not stories:
        return {"similar_posts": []}

    logger.info(f"[{tenant_id}] Loading context for {len(stories)} stories")

    config = get_core_config()

    try:
        from contextcore import BrainClient

        client = BrainClient(host=config.brain.grpc_endpoint)

        for story in stories:
            fact = story.get("fact", {})
            headline = fact.get("headline", "")

            if headline:
                similar = await client.search(
                    tenant_id=tenant_id,
                    query_text=headline,
                    source_types=["news_post"],
                    limit=3,
                )
                story["similar_posts"] = [{"content": s.content} for s in similar]
            else:
                story["similar_posts"] = []

    except ImportError:
        logger.warning("contextcore not available, skipping RAG context")
        for story in stories:
            story["similar_posts"] = []
    except Exception as e:
        logger.warning(f"Failed to load context: {e}")
        for story in stories:
            story["similar_posts"] = []

    return {"stories": stories}


async def generate_posts_node(state: NewsEngineState) -> Dict[str, Any]:
    """Generate posts using persona prompts."""
    stories = state.get("stories", [])
    tenant_id = state.get("tenant_id", "default")

    if not stories:
        return {"posts": [], "generation_errors": []}

    logger.info(f"[{tenant_id}] Generating posts for {len(stories)} stories")

    config = get_core_config()
    overrides = state.get("prompt_overrides", {})
    agent_overrides = overrides.get("agents", {})

    posts = []
    errors = []

    # Initialize LanguageTool if enabled (will be closed at the end)
    if config.news_engine.language_tool_enabled:
        init_language_tool(lang=config.news_engine.language_tool_lang)

    try:
        model = model_registry.get_llm_with_fallback(
            key=config.models.default_llm,
            fallback_keys=config.models.fallback_llms,
            strategy="fallback",
            config=config,
        )

        # Get base prompt from overrides or default
        base_prompt = overrides.get("base_prompt", BASE_AGENT_PROMPT)

        # Build all requests for batch processing
        requests = []
        story_metadata = []  # Track which request maps to which story

        for story in stories:
            agent = story.get("assigned_agent", "constructive_analyst")
            fact = story.get("fact", {})
            angle = story.get("angle", "")
            similar = story.get("similar_posts", [])

            # Build system prompt
            persona = agent_overrides.get(agent, AGENT_PERSONAS.get(agent, ""))
            system_prompt = base_prompt + persona

            # Build user prompt
            context_text = ""
            if similar:
                context_text = "\nSimilar posts for style reference (DO NOT copy):\n"
                for s in similar[:2]:
                    context_text += f"- {s.get('content', '')[:200]}...\n"

            user_prompt = f"""Write a post about this news:

Headline: {fact.get("headline", "")}
Summary: {fact.get("summary", "")}
Source: {fact.get("url", "")}
Angle: {angle}
{context_text}
Write the post following the language and style rules from your system prompt."""

            request = ModelRequest(
                system=system_prompt,
                parts=[TextPart(text=user_prompt)],
                temperature=0.8,
                # Reasoning models (gpt-5, o1) need extra tokens for CoT reasoning
                # Budget: ~2000 reasoning + ~2000 response = 4000 minimum
                max_output_tokens=8000,
            )
            requests.append(request)
            story_metadata.append(
                {
                    "agent": agent,
                    "headline": fact.get("headline", ""),
                    "url": fact.get("url", ""),
                }
            )

        # Generate posts in parallel for speed
        logger.info(f"[{tenant_id}] Generating {len(requests)} posts in parallel")

        async def generate_single(request: ModelRequest, meta: dict) -> dict | None:
            """Generate a single post, return None on failure or empty content."""
            try:
                response = await model.generate(request)
                content = response.text.strip()

                # Validate content is not empty
                if not content:
                    logger.warning(f"Empty response for {meta['headline'][:50]} - skipping")
                    errors.append(f"{meta['agent']}: Empty LLM response")
                    return None

                # Apply LanguageTool grammar/spell check if enabled
                if config.news_engine.language_tool_enabled:
                    content = await apply_language_tool(
                        content,
                        auto_correct=config.news_engine.language_tool_auto_correct,
                    )

                # Add signature and hashtags
                agent_name = meta["agent"]
                signature = AGENT_SIGNATURE.get(agent_name, "")
                hashtags = AGENT_HASHTAGS.get(agent_name, "")

                # Append signature block if not already in content
                if signature and "---" not in content:
                    content = f"{content}\n\n---\n✍️ {signature}\n{hashtags}"

                return {
                    "agent": agent_name,
                    "rubric_name": AGENT_RUBRIC_NAME.get(agent_name, "📰 Новини"),
                    "headline": meta["headline"],
                    "content": content,
                    "emoji": AGENT_EMOJI.get(agent_name, "📰"),
                    "fact_url": meta["url"],
                    "signature": signature,
                    "hashtags": hashtags,
                }
            except Exception as e:
                logger.error(f"Generation failed for {meta['headline'][:50]}: {e}")
                errors.append(f"{meta['agent']}: {str(e)}")
                return None

        # Run generations with limited concurrency to avoid rate limits
        import asyncio

        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

        async def limited_generate(req, meta):
            async with semaphore:
                return await generate_single(req, meta)

        results = await asyncio.gather(
            *(limited_generate(req, meta) for req, meta in zip(requests, story_metadata)),
            return_exceptions=False,  # Exceptions handled in generate_single
        )

        # Collect successful posts (non-None only)
        for result in results:
            if result is not None:
                posts.append(result)

    except Exception as e:
        logger.error(f"Generation setup failed: {e}")
        errors.append(f"Setup: {str(e)}")

    finally:
        # Always close LanguageTool server
        if config.news_engine.language_tool_enabled:
            close_language_tool()

    logger.info(f"[{tenant_id}] Generated {len(posts)} posts, {len(errors)} errors")

    return {
        "posts": posts,
        "generation_errors": errors,
    }


async def store_posts_node(state: NewsEngineState) -> Dict[str, Any]:
    """Store generated posts to Brain for deduplication and archive."""
    posts = state.get("posts", [])
    tenant_id = state.get("tenant_id", "default")

    if not posts:
        return {"result": {"status": "no_posts", "posts_count": 0}}

    logger.info(f"[{tenant_id}] Storing {len(posts)} posts to Brain")

    config = get_core_config()
    stored_count = 0

    try:
        from contextcore import BrainClient

        client = BrainClient(host=config.brain.grpc_endpoint)

        for post in posts:
            try:
                post_id = await client.upsert_news_post(
                    tenant_id=tenant_id,
                    headline=post.get("headline", ""),
                    content=post.get("content", ""),
                    agent=post.get("agent", ""),
                    emoji=post.get("emoji", "📰"),
                    fact_url=post.get("fact_url", ""),
                    fact_id=post.get("fact_id", ""),
                )

                if post_id:
                    stored_count += 1
                    logger.debug(f"Stored post: {post_id}")
                else:
                    logger.warning(f"Failed to store post: {post.get('headline', '')[:30]}")

            except Exception as e:
                logger.warning(f"Failed to store post '{post.get('headline', '')[:30]}': {e}")

    except ImportError:
        logger.warning("contextcore not available, skipping storage")
    except Exception as e:
        logger.error(f"Storage failed: {e}")

    logger.info(f"[{tenant_id}] Stored {stored_count}/{len(posts)} posts")

    return {
        "result": {
            "status": "completed",
            "posts_count": len(posts),
            "stored_count": stored_count,
            "errors_count": len(state.get("generation_errors", [])),
        },
    }


def create_agents_subgraph():
    """Build the agents subgraph."""
    workflow = StateGraph(NewsEngineState)

    workflow.add_node("load_context", load_context_node)
    workflow.add_node("generate", generate_posts_node)
    workflow.add_node("store", store_posts_node)

    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "generate")
    workflow.add_edge("generate", "store")
    workflow.add_edge("store", END)

    return workflow.compile()
