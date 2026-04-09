"""
Agents subgraph - generate posts using personas.

Uses LLM with persona-specific prompts to generate
unique voices for each reporter.
"""

from __future__ import annotations

from typing import Any, Dict

from contextcore import get_context_unit_logger
from langgraph.graph import END, StateGraph

from contextrouter.core import get_core_config
from contextrouter.cortex.graphs.config_resolution import get_node_manifest_config
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

logger = get_context_unit_logger(__name__)


async def load_context_node(state: NewsEngineState) -> Dict[str, Any]:
    """Load RAG context - similar past posts for each story."""
    stories = state.get("stories", [])
    tenant_id = state.get("tenant_id", "default")

    if not stories:
        return {"similar_posts": []}

    logger.info("[%s] Loading context for %s stories", tenant_id, len(stories))

    config = get_core_config()

    try:
        from contextcore import BrainClient

        from contextrouter.core.brain_token import get_brain_service_token

        client = BrainClient(host=config.brain.grpc_endpoint, token=get_brain_service_token())

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
        logger.warning("Failed to load context: %s", e)
        for story in stories:
            story["similar_posts"] = []

    return {"stories": stories}


async def generate_posts_node(state: NewsEngineState) -> Dict[str, Any]:
    """Generate posts using persona prompts."""
    stories = state.get("stories", [])
    tenant_id = state.get("tenant_id", "default")

    if not stories:
        return {"posts": [], "generation_errors": []}

    logger.info("[%s] Generating posts for %s stories", tenant_id, len(stories))

    config = get_core_config()
    overrides = state.get("prompt_overrides", {})
    agent_overrides = overrides.get("agents", {})

    posts = []
    errors = []

    # Initialize LanguageTool if enabled (will be closed at the end)
    if config.news_engine.language_tool_enabled:
        init_language_tool(lang=config.news_engine.language_tool_lang)

    try:
        node_config = get_node_manifest_config(state, "generate")
        model_name = node_config.get("model", config.models.default_llm)

        model = model_registry.create_llm(
            model_name,
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
        logger.info("[%s] Generating %s posts in parallel", tenant_id, len(requests))

        async def generate_single(request: ModelRequest, meta: dict) -> dict | None:
            """Generate a single post, return None on failure or empty content."""
            try:
                response = await model.generate(request)
                content = response.text.strip()

                # Validate content is not empty
                if not content:
                    logger.warning("Empty response for %s - skipping", meta["headline"][:50])
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
                logger.error("Generation failed for %s: %s", meta["headline"][:50], e)
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
        logger.error("Generation setup failed: %s", e)
        errors.append(f"Setup: {str(e)}")

    finally:
        # Always close LanguageTool server
        if config.news_engine.language_tool_enabled:
            close_language_tool()

    logger.info("[%s] Generated %s posts, %s errors", tenant_id, len(posts), len(errors))

    return {
        "posts": posts,
        "generation_errors": errors,
    }


async def store_posts_node(state: NewsEngineState) -> Dict[str, Any]:
    """Store generated posts by calling the federated tool on the client."""
    posts = state.get("posts", [])
    tenant_id = state.get("tenant_id", "default")

    if not posts:
        return {"result": {"status": "no_posts", "posts_count": 0}}

    logger.info(
        "[%s] Calling federated tool 'store_news_results' to store %s posts", tenant_id, len(posts)
    )

    try:
        from contextcore.sdk.clients.router import RouterClient

        # We instantiate a stateless router client
        # In the context of a graph execution on contextrouter,
        # we can still make an outgoing gRPC call to contextrouter's self,
        # which will proxy it to the connected bi-di client.
        client = RouterClient()
        await client.execute_tool(
            tool_name="store_news_results", target_project=tenant_id, args={"posts": posts}
        )
        logger.info("[%s] Successfully executed 'store_news_results'.", tenant_id)

    except Exception as e:
        logger.error("[%s] Failed to store posts via federated tool: %s", tenant_id, e)

    return {
        "result": {
            "status": "completed",
            "posts_count": len(posts),
            "errors_count": len(state.get("generation_errors", [])),
        },
    }


def create_agents_subgraph():
    """Build the agents subgraph."""
    from contextrouter.cortex.graphs.secure_node import make_secure_node

    workflow = StateGraph(NewsEngineState)

    logger.debug("Building agents subgraph")

    try:
        from contextrouter.core import get_core_config

        provider = get_core_config().models.default_llm.split("/")[0]
    except Exception:
        provider = "openai"

    secure_load = make_secure_node("load_context", load_context_node)
    secure_generate = make_secure_node("generate", generate_posts_node, model_secret_ref=provider)
    secure_store = make_secure_node("store", store_posts_node)

    workflow.add_node("load_context", secure_load)
    workflow.add_node("generate", secure_generate)
    workflow.add_node("store", secure_store)
    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "generate")
    workflow.add_edge("generate", "store")
    workflow.add_edge("store", END)

    return workflow.compile()
