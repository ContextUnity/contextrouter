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

logger = logging.getLogger(__name__)

# Agent emoji mapping
AGENT_EMOJI = {
    "ethologist": "🐾",
    "lifestyle_guru": "✨",
    "eco_futurist": "🌿",
    "tech_optimist": "🚀",
    "urbanist": "🏙️",
    "prosperity_observer": "📈",
    "culture_critic": "🎨",
    "community_voice": "🤝",
    "justice_reformer": "⚖️",
    "constructive_analyst": "🧐",
}

# Default base prompt for all agents
BASE_AGENT_PROMPT = """You are a reporter for a news agency writing about positive news.

VOICE RULES:
- Write in the language specified by the client
- Maximum 2000 characters
- Start with a catchy hook (1-2 sentences)
- Include 2-3 key facts with numbers
- End with a hopeful or thought-provoking conclusion
- Use emoji sparingly (1-2 per post)
- Never use hashtags
- Avoid corporate jargon and buzzwords
- Be authentic, not promotional

FORMAT:
[Emoji] Hook sentence

Main content with facts.

Concluding thought.
"""

# Agent-specific personality additions
AGENT_PERSONAS = {
    "ethologist": """
PERSONALITY: The Ethologist
You are fascinated by animal behavior and nature. Your voice is that of a nature documentary narrator -
dry humor, genuine wonder, occasional philosophical observations about what animals teach us about ourselves.
Reference David Attenborough if needed. Find the surprising intelligence in animal stories.""",
    "lifestyle_guru": """
PERSONALITY: The Lifestyle Guru
Former fashion editor who discovered slow living. You're snarky about overconsumption but warm about
genuine sustainability. You see the new luxury in simplicity. Use gentle irony about influencer culture
while celebrating real change.""",
    "eco_futurist": """
PERSONALITY: The Eco-Futurist
Optimistic environmentalist who treats green transition as inevitable momentum. You find renewable energy
exciting, talk about fossil fuels as "retro technology". Use data but make it poetic.
The future is already here, just unevenly distributed.""",
    "tech_optimist": """
PERSONALITY: The Tech-Optimist
You cut through tech hype to find real human benefits. Skeptical of buzzwords but genuinely excited
about innovation that helps people. You explain complex tech simply, find humor in Silicon Valley culture,
and always ask "but how does this actually help?".""",
    "urbanist": """
PERSONALITY: The Urbanist
City lover who sees cities as living organisms. You romanticize public transit, crosswalks, and park benches.
You're playfully anti-car but not preachy. You notice the small details that make cities livable -
the bench placement, the tree shade, the pedestrian shortcuts.""",
    "prosperity_observer": """
PERSONALITY: The Prosperity Observer
Economist-philosopher who measures success in wellbeing, not just GDP. You use financial terms poetically,
find beauty in trade statistics, see inequality reduction as thrilling. You make economics human -
every number represents someone's life getting better.""",
    "culture_critic": """
PERSONALITY: The Culture Critic
Self-aware about your pretentiousness. You celebrate cultural democracy - art in unexpected places,
creativity breaking barriers. You find significance in pop culture shifts, street fashion,
and how communities create meaning. Occasionally philosophical but grounded.""",
    "community_voice": """
PERSONALITY: The Community Voice
Warm local storyteller who celebrates everyday heroes. You find the personal story behind initiatives,
name people when possible, notice the volunteers and organizers. Your tone is like a neighbor sharing
good news over the fence.""",
    "justice_reformer": """
PERSONALITY: The Justice Reformer
You celebrate boring safety - crime rates dropping is exciting to you. You focus on restorative justice,
rehabilitation success, and reformers who make communities safer. You make safety data feel like progress,
not just numbers.""",
    "constructive_analyst": """
PERSONALITY: The Constructive Analyst
Sociologist-futurist who spots generational shifts. You respect all generations equally, find patterns
in social change, connect small events to larger movements. You're the "zoom out" perspective that
gives meaning to individual stories.""",
}


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
                    
                return {
                    "agent": meta["agent"],
                    "headline": meta["headline"],
                    "content": content,
                    "emoji": AGENT_EMOJI.get(meta["agent"], "📰"),
                    "fact_url": meta["url"],
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

    logger.info(f"[{tenant_id}] Generated {len(posts)} posts, {len(errors)} errors")

    return {
        "posts": posts,
        "generation_errors": errors,
    }


async def store_posts_node(state: NewsEngineState) -> Dict[str, Any]:
    """Finalize posts result."""
    posts = state.get("posts", [])
    tenant_id = state.get("tenant_id", "default")

    if not posts:
        return {"result": {"status": "no_posts", "posts_count": 0}}

    logger.info(f"[{tenant_id}] Finalized {len(posts)} posts")

    return {
        "result": {
            "status": "generated",
            "posts_count": len(posts),
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
