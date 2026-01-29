"""
Showrunner subgraph steps.

Pipeline:
1. analyze_node - Score and rank facts
2. plan_node - LLM-based editorial planning
3. assign_node - Finalize assignments
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

# Available agents
AGENTS = [
    "ethologist",
    "lifestyle_guru",
    "eco_futurist",
    "tech_optimist",
    "urbanist",
    "prosperity_observer",
    "culture_critic",
    "community_voice",
    "justice_reformer",
    "constructive_analyst",
]

# Default showrunner prompt
DEFAULT_SHOWRUNNER_PROMPT = """You are the Showrunner for Pink Pony News Agency, a positive Telegram channel.

Your job is to select the best stories and assign them to the right reporters (agents).

Available reporters and their specialties:
- ethologist: Wildlife, animal behavior, nature documentaries
- lifestyle_guru: Sustainable living, slow life, conscious consumption
- eco_futurist: Climate solutions, renewable energy, green tech
- tech_optimist: Technology benefits, innovation, digital progress
- urbanist: Cities, public transit, walkability, urban design
- prosperity_observer: Economics of wellbeing, inequality reduction
- culture_critic: Art, media, cultural democracy
- community_voice: Local heroes, grassroots initiatives
- justice_reformer: Criminal justice reform, restorative practices
- constructive_analyst: Big picture trends, generational shifts

Given a list of facts, create an editorial plan:
1. Select up to 10 best stories (diverse categories)
2. Assign each story to the most fitting reporter
3. Create a unique angle for each story

Respond with JSON:
{
  "stories": [
    {
      "fact_index": 0,
      "agent": "eco_futurist",
      "angle": "Frame as inevitable green momentum",
      "priority": 1
    }
  ],
  "rejected_indices": [2, 5],
  "notes": "Overall tone for today's edition"
}"""


async def analyze_node(state: NewsEngineState) -> Dict[str, Any]:
    """Analyze and score facts for selection."""
    facts = state.get("facts", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info(f"[{tenant_id}] Analyzing {len(facts)} facts")

    # Simple scoring based on available metadata
    scored_facts = []
    for i, fact in enumerate(facts):
        score = 0.0

        # Significance score from archivist
        score += 0.4 * (fact.get("significance_score", 5) / 10)

        # Has suggested agents
        if fact.get("suggested_agents"):
            score += 0.2

        # Has URL (verifiable source)
        if fact.get("url"):
            score += 0.2

        # Category present
        if fact.get("category") and fact.get("category") != "unknown":
            score += 0.2

        scored_facts.append(
            {
                **fact,
                "index": i,
                "selection_score": round(score, 2),
            }
        )

    # Sort by score
    scored_facts.sort(key=lambda x: x["selection_score"], reverse=True)

    return {"facts": scored_facts}


async def plan_node(state: NewsEngineState) -> Dict[str, Any]:
    """Create editorial plan using LLM."""
    facts = state.get("facts", [])
    tenant_id = state.get("tenant_id", "default")

    if not facts:
        return {
            "editorial_plan": {"stories": [], "notes": "No facts to plan"},
            "selected_stories": [],
        }

    config = get_core_config()

    overrides = state.get("prompt_overrides", {})
    system_prompt = overrides.get("showrunner", DEFAULT_SHOWRUNNER_PROMPT)

    logger.info(f"[{tenant_id}] LLM planning for {len(facts)} facts")

    try:
        model = model_registry.get_llm_with_fallback(
            key=config.models.default_llm,
            fallback_keys=config.models.fallback_llms,
            strategy="fallback",
            config=config,
        )

        # Build facts summary
        facts_text = ""
        for i, fact in enumerate(facts[:15]):  # Limit to 15
            facts_text += f"""
{i}. {fact.get("headline", "")}
   Category: {fact.get("category", "unknown")}
   Summary: {fact.get("summary", "")[:200]}
   Score: {fact.get("selection_score", 0)}
"""

        request = ModelRequest(
            system=system_prompt,
            parts=[
                TextPart(
                    text=f"Today's facts:\n{facts_text}\n\nCreate the editorial plan. Respond with valid JSON ONLY, no markdown code blocks."
                )
            ],
            temperature=0.5,
            max_output_tokens=8000,  # Extra for reasoning models
        )

        response = await model.generate(request)

        # Parse response
        text = response.text
        logger.debug(f"LLM plan raw response: {text}")

        # Strip markdown code blocks if present
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "")
        elif "```" in text:
            text = text.replace("```", "")

        start = text.find("{")
        end = text.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                plan_data = json.loads(text[start:end])

                stories = []
                raw_items = plan_data.get("stories", [])

                # Support alternative format: {"editorial_plan": [...]}
                if not raw_items and "editorial_plan" in plan_data:
                    if isinstance(plan_data["editorial_plan"], list):
                        raw_items = plan_data["editorial_plan"]

                for item in raw_items:
                    fact = None

                    # 1. Try by index
                    if "fact_index" in item:
                        idx = item["fact_index"]
                        if isinstance(idx, int) and 0 <= idx < len(facts):
                            fact = facts[idx]

                    # 2. Try by headline match (if index failed)
                    if not fact and "story" in item:
                        target = item["story"].lower()
                        for f in facts:
                            f_headline = f.get("headline", "").lower()
                            # Check for reasonable substring match
                            if f_headline and (target in f_headline or f_headline in target):
                                fact = f
                                break

                    if fact:
                        stories.append(
                            {
                                "fact": fact,
                                "assigned_agent": item.get(
                                    "reporter", item.get("agent", "constructive_analyst")
                                ),
                                "angle": item.get("angle", ""),
                                "priority": item.get("priority", 5),
                            }
                        )

                if not stories:
                    logger.warning("LLM returned 0 parsable stories, falling back to heuristic")
                    return _heuristic_plan(facts)

                return {
                    "editorial_plan": {
                        "stories": stories,
                        "notes": plan_data.get("notes", ""),
                    },
                    "selected_stories": stories,
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM plan JSON: {e}")
                return _heuristic_plan(facts)
        else:
            logger.warning("No JSON found in LLM plan, using heuristic")
            return _heuristic_plan(facts)

    except Exception as e:
        logger.error(f"LLM planning failed: {e}")
        return _heuristic_plan(facts)


def _heuristic_plan(facts: list) -> Dict[str, Any]:
    """Fallback heuristic planning."""
    stories = []
    seen_categories = set()

    for fact in facts[:10]:
        category = fact.get("category", "unknown")

        # Assign agent based on category
        category_agents = {
            "environment": "eco_futurist",
            "technology": "tech_optimist",
            "community": "community_voice",
            "urban": "urbanist",
            "nature": "ethologist",
            "energy": "eco_futurist",
            "innovation": "tech_optimist",
        }

        agent = category_agents.get(category, "constructive_analyst")

        # Use suggested agents if available
        if fact.get("suggested_agents"):
            agent = fact["suggested_agents"][0]

        stories.append(
            {
                "fact": fact,
                "assigned_agent": agent,
                "angle": _create_angle(fact),
                "priority": len(stories) + 1,
            }
        )

        seen_categories.add(category)

    return {
        "editorial_plan": {
            "stories": stories,
            "notes": "Heuristic plan (LLM unavailable)",
        },
        "selected_stories": stories,
    }


def _create_angle(fact: dict) -> str:
    """Generate a creative angle for the story."""
    category = fact.get("category", "unknown")

    category_angles = {
        "environment": "Show the momentum of green transition",
        "technology": "Find the human benefit behind the tech",
        "community": "Tell the personal story behind the numbers",
        "urban": "Describe the city as a living, breathing space",
        "nature": "Highlight how nature adapts and thrives",
        "energy": "Frame as inevitable energy revolution",
        "innovation": "Connect innovation to everyday life impact",
    }

    return category_angles.get(category, "Find the human angle")


async def finalize_node(state: NewsEngineState) -> Dict[str, Any]:
    """Finalize the editorial plan."""
    stories = state.get("selected_stories", [])
    tenant_id = state.get("tenant_id", "default")

    logger.info(f"[{tenant_id}] Finalized plan with {len(stories)} stories")

    return {
        "stories": stories,  # Pass to agents subgraph
        "result": {
            "status": "planned",
            "stories_count": len(stories),
        },
    }


def create_showrunner_subgraph():
    """Build the showrunner subgraph."""
    workflow = StateGraph(NewsEngineState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
