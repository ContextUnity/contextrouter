"""
Heuristic planning utilities for the Showrunner.

Used as fallback when LLM planning fails.
"""

from __future__ import annotations

from typing import Any, Dict

# Category to agent mapping
CATEGORY_AGENTS = {
    "environment": "eco_futurist",
    "technology": "tech_optimist",
    "community": "community_voice",
    "urban": "urbanist",
    "nature": "ethologist",
    "energy": "eco_futurist",
    "innovation": "tech_optimist",
    "wildlife": "ethologist",
    "lifestyle": "lifestyle_guru",
    "economy": "prosperity_observer",
    "culture": "culture_critic",
    "justice": "justice_reformer",
    "trends": "constructive_analyst",
    "ukraine_local": "ukraine_correspondent",
}

# Category to angle mapping
CATEGORY_ANGLES = {
    "environment": "Show the momentum of green transition",
    "technology": "Find the human benefit behind the tech",
    "community": "Tell the personal story behind the numbers",
    "urban": "Describe the city as a living, breathing space",
    "nature": "Highlight how nature adapts and thrives",
    "energy": "Frame as inevitable energy revolution",
    "innovation": "Connect innovation to everyday life impact",
    "wildlife": "Reveal the surprising intelligence in animal behavior",
    "lifestyle": "Celebrate the new luxury of simplicity",
    "economy": "Make the numbers personal",
    "culture": "Find significance in pop culture shifts",
    "justice": "Focus on rehabilitation and community safety",
    "trends": "Connect individual stories to larger movements",
    "ukraine_local": "Share the warmth of Ukrainian community spirit",
}


def create_angle(fact: dict) -> str:
    """Generate a creative angle for the story based on category."""
    category = fact.get("category", "unknown")
    return CATEGORY_ANGLES.get(category, "Find the human angle")


def heuristic_plan(facts: list) -> Dict[str, Any]:
    """Fallback heuristic planning when LLM is unavailable.

    Assigns agents based on category and creates default angles.
    """
    stories = []
    seen_categories = set()

    for fact in facts[:10]:
        category = fact.get("category", "unknown")

        # Assign agent based on category
        agent = CATEGORY_AGENTS.get(category, "constructive_analyst")

        # Use suggested agents if available
        if fact.get("suggested_agents"):
            agent = fact["suggested_agents"][0]

        stories.append(
            {
                "fact": fact,
                "assigned_agent": agent,
                "angle": create_angle(fact),
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


__all__ = [
    "CATEGORY_AGENTS",
    "CATEGORY_ANGLES",
    "create_angle",
    "heuristic_plan",
]
