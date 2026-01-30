"""
Showrunner subgraph steps.

Pipeline:
1. analyze_node - Score and rank facts
2. plan_node - LLM-based editorial planning
3. finalize_node - Finalize assignments
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
from .heuristics import heuristic_plan
from .prompts import DEFAULT_SHOWRUNNER_PROMPT

logger = logging.getLogger(__name__)


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

        # Build facts summary with FULL text for quality assessment
        facts_text = ""
        for i, fact in enumerate(facts[:15]):  # Limit to 15
            summary = fact.get("summary", "")
            url = fact.get("url", "")
            facts_text += f"""
{i}. {fact.get("headline", "")}
   Category: {fact.get("category", "unknown")}
   Full text: {summary}
   Source: {url}
"""

        # Use custom user prompt if provided, otherwise default
        user_template = overrides.get("showrunner_user", None)
        if user_template:
            # Replace template variable with actual facts
            user_text = user_template.replace("{{ facts_json }}", facts_text)
        else:
            user_text = f"Today's facts:\n{facts_text}\n\nCreate the editorial plan. Respond with valid JSON ONLY, no markdown code blocks."

        request = ModelRequest(
            system=system_prompt,
            parts=[TextPart(text=user_text)],
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
                logger.debug(f"Parsed plan keys: {list(plan_data.keys())}")

                stories = []
                raw_items = plan_data.get("stories", [])

                # Support alternative format: {"editorial_plan": [...]}
                if not raw_items and "editorial_plan" in plan_data:
                    if isinstance(plan_data["editorial_plan"], list):
                        raw_items = plan_data["editorial_plan"]

                logger.info(f"[{tenant_id}] LLM returned {len(raw_items)} story assignments")

                if not raw_items:
                    logger.warning(f"No stories in plan. Keys found: {list(plan_data.keys())}")
                    return heuristic_plan(facts)

                for item in raw_items:
                    fact = None

                    # 1. Try by index
                    if "fact_index" in item:
                        idx = item["fact_index"]
                        if isinstance(idx, int) and 0 <= idx < len(facts):
                            fact = facts[idx]

                    # 2. Try by headline match (if index failed)
                    if not fact and "story" in item:
                        target = item["story"].lower().strip()
                        # Try exact match first, then substring
                        for f in facts:
                            f_headline = f.get("headline", "").lower().strip()
                            if f_headline == target:
                                fact = f
                                break
                        # Fallback: substring match
                        if not fact:
                            for f in facts:
                                f_headline = f.get("headline", "").lower().strip()
                                if f_headline and (
                                    target[:50] in f_headline or f_headline[:50] in target
                                ):
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
                    else:
                        logger.debug(f"Could not match story: {item.get('story', '')[:50]}")

                if not stories:
                    logger.warning(
                        f"LLM returned {len(raw_items)} items but 0 matched facts, falling back to heuristic"
                    )
                    return heuristic_plan(facts)

                logger.info(
                    f"[{tenant_id}] Matched {len(stories)}/{len(raw_items)} stories from LLM plan"
                )

                return {
                    "editorial_plan": {
                        "stories": stories,
                        "notes": plan_data.get("notes", ""),
                    },
                    "selected_stories": stories,
                }
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse LLM plan JSON: {e}. Response preview: {text[:200]}"
                )
                return heuristic_plan(facts)
        else:
            logger.warning(f"No JSON found in LLM plan. Response preview: {text[:200]}")
            return heuristic_plan(facts)

    except Exception as e:
        logger.error(f"LLM planning failed: {e}")
        return heuristic_plan(facts)


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
