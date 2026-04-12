"""
Content filtering constants and prompts for the Archivist.

Contains banned keywords for content filtering and
the default archivist validation prompt.
"""

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

# Default archivist prompt for LLM validation
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

# Similarity threshold for deduplication
# Items above this similarity score are considered duplicates
# 0.85 = very similar, 0.90 = nearly identical
SIMILARITY_THRESHOLD = 0.85


__all__ = [
    "BANNED_KEYWORDS",
    "DEFAULT_ARCHIVIST_PROMPT",
    "SIMILARITY_THRESHOLD",
]
