"""Search suggestions prompt templates."""

from __future__ import annotations

SEARCH_SUGGESTIONS_PROMPT = (
    "You are an assistant that suggests related topics for further exploration.\n\n"
    "## USER QUESTION\n{query}\n\n"
    "## YOUR TASK\n"
    "Generate 3-5 related topics that the user might want to explore next, based on the user's question.\n\n"
    "## RULES\n"
    "- Topics should be related to but different from the user's question\n"
    "- Keep each suggestion short (3-7 words)\n"
    "- Write suggestions in the SAME LANGUAGE as the user's question\n\n"
    "## RESPONSE FORMAT\n"
    "Respond with valid JSON ONLY:\n\n"
    '```json\n{\n  "suggestions": [\n    "Topic 1",\n    "Topic 2",\n    "Topic 3"\n  ]\n}\n```'
)

# Used when we have retrieval context (top docs / graph facts).
SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT = (
    "You are an assistant that suggests related topics for further exploration.\n\n"
    "## USER QUESTION\n{query}\n\n"
    "## CONTEXT (what the assistant is using to answer)\n{context}\n\n"
    "## YOUR TASK\n"
    "Generate 3-5 follow-up topics the user might want to explore next.\n\n"
    "## RULES\n"
    "- Suggestions MUST be grounded in the context above\n"
    "- Topics should be related to, but not duplicate, the user's question\n"
    "- Keep each suggestion short (3-7 words)\n"
    "- Write suggestions in the SAME LANGUAGE as the user's question\n\n"
    "## RESPONSE FORMAT\n"
    "Respond with valid JSON ONLY:\n\n"
    '```json\n{\n  "suggestions": [\n    "Topic 1",\n    "Topic 2",\n    "Topic 3"\n  ]\n}\n```'
)

IDENTITY_SUGGESTIONS_PROMPT = (
    "The user just asked about the assistant itself (who are you, what can you do, etc.).\n\n"
    "## USER QUESTION\n{query}\n\n"
    "## YOUR TASK\n"
    "Generate 3-5 topics the user might want to explore next.\n\n"
    "## RULES\n"
    "- If the question reveals user's interest (e.g., 'can you help with sales?'), suggest related topics\n"
    "- If generic ('who are you?'), suggest popular/foundational topics from the knowledge base\n"
    "- Keep each suggestion short (3-7 words)\n"
    "- Write suggestions in the SAME LANGUAGE as the user's question\n\n"
    "## RESPONSE FORMAT\n"
    "Respond with valid JSON ONLY:\n\n"
    '```json\n{{\n  "suggestions": [\n    "Topic 1",\n    "Topic 2",\n    "Topic 3"\n  ]\n}}\n```'
)

__all__ = [
    "SEARCH_SUGGESTIONS_PROMPT",
    "SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT",
    "IDENTITY_SUGGESTIONS_PROMPT",
]
