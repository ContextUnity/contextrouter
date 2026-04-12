"""No-results prompt templates."""

from __future__ import annotations

NO_RESULTS_PROMPT = """We searched the available sources but found no relevant content.

## CONVERSATION HISTORY
{conversation_history}

## USER QUESTION
{query}

Write a helpful response that:
- Clearly says that no relevant sources were found
- Uses a warm, conversational tone
- Suggests 3-5 alternative search queries or angles

Formatting:
- Use markdown
- Include a section titled 'Suggestions:' with bullet points

Language:
- Respond in the SAME LANGUAGE as the user's question.

Respond with ONLY the answer text."""

__all__ = ["NO_RESULTS_PROMPT"]
