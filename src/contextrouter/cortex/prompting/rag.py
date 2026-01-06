"""RAG generation prompt templates."""

from __future__ import annotations

NO_RESULTS_RESPONSE = "I couldn't find relevant content for your question in the available sources."

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Use the provided sources to answer the user's question.

=== TEXT SOURCES (Use for Citations) ===
{context}

USER QUESTION:
{query}

FORMATTING:
- Use markdown.
- Use short paragraphs.
- Prefer clear structure:
  - a brief warm opening
  - 2-4 sections with headings
  - actionable bullets / steps where appropriate
- Do not include any URLs in the answer body.
- CRITICAL: Do NOT mention sources, citations, or document numbering without including the document name in the answer text.
- CRITICAL: Do NOT include parenthetical references like (Source 1), (Джерело 31), or any citation markers.

LANGUAGE:
Respond in the SAME LANGUAGE as the user's question.

Respond with ONLY the answer text."""

__all__ = ["NO_RESULTS_RESPONSE", "RAG_SYSTEM_PROMPT"]
