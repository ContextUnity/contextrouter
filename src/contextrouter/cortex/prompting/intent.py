"""Intent detection prompt templates."""

from __future__ import annotations

INTENT_DETECTION_PROMPT = (
    "Classify user intent. Return ONLY valid JSON.\n\n"
    "INTENTS:\n"
    "- rag_and_web: Questions requiring source retrieval\n"
    "- translate: Translation requests\n"
    "- summarize: Summarization requests\n"
    "- rewrite: Rewriting/editing requests\n"
    "- identity: User asks about THIS CHATBOT they are talking to (who are you, what can you do, are you AI, your capabilities). "
    "NOT for: questions about other people/entities, philosophical questions about self/identity/purpose.\n\n"
    "FIELDS:\n"
    "- intent: one of above\n"
    "- ignore_history: true if user starts new topic\n"
    "- cleaned_query: user's request in ORIGINAL language (no 'new topic' prefix)\n"
    "- user_language: ISO 639-1 code of cleaned_query\n"
    "- retrieval_queries: 1-5 ENGLISH ONLY search queries for retrieval. Empty [] for identity intent. "
    "Translate non-English queries. Split multi-part questions. Disambiguate names vs concepts.\n"
    "- taxonomy_concepts: 0-10 canonical taxonomy terms if taxonomy context provided, else []\n\n"
    "JSON SCHEMA:\n"
    "{\n"
    '  "intent": "rag_and_web" | "translate" | "summarize" | "rewrite" | "identity",\n'
    '  "ignore_history": boolean,\n'
    '  "cleaned_query": "string",\n'
    '  "retrieval_queries": ["string"],\n'
    '  "user_language": "string",\n'
    '  "taxonomy_concepts": ["string"]\n'
    "}\n"
)

__all__ = ["INTENT_DETECTION_PROMPT"]
