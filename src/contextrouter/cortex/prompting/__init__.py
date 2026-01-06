"""Prompt templates owned by the brain, split by concern.

Preferred imports:
- `contextrouter.cortex.prompting.intent`
- `contextrouter.cortex.prompting.rag`
- `contextrouter.cortex.prompting.no_results`
- `contextrouter.cortex.prompting.identity`
- `contextrouter.cortex.prompting.suggest`
"""

from __future__ import annotations

from .identity import IDENTITY_PROMPT, IDENTITY_RESPONSE
from .intent import INTENT_DETECTION_PROMPT
from .no_results import NO_RESULTS_PROMPT
from .rag import NO_RESULTS_RESPONSE, RAG_SYSTEM_PROMPT
from .suggest import (
    IDENTITY_SUGGESTIONS_PROMPT,
    SEARCH_SUGGESTIONS_PROMPT,
    SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT,
)

__all__ = [
    "INTENT_DETECTION_PROMPT",
    "IDENTITY_RESPONSE",
    "IDENTITY_PROMPT",
    "NO_RESULTS_RESPONSE",
    "RAG_SYSTEM_PROMPT",
    "NO_RESULTS_PROMPT",
    "SEARCH_SUGGESTIONS_PROMPT",
    "SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT",
    "IDENTITY_SUGGESTIONS_PROMPT",
]
