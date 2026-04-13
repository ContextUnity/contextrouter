"""Prompt templates owned by the brain, split by concern.

Preferred imports:
- `contextunity.router.cortex.prompting.intent`
- `contextunity.router.cortex.prompting.rag`
- `contextunity.router.cortex.prompting.no_results`
- `contextunity.router.cortex.prompting.identity`
- `contextunity.router.cortex.prompting.suggest`
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
