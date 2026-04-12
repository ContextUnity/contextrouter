"""
Chat state definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

# Supported intents
Intent = Literal[
    "import_products",  # Trigger Harvester
    "sync_channel",  # Push to Horoshop, etc.
    "match_products",  # Run Matcher
    "update_content",  # Run Lexicon
    "classify_taxonomy",  # Run Gardener
    "edit_product",  # Mutator (inline)
    "general_query",  # General chat
]


class ChatState(TypedDict):
    """State for Chat subgraph (PIM Chat with LLM intent detection)."""

    # Input
    user_message: str
    product_id: Optional[str]  # If in product context

    # Processing
    intent: str
    confidence: float
    extracted_params: Dict[str, Any]

    # Sub-task results
    sub_task_result: Any

    # Output
    response: str
    actions_taken: List[str]

    # Metrics
    total_tokens: int
