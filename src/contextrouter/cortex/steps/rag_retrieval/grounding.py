"""Grounding step (native LLM grounding).

When vertex_grounding provider is active, this step bypasses explicit retrieval
and generation, calling the LLM directly with grounding enabled.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from contextrouter.cortex import AgentState
from contextrouter.cortex.models import Citation
from contextrouter.cortex.prompting import RAG_SYSTEM_PROMPT
from contextrouter.cortex.services import get_graph_service
from contextrouter.modules.providers.storage.vertex_grounding import generate_with_grounding

logger = logging.getLogger(__name__)


async def generate_with_native_grounding(state: AgentState) -> dict[str, object]:
    """Generate response using native LLM grounding.

    This bypasses the explicit RAG pipeline (retrieval, reranking, custom prompts).
    The LLM automatically retrieves from the datastore and generates a response.

    Now supports:
    - Custom system prompt (via rag_system_prompt_override)
    - Graph facts enrichment
    - Style prompt
    """
    user_query = state.get("user_query", "")
    messages = state.get("messages", [])
    intent_text = state.get("intent_text", user_query)

    if not user_query:
        logger.warning("Grounding: no user_query in state")
        return {
            "messages": [AIMessage(content="I need a question to answer.")],
            "citations": [],
            "generation_complete": True,
        }

    # Build system prompt similar to build_rag_prompt
    rag_system_prompt_override = str(state.get("rag_system_prompt_override") or "")
    system_prompt_template = rag_system_prompt_override.strip() or RAG_SYSTEM_PROMPT

    logger.debug(
        "Grounding: system_prompt_template length=%d has_override=%s",
        len(system_prompt_template),
        bool(rag_system_prompt_override.strip()),
    )

    # Format system prompt (without context since grounding retrieves automatically)
    # For grounding, we format with empty context since retrieval is automatic
    # The rag_system_prompt_override from API should already contain all business logic and restrictions
    system_prompt = system_prompt_template.format(
        query=intent_text or user_query,
        context="",  # Grounding retrieves automatically, so no explicit context
        graph_facts="",  # Will be added separately
    )

    logger.debug(
        "Grounding: formatted system_prompt length=%d preview=%r",
        len(system_prompt),
        system_prompt[:300] if len(system_prompt) > 300 else system_prompt,
    )

    # Get graph facts from taxonomy_concepts (same logic as RetrievalPipeline._get_graph_facts)
    # NOTE: detect_intent node runs BEFORE routing, so taxonomy_concepts should be in state
    # But we check anyway and fetch graph_facts if taxonomy_concepts exist
    taxonomy_concepts = state.get("taxonomy_concepts") or []
    graph_facts = state.get("graph_facts", []) or []

    logger.debug(
        "Grounding: taxonomy_concepts in state: count=%d concepts=%s",
        len(taxonomy_concepts),
        taxonomy_concepts[:5] if taxonomy_concepts else [],
    )

    # If graph_facts are not in state yet, fetch them from GraphService using taxonomy_concepts
    # This is the same logic as RetrievalPipeline._get_graph_facts()
    if not graph_facts and taxonomy_concepts:
        logger.info(
            "Grounding: fetching graph facts for concepts=%d concepts_list=%s",
            len(taxonomy_concepts),
            taxonomy_concepts[:5] if taxonomy_concepts else [],
        )
        try:
            service = get_graph_service()
            graph_facts = service.get_facts(taxonomy_concepts)[:50]
            logger.info(
                "Grounding: graph facts fetched: concepts=%d facts=%d",
                len(taxonomy_concepts),
                len(graph_facts),
            )
        except Exception:
            logger.exception(
                "Grounding: graph facts lookup failed: concepts=%s", taxonomy_concepts[:5]
            )
            graph_facts = []
    elif not taxonomy_concepts:
        logger.debug("Grounding: no taxonomy_concepts in state - graph facts will be empty")

    # Get style prompt
    style_prompt = str(state.get("style_prompt") or "")

    # Get hard filter for grounding (e.g., "source_type: ANY('book', 'video')")
    # This is a universal filter that can be set by the API to restrict what grounding searches
    rag_filter = str(state.get("rag_filter") or "").strip() or None

    logger.info(
        "Grounding: generating with native LLM grounding for query=%r graph_facts=%d filter=%s",
        user_query[:80],
        len(graph_facts),
        rag_filter or "none",
    )

    try:
        response_text, citations = await generate_with_grounding(
            query=user_query,
            messages=messages,
            system_prompt=system_prompt,
            graph_facts=graph_facts,
            style_prompt=style_prompt,
            filter=rag_filter,
        )

        # If filter is applied but no citations found, return "no results" message
        # This prevents LLM from generating answers from its own knowledge when no relevant documents exist
        if rag_filter and len(citations) == 0:
            logger.warning(
                "Grounding: filter applied but no citations found - no relevant documents in filtered datastore"
            )
            # Use no_results_response function to properly format the prompt with conversation history
            from langchain_core.messages import AIMessage as LangChainAIMessage
            from langchain_core.messages import HumanMessage

            from contextrouter.cortex.steps.rag_retrieval.no_results import no_results_response

            # Format conversation history from messages (same logic as RAGStrategy._format_history)
            conversation_history = ""
            if messages:
                history_parts = []
                # Get last 10 messages excluding the current query (last message)
                for msg in list(messages[:-1])[-10:]:
                    if isinstance(msg, HumanMessage) or (
                        hasattr(msg, "type") and msg.type == "human"
                    ):
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        content = " ".join((str(content) or "").split())  # Normalize whitespace
                        if content:
                            history_parts.append(f"User: {content[:500]}")
                    elif isinstance(msg, LangChainAIMessage) or (
                        hasattr(msg, "type") and msg.type == "ai"
                    ):
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        content = " ".join((str(content) or "").split())  # Normalize whitespace
                        if content:
                            history_parts.append(f"Assistant: {content[:500]}")
                conversation_history = "\n".join(history_parts)

            # Generate proper no-results response using LLM
            no_results_msg = await no_results_response(
                user_query=user_query,
                conversation_history=conversation_history,
                prompt_override=str(state.get("no_results_prompt") or ""),
            )

            # Extract content from AIMessage
            response_content = (
                no_results_msg.content
                if hasattr(no_results_msg, "content")
                else str(no_results_msg)
            )

            return {
                "messages": [no_results_msg],
                "citations": [],
                "grounding_response": response_content,  # Mark that grounding was attempted
                "grounding_citations": [],
                "generation_complete": True,
            }

        # Convert citations from dicts to Citation objects
        # format_citations_to_ui expects Citation Pydantic models, not dicts
        citation_objects = []
        for cit in citations:
            # Create Citation object from grounding citation dict
            # Grounding citations have: title, uri, chunk_id
            # Use "knowledge" as source_type since grounding retrieves from knowledge base
            citation_objects.append(
                Citation(
                    title=cit.get("title", "") or "",
                    url=cit.get("uri", "") or cit.get("url", "") or "",
                    source_type="knowledge",  # Grounding retrieves from knowledge base
                    relevance=0.0,  # Grounding doesn't provide relevance scores
                    content="",  # Grounding doesn't provide content snippets
                )
            )

        logger.info(
            "Grounding COMPLETE: response_len=%d citations=%d",
            len(response_text),
            len(citation_objects),
        )

        return {
            "messages": [AIMessage(content=response_text)],
            "citations": citation_objects,  # Return Citation objects, not dicts
            "grounding_response": response_text,  # Store for generate node
            "grounding_citations": citation_objects,
            "generation_complete": True,
        }
    except Exception:
        logger.exception("Grounding failed for query: %s", user_query[:50])
        # Set grounding_response to indicate grounding was attempted but failed
        # This prevents generate node from falling back to RAG
        error_msg = "I apologize, but I encountered an error with grounding. Please try again or switch to Vertex AI Search provider."
        return {
            "messages": [AIMessage(content=error_msg)],
            "citations": [],
            "grounding_response": error_msg,  # Mark that grounding was attempted
            "grounding_citations": [],
            "generation_complete": True,
        }


__all__ = ["generate_with_native_grounding"]
