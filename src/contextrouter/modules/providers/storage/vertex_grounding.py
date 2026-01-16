"""Vertex Grounding provider (native LLM grounding).

This provider uses Google Gen AI SDK (google-genai) with Vertex AI Gemini API
and native grounding support. When grounding is enabled, the LLM automatically
retrieves from the datastore and generates responses with citations, bypassing
the explicit RAG pipeline.

Uses modern google-genai SDK instead of deprecated vertexai.generative_models.

SUPPORTED FEATURES:
- Custom system prompt (via rag_system_prompt_override)
- Graph facts enrichment (via graph_facts from state)
- Style prompt (via style_prompt from state)

LIMITATIONS:
- No reranking (LLM decides relevance)
- No custom citation formatting (uses LLM's native citations)
- Retrieval is automatic (no explicit control over retrieval queries)
"""

from __future__ import annotations

import logging
from typing import Any

from contextrouter.core import get_core_config
from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.exceptions import ProviderError
from contextrouter.core.interfaces import BaseProvider, IRead, IWrite, secured
from contextrouter.core.tokens import BiscuitToken
from contextrouter.modules.retrieval.rag.settings import get_effective_data_store_id

logger = logging.getLogger(__name__)


async def generate_with_grounding(
    *,
    query: str,
    messages: list[Any] | None = None,
    system_prompt: str | None = None,
    graph_facts: list[str] | None = None,
    style_prompt: str | None = None,
    filter: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate response using Vertex AI Gemini with native grounding.

    This function uses Vertex AI's native grounding feature, where the LLM directly
    retrieves from the datastore and generates a response in a single call, bypassing
    explicit retrieval and reranking steps.

    **Parameters:**

    - `query`: User query string
    - `messages`: Optional conversation history (list of LangChain messages)
    - `system_prompt`: Optional system prompt override (passed as `system_instruction` to the model)
    - `graph_facts`: Optional knowledge graph facts to include in `system_instruction`
    - `style_prompt`: Optional style/persona prompt (appended to `system_instruction`)
    - `filter`: Optional hard filter for VertexAISearch (e.g., `'source_type: ANY("book", "video")'`)

    **IMPORTANT LIMITATIONS:**
    1. **System Prompt Limitations**: While `system_prompt` is passed to the model via `system_instruction`,
       the effectiveness of complex instructions may be limited. The model may not strictly follow
       all constraints, especially topic restrictions or behavioral rules. This is a known limitation
       of native grounding - the LLM has more autonomy and may generate responses from its own
       knowledge even when instructed not to.
    2. **Filter Limitations**: The `filter` parameter applies a hard filter at the datastore level,
       but if no documents match the filter, the LLM may still generate responses from its own
       knowledge. The function returns empty citations in this case, but the response text may
       still contain generated content.
    3. **Testing Status**: This functionality has **NOT been thoroughly tested** and is **NOT recommended**
       for production use. While it may be faster than the traditional RAG pipeline, it lacks the
       fine-grained control and reliability of explicit retrieval + reranking + generation.
    4. **No Custom Reranking**: Unlike the traditional RAG pipeline, grounding does not support
       custom reranking models or relevance scoring. The LLM's internal retrieval is opaque.
    5. **Citation Quality**: Citations may be less reliable or complete compared to explicit retrieval.
       The function attempts to extract citations from `grounding_metadata` and `citation_metadata`,
       but the format and completeness may vary.
    **Returns:**
        Tuple of (response_text, citations_list) where:
        - `response_text`: Generated response string
        - `citations_list`: List of citation dicts with keys: `title`, `uri`, `chunk_id`
    **Raises:**
        ProviderError: If grounding fails (configuration errors, API errors, etc.)
    """
    cfg = get_core_config()
    project_id = cfg.vertex.project_id
    # IMPORTANT: Discovery Engine location is often "global" even when Vertex LLM is regional.
    # Use the same location logic as vertex_search.py to ensure consistency.
    location = (
        (getattr(cfg.vertex, "data_store_location", "") or "").strip()
        or (getattr(cfg.vertex, "discovery_engine_location", "") or "").strip()
        or "global"
    )

    if not project_id:
        logger.error("vertex.project_id must be set (TOML or env)")
        raise ProviderError(
            "Vertex project_id not configured",
            code="VERTEX_GROUNDING_CONFIG_ERROR",
        )

    try:
        data_store_id = get_effective_data_store_id()
    except ValueError as e:
        logger.error("Failed to resolve RAG data_store_id: %s", e)
        raise ProviderError(
            f"Failed to resolve datastore: {str(e)}",
            code="VERTEX_GROUNDING_DATASTORE_ERROR",
        ) from e

    # Build datastore resource name for grounding
    # Format: projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}
    datastore_resource = (
        f"projects/{project_id}/locations/{location}"
        f"/collections/default_collection"
        f"/dataStores/{data_store_id}"
    )

    logger.info(
        "Vertex Grounding START: query=%r datastore=%s location=%s",
        query[:80],
        data_store_id,
        location,
    )

    try:
        # Use google-genai SDK (modern, non-deprecated API)
        from google import genai
        from google.genai import types

        # Initialize client with Vertex AI
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=cfg.vertex.location
            or "us-central1",  # Vertex AI LLM location, not datastore location
        )

        # Create VertexAISearch instance with datastore resource name
        # Format: projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}
        # Optionally include filter for hard filtering (e.g., "source_type: ANY('book', 'video')")
        vertex_ai_search_kwargs = {
            "datastore": datastore_resource,
        }
        if filter:
            vertex_ai_search_kwargs["filter"] = filter
            logger.info("Vertex Grounding: applying hard filter: %r", filter)

        vertex_ai_search = types.VertexAISearch(**vertex_ai_search_kwargs)

        # Create Retrieval instance with VertexAISearch
        retrieval = types.Retrieval(
            vertex_ai_search=vertex_ai_search,
            disable_attribution=False,
        )

        # Create Tool with Retrieval
        grounding_tool = types.Tool(
            retrieval=retrieval,
        )

        # Build system instruction from system_prompt, graph_facts, and style_prompt
        system_instruction_parts = []

        # Add system prompt if provided
        if system_prompt:
            system_instruction_parts.append(system_prompt.strip())

        # Add graph facts if provided
        if graph_facts:
            graph_facts_text = "\n".join([str(f).strip() for f in graph_facts if str(f).strip()])
            if graph_facts_text:
                system_instruction_parts.append(
                    "\n\n=== GRAPH FACTS (Use for Logic/Reasoning) ===\n" + graph_facts_text
                )

        # Add style prompt if provided
        if style_prompt:
            system_instruction_parts.append("\n\n" + style_prompt.strip())

        system_instruction = (
            "\n".join(system_instruction_parts).strip() if system_instruction_parts else None
        )

        # Log system instruction for debugging
        if system_instruction:
            logger.info(
                "Vertex Grounding: system_instruction length=%d preview=%r",
                len(system_instruction),
                system_instruction[:200] if len(system_instruction) > 200 else system_instruction,
            )
        else:
            logger.warning("Vertex Grounding: no system_instruction provided!")

        # Build contents from messages and query
        contents_list = []
        if messages:
            # Convert LangChain messages to Content objects
            # Skip SystemMessage as we'll use system_instruction instead
            for msg in messages:
                # Skip system messages - they'll be in system_instruction
                if hasattr(msg, "type") and msg.type == "system":
                    continue
                if isinstance(msg, dict) and msg.get("type") == "system":
                    continue

                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str):
                        contents_list.append(content)
                    else:
                        contents_list.append(str(content))
                elif isinstance(msg, dict):
                    contents_list.append(str(msg.get("content", "")))
                else:
                    contents_list.append(str(msg))
        contents_list.append(query)

        # Build config with system_instruction if available
        config_dict = {
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "tools": [grounding_tool],
        }
        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        # Generate with grounding using google-genai
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Use latest grounding-capable model
            contents=contents_list,
            config=types.GenerateContentConfig(**config_dict),
        )

        # Extract response text and citations from candidate
        # google-genai response has candidates[0].content.parts[0].text
        # grounding_metadata is in candidate.grounding_metadata, not response.grounding_metadata
        response_text = ""
        citations = []

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Extract text from content.parts
            if hasattr(candidate, "content") and candidate.content:
                if hasattr(candidate.content, "parts") and candidate.content.parts:
                    text_parts = [
                        p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
                    ]
                    response_text = "".join(text_parts)

            # Extract citations from candidate.grounding_metadata or citation_metadata
            # Check citation_metadata first (might be simpler format)
            if hasattr(candidate, "citation_metadata") and candidate.citation_metadata:
                logger.debug(
                    "Found citation_metadata in candidate: citations=%s",
                    len(candidate.citation_metadata.citations)
                    if hasattr(candidate.citation_metadata, "citations")
                    and candidate.citation_metadata.citations
                    else 0,
                )
                if (
                    hasattr(candidate.citation_metadata, "citations")
                    and candidate.citation_metadata.citations
                ):
                    for cit in candidate.citation_metadata.citations:
                        citations.append(
                            {
                                "title": getattr(cit, "title", "") or "",
                                "uri": getattr(cit, "uri", "") or getattr(cit, "url", "") or "",
                                "chunk_id": "",
                            }
                        )

            # Also check grounding_metadata for retrieved_context citations
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                logger.debug(
                    "Grounding metadata available: grounding_chunks=%s",
                    len(metadata.grounding_chunks)
                    if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks
                    else 0,
                )

                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    logger.debug("Processing %d grounding chunks", len(metadata.grounding_chunks))
                    for idx, chunk in enumerate(metadata.grounding_chunks):
                        logger.debug(
                            "Chunk %d: has retrieved_context=%s has web=%s has maps=%s",
                            idx,
                            hasattr(chunk, "retrieved_context")
                            and chunk.retrieved_context is not None,
                            hasattr(chunk, "web") and chunk.web is not None,
                            hasattr(chunk, "maps") and chunk.maps is not None,
                        )

                        # Check for retrieved_context (google-genai format)
                        if hasattr(chunk, "retrieved_context") and chunk.retrieved_context:
                            ctx = chunk.retrieved_context
                            logger.debug(
                                "Retrieved context: title=%r uri=%r has_rag_chunk=%s",
                                getattr(ctx, "title", ""),
                                getattr(ctx, "uri", ""),
                                hasattr(ctx, "rag_chunk") and ctx.rag_chunk is not None,
                            )

                            # Extract chunk_id from rag_chunk if available
                            chunk_id = ""
                            if hasattr(ctx, "rag_chunk") and ctx.rag_chunk:
                                rag_chunk = ctx.rag_chunk
                                if isinstance(rag_chunk, dict):
                                    chunk_id = rag_chunk.get("chunk_id", "")
                                elif hasattr(rag_chunk, "chunk_id"):
                                    chunk_id = getattr(rag_chunk, "chunk_id", "")

                            citations.append(
                                {
                                    "title": getattr(ctx, "title", "") or "",
                                    "uri": getattr(ctx, "uri", "") or "",
                                    "chunk_id": chunk_id,
                                }
                            )
                else:
                    logger.warning("No grounding_chunks found in grounding_metadata")
            else:
                logger.warning("No grounding_metadata found in candidate")

        if not response_text:
            # Fallback to string representation
            response_text = str(response)

        logger.info(
            "Vertex Grounding COMPLETE: query=%r response_len=%d citations=%d filter=%s",
            query[:80],
            len(response_text),
            len(citations),
            filter or "none",
        )

        # If filter is applied but no citations found, this means grounding didn't find any relevant documents
        # In this case, we should return an error instead of letting LLM generate from its own knowledge
        if filter and len(citations) == 0:
            logger.warning(
                "Vertex Grounding: filter applied but no citations found - grounding found no relevant documents. "
                "This likely means the query doesn't match any filtered documents."
            )
            # Return empty response to indicate no relevant documents were found
            # The calling code should handle this and return a "no results" message
            return "", []

        return response_text, citations

    except ImportError as e:
        logger.error("Google Gen AI SDK not available: %s", e)
        raise ProviderError(
            "Google Gen AI SDK not installed. Install with: pip install 'google-genai' or install contextrouter with vertex extras: pip install 'contextrouter[vertex]'",
            code="VERTEX_GROUNDING_IMPORT_ERROR",
        ) from e
    except Exception as e:
        logger.exception("Vertex Grounding failed for query: %s", query[:50])
        raise ProviderError(
            f"Vertex Grounding failed: {str(e)}",
            code="VERTEX_GROUNDING_ERROR",
            query=query[:50],
        ) from e


class VertexGroundingProvider(BaseProvider, IRead, IWrite):
    """Vertex Grounding provider with native LLM grounding.
    **IMPORTANT:**
    This provider bypasses the explicit RAG pipeline and uses Vertex AI Gemini's
    native grounding capabilities. The LLM automatically retrieves from the datastore
    and generates responses with citations.
    **LIMITATIONS:**
    - No custom system prompt
    - No graph facts enrichment
    - No custom citation formatting
    - No reranking
    """

    @secured()
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: BiscuitToken,
    ) -> list[BisquitEnvelope]:
        """Generate response with native grounding.
        Note: This bypasses explicit retrieval and returns a generated response
        directly from the LLM with grounding. The 'limit' parameter is ignored
        as the LLM controls retrieval.
        """
        response_text, citations = await generate_with_grounding(query=query)

        # Wrap response in envelope
        # Note: This is a generated response, not retrieved documents
        env = BisquitEnvelope(
            content=response_text,
            provenance=[],
            metadata={
                "source": "vertex_grounding",
                "citations": citations,
                "grounding_enabled": True,
            },
        )
        env.add_trace("provider:vertex_grounding")
        return [env]

    @secured()
    async def write(self, data: BisquitEnvelope, *, token: BiscuitToken) -> None:
        _ = data, token
        raise NotImplementedError("VertexGroundingProvider.write is not implemented")

    async def sink(self, envelope: BisquitEnvelope, *, token: BiscuitToken) -> Any:
        await self.write(envelope, token=token)
        return None


__all__ = ["VertexGroundingProvider", "generate_with_grounding"]
