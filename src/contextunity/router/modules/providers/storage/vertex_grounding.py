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

from collections.abc import Sequence
from typing import TypedDict, TypeGuard

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ProviderError
from contextunity.core.types import is_json_dict

from contextunity.router.core import get_core_config
from contextunity.router.modules.retrieval.rag.settings import get_effective_data_store_id

logger = get_contextunit_logger(__name__)


class VertexGroundingCitation(TypedDict):
    """Citation record extracted from Vertex grounding metadata."""

    title: str
    uri: str
    chunk_id: str


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _safe_getattr(obj: object, name: str, default: object = "") -> object:
    return getattr(obj, name, default)


def _obj_attr(obj: object, name: str, default: str = "") -> str:
    val: object = _safe_getattr(obj, name, default)
    if val is None:
        return default
    return str(val)


def _obj_seq(obj: object, name: str) -> tuple[object, ...]:
    raw: object = _safe_getattr(obj, name, None)
    if raw is None:
        return ()
    if _is_object_list(raw):
        return tuple(raw)
    if _is_object_tuple(raw):
        return raw
    return (raw,)


class _GenAIModelsAdapter:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def generate_content(
        self,
        *,
        model: str,
        contents: list[str],
        config: object,
    ) -> object:
        generate_obj: object = _safe_getattr(self._inner, "generate_content")
        if not callable(generate_obj):
            raise ProviderError(
                "Gen AI models.generate_content is not callable",
                code="VERTEX_GROUNDING_IMPORT_ERROR",
            )
        return generate_obj(model=model, contents=contents, config=config)


class _GenAIClientAdapter:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    @property
    def models(self) -> _GenAIModelsAdapter:
        return _GenAIModelsAdapter(_safe_getattr(self._inner, "models"))


class _GenAIClientFactoryAdapter:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __call__(
        self,
        *,
        vertexai: bool,
        project: str,
        location: str,
    ) -> _GenAIClientAdapter:
        constructor_obj: object = _safe_getattr(self._inner, "__call__")
        if not callable(constructor_obj):
            raise ProviderError(
                "Gen AI Client factory is not callable",
                code="VERTEX_GROUNDING_IMPORT_ERROR",
            )
        inner_obj: object = constructor_obj(vertexai=vertexai, project=project, location=location)
        return _GenAIClientAdapter(inner_obj)


class _GenAITypeFactoryAdapter:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __call__(self, **kwargs: object) -> object:
        constructor_obj: object = _safe_getattr(self._inner, "__call__")
        if not callable(constructor_obj):
            raise ProviderError(
                "Gen AI type factory is not callable",
                code="VERTEX_GROUNDING_IMPORT_ERROR",
            )
        built: object = constructor_obj(**kwargs)
        return built


def _rag_chunk_id(rag_chunk: object) -> str:
    if is_json_dict(rag_chunk):
        raw = rag_chunk.get("chunk_id", "")
        return str(raw) if raw is not None else ""
    return _obj_attr(rag_chunk, "chunk_id")


def _citation_from_metadata_item(cit: object) -> VertexGroundingCitation:
    uri = _obj_attr(cit, "uri") or _obj_attr(cit, "url")
    return {
        "title": _obj_attr(cit, "title"),
        "uri": uri,
        "chunk_id": "",
    }


def _citation_from_retrieved_context(ctx: object) -> VertexGroundingCitation:
    chunk_id = ""
    rag_chunk_raw: object = _safe_getattr(ctx, "rag_chunk", None)
    rag_chunk: object | None = rag_chunk_raw if rag_chunk_raw is not None else None
    if rag_chunk is not None:
        chunk_id = _rag_chunk_id(rag_chunk)
    return {
        "title": _obj_attr(ctx, "title"),
        "uri": _obj_attr(ctx, "uri"),
        "chunk_id": chunk_id,
    }


def _extract_text_from_candidate(candidate: object) -> str:
    content_raw: object = _safe_getattr(candidate, "content", None)
    content: object | None = content_raw if content_raw is not None else None
    if content is None:
        return ""
    parts = _obj_seq(content, "parts")
    text_parts: list[str] = []
    for part in parts:
        text: object | None = _safe_getattr(part, "text", None)
        if isinstance(text, str) and text:
            text_parts.append(text)
    return "".join(text_parts)


def _extract_citations_from_candidate(candidate: object) -> list[VertexGroundingCitation]:
    citations: list[VertexGroundingCitation] = []

    citation_metadata_raw: object = _safe_getattr(candidate, "citation_metadata", None)
    citation_metadata: object | None = (
        citation_metadata_raw if citation_metadata_raw is not None else None
    )
    if citation_metadata is not None:
        for cit in _obj_seq(citation_metadata, "citations"):
            citations.append(_citation_from_metadata_item(cit))

    grounding_metadata_raw: object = _safe_getattr(candidate, "grounding_metadata", None)
    grounding_metadata: object | None = (
        grounding_metadata_raw if grounding_metadata_raw is not None else None
    )
    if grounding_metadata is None:
        return citations

    for chunk in _obj_seq(grounding_metadata, "grounding_chunks"):
        retrieved_raw: object = _safe_getattr(chunk, "retrieved_context", None)
        retrieved: object | None = retrieved_raw if retrieved_raw is not None else None
        if retrieved is not None:
            citations.append(_citation_from_retrieved_context(retrieved))

    return citations


def _message_to_content(msg: object) -> str | None:
    """Return user/assistant content string, or None to skip (system messages)."""
    if is_json_dict(msg):
        if msg.get("type") == "system":
            return None
        raw = msg.get("content", "")
        return str(raw) if raw is not None else ""
    msg_type: object | None = _safe_getattr(msg, "type", None)
    if msg_type == "system":
        return None
    if hasattr(msg, "content"):
        content_obj: object = _safe_getattr(msg, "content")
        if isinstance(content_obj, str):
            return content_obj
        return str(content_obj)
    return str(msg)


def _load_genai_client_factory() -> _GenAIClientFactoryAdapter:
    import importlib

    genai = importlib.import_module("google.genai")
    client_factory_obj: object = getattr(genai, "Client", None)
    if not callable(client_factory_obj):
        raise ProviderError(
            "Google Gen AI SDK Client is not callable",
            code="VERTEX_GROUNDING_IMPORT_ERROR",
        )
    return _GenAIClientFactoryAdapter(client_factory_obj)


def _load_genai_type(name: str) -> _GenAITypeFactoryAdapter:
    import importlib

    types_mod = importlib.import_module("google.genai.types")
    type_factory_obj: object = getattr(types_mod, name, None)
    if not callable(type_factory_obj):
        raise ProviderError(
            f"Google Gen AI SDK type {name!r} is not callable",
            code="VERTEX_GROUNDING_IMPORT_ERROR",
        )
    return _GenAITypeFactoryAdapter(type_factory_obj)


async def generate_with_grounding(
    *,
    query: str,
    messages: Sequence[object] | None = None,
    system_prompt: str | None = None,
    graph_facts: list[str] | None = None,
    style_prompt: str | None = None,
    filter: str | None = None,
) -> tuple[str, list[VertexGroundingCitation]]:
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

    datastore_resource = (
        f"projects/{project_id}/locations/{location}"
        "/collections/default_collection"
        f"/dataStores/{data_store_id}"
    )

    logger.info(
        "Vertex Grounding START: query=%r datastore=%s location=%s",
        query[:80],
        data_store_id,
        location,
    )

    try:
        client_factory = _load_genai_client_factory()
        client = client_factory(
            vertexai=True,
            project=project_id,
            location=cfg.vertex.location or "us-central1",
        )

        vertex_ai_search_kwargs: dict[str, str] = {
            "datastore": datastore_resource,
        }
        if filter:
            vertex_ai_search_kwargs["filter"] = filter
            logger.info("Vertex Grounding: applying hard filter: %r", filter)

        vertex_ai_search = _load_genai_type("VertexAISearch")(**vertex_ai_search_kwargs)
        retrieval = _load_genai_type("Retrieval")(
            vertex_ai_search=vertex_ai_search,
            disable_attribution=False,
        )
        grounding_tool = _load_genai_type("Tool")(retrieval=retrieval)

        system_instruction_parts: list[str] = []

        if system_prompt:
            system_instruction_parts.append(system_prompt.strip())

        if graph_facts:
            graph_facts_text = "\n".join([str(f).strip() for f in graph_facts if str(f).strip()])
            if graph_facts_text:
                system_instruction_parts.append(
                    "\n\n=== GRAPH FACTS (Use for Logic/Reasoning) ===\n" + graph_facts_text
                )

        if style_prompt:
            system_instruction_parts.append("\n\n" + style_prompt.strip())

        system_instruction = (
            "\n".join(system_instruction_parts).strip() if system_instruction_parts else None
        )

        if system_instruction:
            logger.info(
                "Vertex Grounding: system_instruction length=%d preview=%r",
                len(system_instruction),
                system_instruction[:200] if len(system_instruction) > 200 else system_instruction,
            )
        else:
            logger.warning("Vertex Grounding: no system_instruction provided!")

        contents_list: list[str] = []
        if messages:
            for msg in messages:
                content = _message_to_content(msg)
                if content is not None:
                    contents_list.append(content)
        contents_list.append(query)

        config_dict: dict[str, object] = {
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "tools": [grounding_tool],
        }
        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        generate_content_config = _load_genai_type("GenerateContentConfig")(**config_dict)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents_list,
            config=generate_content_config,
        )

        response_text = ""
        citations: list[VertexGroundingCitation] = []

        candidates = _obj_seq(response, "candidates")
        if candidates:
            candidate = candidates[0]
            response_text = _extract_text_from_candidate(candidate)
            citations = _extract_citations_from_candidate(candidate)
            if citations:
                logger.debug(
                    "Vertex Grounding: extracted %d citations from candidate metadata",
                    len(citations),
                )
            else:
                logger.warning("No grounding_chunks found in grounding_metadata")
        else:
            logger.warning("No grounding_metadata found in candidate")

        if not response_text:
            response_text = str(response)

        logger.info(
            "Vertex Grounding COMPLETE: query=%r response_len=%d citations=%d filter=%s",
            query[:80],
            len(response_text),
            len(citations),
            filter or "none",
        )

        if filter and len(citations) == 0:
            logger.warning(
                (
                    "Vertex Grounding: filter applied but no citations found - "
                    "grounding found no relevant documents. "
                    "This likely means the query doesn't match any filtered documents."
                )
            )
            return "", []

        return response_text, citations

    except ImportError as e:
        logger.error("Google Gen AI SDK not available: %s", e)
        raise ProviderError(
            (
                "Google Gen AI SDK not installed. Install with: pip install 'google-genai' "
                "or install contextunity.router with vertex extras: "
                "pip install 'contextunity.router[vertex]'"
            ),
            code="VERTEX_GROUNDING_IMPORT_ERROR",
        ) from e
    except Exception as e:
        logger.exception("Vertex Grounding failed for query: %s", query[:50])
        raise ProviderError(
            f"Vertex Grounding failed: {str(e)}",
            code="VERTEX_GROUNDING_ERROR",
            query=query[:50],
        ) from e


__all__ = ["VertexGroundingCitation", "generate_with_grounding"]
