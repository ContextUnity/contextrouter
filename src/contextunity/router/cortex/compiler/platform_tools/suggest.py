"""Search suggestions step (pure function)."""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.types import is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict

from contextunity.router.cortex.types import GraphState, StateUpdate
from contextunity.router.cortex.utils.json import strip_json_fence
from contextunity.router.cortex.utils.pipeline import pipeline_log
from contextunity.router.modules.observability import retrieval_span
from contextunity.router.modules.retrieval.rag.models import RetrievedDoc


class SuggestConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str | None = None


logger = get_contextunit_logger(__name__)


def _dynamic_state(state: GraphState) -> dict[str, object]:
    """Return the dynamic state bag."""
    dynamic_raw = state.get("dynamic")
    if is_object_dict(dynamic_raw):
        return dict(dynamic_raw)
    return {}


def _build_suggestions_context(state: GraphState, *, max_docs: int = 12) -> str:
    dyn = _dynamic_state(state)
    retrieved_docs_raw: object = dyn.get("retrieved_docs") or []
    retrieved_docs = (
        [doc for doc in retrieved_docs_raw if isinstance(doc, RetrievedDoc)]
        if is_object_list(retrieved_docs_raw)
        else []
    )
    graph_facts_raw: object = dyn.get("graph_facts") or []
    graph_facts = (
        [f for f in graph_facts_raw if isinstance(f, str)]
        if is_object_list(graph_facts_raw)
        else []
    )
    taxonomy_raw: object = dyn.get("taxonomy_concepts") or []
    taxonomy_concepts = (
        [c for c in taxonomy_raw if isinstance(c, str)] if is_object_list(taxonomy_raw) else []
    )

    parts: list[str] = []
    if taxonomy_concepts:
        parts.append(f"Taxonomy concepts: {', '.join(taxonomy_concepts[:12])}")
    if graph_facts:
        facts = [f.strip() for f in graph_facts if f.strip()]
        if facts:
            parts.append("Graph facts:\n- " + "\n- ".join(facts[:12]))

    doc_lines: list[str] = []
    for d in retrieved_docs[:max_docs]:
        st = d.source_type or "source"
        title = d.title or "" or (d.session_title or "")
        chapter = d.chapter or ""
        page = d.page_start
        q = d.question or ""

        bits: list[str] = []
        if title:
            bits.append(title)
        if chapter:
            bits.append(f"ch: {chapter}")
        if page is not None and str(page).strip():
            bits.append(f"p: {page}")
        if q:
            bits.append(f"Q: {q[:120]}")
        if bits:
            doc_lines.append(f"- [{st}] " + " | ".join(bits))

    if doc_lines:
        parts.append("Top sources:\n" + "\n".join(doc_lines))

    return ("\n\n".join(parts) + "\n") if parts else ""


async def generate_search_suggestions(state: GraphState) -> StateUpdate:
    """Generate search suggestions for the next user turn."""

    dyn = _dynamic_state(state)
    enabled = dyn.get("enable_suggestions", True)
    if not enabled:
        return {"dynamic": {"search_suggestions": []}}

    user_query_raw = dyn.get("user_query")
    user_query = user_query_raw if isinstance(user_query_raw, str) else ""
    if not user_query:
        return {"dynamic": {"search_suggestions": []}}

    from contextunity.router.cortex.config_resolution import get_node_manifest_config
    from contextunity.router.modules.models import model_registry
    from contextunity.router.modules.models.types import ModelRequest, TextPart

    node_config = get_node_manifest_config(state, "suggest")
    model_key = node_config.get("model") or "vertex/gemini-2.5-flash-lite"

    llm = model_registry.create_llm(model_key)

    from contextunity.router.cortex.compiler.platform_tools.prompts import (
        SEARCH_SUGGESTIONS_PROMPT,
        SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT,
    )

    context = _build_suggestions_context(state)

    # NOTE: Don't use `str.format` here because the prompt templates contain JSON examples
    # with `{ ... }` which would be treated as formatting placeholders and crash.
    def _fill(template: str, *, query: str, ctx: str) -> str:
        return template.replace("{query}", query).replace("{context}", ctx)

    # Host override wins (API passes `search_suggestions_prompt_override`).
    override = dyn.get("search_suggestions_prompt_override")
    if isinstance(override, str) and override.strip():
        base_prompt = override.strip()
    else:
        base_prompt = (
            SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT if context else SEARCH_SUGGESTIONS_PROMPT
        )

    system_prompt = _fill(base_prompt, query=str(user_query), ctx=str(context or ""))

    with retrieval_span(name="suggest", input_data={"query": user_query[:200]}):
        # Build prompt from system and user messages
        full_prompt = f"{system_prompt}\n\n{user_query}"

        request = ModelRequest(
            parts=[TextPart(text=full_prompt)],
            temperature=0.2,
            max_output_tokens=256,
        )

        resp = await llm.generate(request)

    text = resp.text
    raw = strip_json_fence(text)
    pipeline_log("suggest.raw", text=raw[:200])

    try:
        payload = json_loads(raw)
        if is_object_dict(payload):
            suggestions: object = payload.get("suggestions") or []
        else:
            suggestions = []
    except Exception:  # graceful-degrade: tool failure returns empty result
        logger.warning("Failed to parse suggestions JSON: %s", raw[:200])
        suggestions = []

    out: list[str] = []
    if is_object_list(suggestions):
        for s in suggestions:
            if isinstance(s, str) and s.strip():
                out.append(s.strip()[:100])
    out = out[:4]

    return {"dynamic": {"search_suggestions": out}}


__all__ = ["generate_search_suggestions"]
