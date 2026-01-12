"""Search suggestions step (pure function)."""

from __future__ import annotations

import json
import logging

from contextrouter.cortex import AgentState
from contextrouter.modules.observability import retrieval_span

from ...nodes.utils import pipeline_log
from ...utils.json import strip_json_fence

logger = logging.getLogger(__name__)


def _build_suggestions_context(state: AgentState, *, max_docs: int = 12) -> str:
    retrieved_docs = state.get("retrieved_docs") or []
    graph_facts = state.get("graph_facts") or []
    taxonomy_concepts = state.get("taxonomy_concepts") or []

    parts: list[str] = []
    if taxonomy_concepts:
        parts.append(f"Taxonomy concepts: {', '.join(taxonomy_concepts[:12])}")
    if graph_facts:
        facts = [str(f).strip() for f in graph_facts if isinstance(f, str) and f.strip()]
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


async def generate_search_suggestions(state: AgentState) -> dict[str, object]:
    """Generate search suggestions for the next user turn."""

    enabled = state.get("enable_suggestions", True)
    if not enabled:
        return {"search_suggestions": []}

    user_query = state.get("user_query")
    if not user_query:
        return {"search_suggestions": []}

    from contextrouter.core import get_core_config
    from contextrouter.modules.models import model_registry
    from contextrouter.modules.models.types import ModelRequest, TextPart

    core_cfg = get_core_config()

    # Get suggestions model with fallback support
    # TODO: Update to use new config structure once breaking changes are applied
    suggestions_cfg = core_cfg.models.rag.suggestions
    suggestions_model_key = suggestions_cfg.model or core_cfg.models.default_llm
    fallback_keys = list(suggestions_cfg.fallback or [])
    strategy = suggestions_cfg.strategy or "fallback"

    model = model_registry.get_llm_with_fallback(
        key=suggestions_model_key,
        fallback_keys=fallback_keys,
        strategy=strategy,
        config=core_cfg,
    )

    # Direct model usage with new multimodal interface
    llm = model

    from contextrouter.cortex.prompting import (
        SEARCH_SUGGESTIONS_PROMPT,
        SEARCH_SUGGESTIONS_WITH_CONTEXT_PROMPT,
    )

    context = _build_suggestions_context(state)

    # NOTE: Don't use `str.format` here because the prompt templates contain JSON examples
    # with `{ ... }` which would be treated as formatting placeholders and crash.
    def _fill(template: str, *, query: str, ctx: str) -> str:
        return template.replace("{query}", query).replace("{context}", ctx)

    # Host override wins (API passes `search_suggestions_prompt_override`).
    override = state.get("search_suggestions_prompt_override")
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
        data = json.loads(raw)
        suggestions = data.get("suggestions") or []
    except Exception:
        logger.warning("Failed to parse suggestions JSON: %s", raw[:200])
        suggestions = []

    out: list[str] = []
    if isinstance(suggestions, list):
        for s in suggestions:
            if isinstance(s, str) and s.strip():
                out.append(s.strip()[:100])
    out = out[:4]

    return {"search_suggestions": out}


__all__ = ["generate_search_suggestions"]
