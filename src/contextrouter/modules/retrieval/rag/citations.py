"""Citation extraction and deduplication for the RAG capability."""

from __future__ import annotations

import logging
from collections.abc import Callable

from .models import Citation, RetrievedDoc
from .settings import get_rag_retrieval_settings

logger = logging.getLogger(__name__)


CitationBuilder = Callable[[RetrievedDoc], Citation | None]


def _debug_citations_enabled() -> bool:
    return logger.isEnabledFor(logging.DEBUG)


def _build_book(doc: RetrievedDoc) -> Citation | None:
    book_title = doc.book_title or ""
    quote = doc.quote or doc.content or ""
    return Citation(
        source_type="book",
        title=book_title or (doc.title or "Unknown Book"),
        content=quote,
        book_title=book_title,
        chapter=doc.chapter,
        chapter_number=doc.chapter_number,
        page_start=doc.page_start,
        page_end=doc.page_end,
        keywords=doc.keywords or [],
        quote=quote,
        relevance=doc.relevance,
        metadata=dict(doc.metadata or {}),
    )


def _build_video(doc: RetrievedDoc) -> Citation | None:
    video_id = doc.video_id or ""
    quote = doc.quote or doc.content or ""
    video_name = doc.video_name or doc.title or video_id or "Video"
    return Citation(
        source_type="video",
        title=video_name,
        content=quote,
        video_id=video_id,
        video_url=doc.video_url,
        timestamp=doc.timestamp,
        timestamp_seconds=doc.timestamp_seconds,
        keywords=doc.keywords or [],
        summary=doc.summary or "",
        quote=quote,
        relevance=doc.relevance,
        metadata=dict(doc.metadata or {}),
    )


def _build_qa(doc: RetrievedDoc) -> Citation | None:
    question = doc.question or ""
    answer = doc.answer or ""
    if not answer.strip():
        answer = doc.content or ""
    return Citation(
        source_type="qa",
        title=doc.session_title or question or doc.title or "Q&A Session",
        content=answer,
        session_title=doc.session_title,
        question=question,
        answer=answer,
        summary=doc.summary or "",
        keywords=doc.keywords or [],
        relevance=doc.relevance,
        metadata=dict(doc.metadata or {}),
    )


def _build_web(doc: RetrievedDoc) -> Citation | None:
    url = (doc.url or "").strip()
    if not url:
        return None
    summary = doc.summary or doc.snippet or doc.content or ""
    return Citation(
        source_type="web",
        title=doc.title or url,
        content=summary,
        url=url,
        summary=summary,
        relevance=doc.relevance,
        metadata=dict(doc.metadata or {}),
    )


def build_citations(
    documents: list[RetrievedDoc],
    *,
    citations_books: int | None = None,
    citations_videos: int | None = None,
    citations_qa: int | None = None,
    citations_web: int | None = None,
    builders: dict[str, CitationBuilder] | None = None,
) -> list[Citation]:
    """Build citations from RetrievedDoc list.

    This function is RAG-specific. Unknown source types are ignored by default.
    You can extend supported source types via `builders`.
    """

    cfg = get_rag_retrieval_settings()
    book_limit = cfg.citations_books if citations_books is None else citations_books
    video_limit = cfg.citations_videos if citations_videos is None else citations_videos
    qa_limit = cfg.citations_qa if citations_qa is None else citations_qa
    web_limit = cfg.citations_web if citations_web is None else citations_web

    book_limit = max(0, int(book_limit))
    video_limit = max(0, int(video_limit))
    qa_limit = max(0, int(qa_limit))
    web_limit = max(0, int(web_limit))

    default_builders: dict[str, CitationBuilder] = {
        "book": _build_book,
        "video": _build_video,
        "qa": _build_qa,
        "web": _build_web,
    }
    if isinstance(builders, dict):
        default_builders.update(builders)

    citations: list[Citation] = []
    seen_book_pages: set[tuple[str, float | None]] = set()
    seen_videos: set[tuple[str, str | None]] = set()
    seen_qa: set[str] = set()
    seen_web: set[str] = set()
    web_count = 0
    book_count = 0
    video_count = 0
    qa_count = 0

    if _debug_citations_enabled() and documents:
        sample = documents[0]
        logger.info("DEBUG_CITATIONS sample_doc=%r", sample.model_dump())

    for doc in documents:
        st = str(doc.source_type or "")
        builder = default_builders.get(st)
        if builder is None:
            logger.debug("Skipping unknown source_type=%s", st)
            continue

        if st == "book":
            if book_limit == 0 or book_count >= book_limit:
                continue
            key = (doc.book_title or "", doc.page_start)
            if key in seen_book_pages:
                continue
            seen_book_pages.add(key)
            c = builder(doc)
            if c is not None:
                citations.append(c)
                book_count += 1
            continue

        if st == "video":
            if video_limit == 0 or video_count >= video_limit:
                continue
            key = (doc.video_id or "", doc.timestamp)
            if key in seen_videos:
                continue
            seen_videos.add(key)
            c = builder(doc)
            if c is not None:
                citations.append(c)
                video_count += 1
            continue

        if st == "qa":
            if qa_limit == 0 or qa_count >= qa_limit:
                continue
            q = doc.question or ""
            if q in seen_qa:
                continue
            seen_qa.add(q)
            c = builder(doc)
            if c is not None:
                citations.append(c)
                qa_count += 1
            continue

        if st == "web":
            if web_limit == 0:
                continue
            url = (doc.url or "").strip()
            if not url or url in seen_web or web_count >= web_limit:
                continue
            seen_web.add(url)
            c = builder(doc)
            if c is not None:
                citations.append(c)
                web_count += 1
            continue

        # Custom types: no dedup rules by default
        c = builder(doc)
        if c is not None:
            citations.append(c)

    return citations


__all__ = ["build_citations", "CitationBuilder"]
