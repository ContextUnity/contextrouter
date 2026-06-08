"""Tests for RAG pure functions: MMR selection, citations, and UI formatting.

Zero-mock tests for retrieval utility modules:
- mmr.py: Maximal Marginal Relevance document selection
- citations.py: Citation extraction, deduplication, and per-type limits
- formatting/citations.py: snake_case → camelCase UI schema conversion
"""

from __future__ import annotations

import pytest

from contextunity.router.modules.retrieval.rag.formatting.citations import (
    _clean_text,
    _coerce_page,
    _coerce_timestamp_seconds,
    format_citations_to_ui,
)
from contextunity.router.modules.retrieval.rag.mmr import (
    _jaccard,
    _tokens,
    mmr_select,
)
from contextunity.router.modules.retrieval.rag.models import Citation, RetrievedDoc

# ── Helpers ──────────────────────────────────────────────────────────


def _doc(
    source_type: str = "book",
    content: str = "test content",
    title: str | None = None,
    relevance: float = 0.9,
    **kwargs,
) -> RetrievedDoc:
    return RetrievedDoc(
        source_type=source_type, content=content, title=title, relevance=relevance, **kwargs
    )


def _citation(
    source_type: str = "video",
    title: str = "Test",
    content: str = "test",
    **kwargs,
) -> Citation:
    return Citation(source_type=source_type, title=title, content=content, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# MMR — _tokens, _jaccard, mmr_select
# ═══════════════════════════════════════════════════════════════════════


class TestTokenizer:
    """_tokens extracts lowercase tokens > 2 chars."""

    def test_basic_extraction(self):
        result = _tokens("Hello World foo")
        assert "hello" in result
        assert "world" in result
        assert "foo" in result

    def test_strips_short_tokens(self):
        result = _tokens("A is on it")
        assert "is" not in result
        assert "on" not in result
        assert "it" not in result

    def test_special_chars_ignored(self):
        result = _tokens("hello! @world #test")
        assert "hello" in result
        assert "world" in result


class TestJaccard:
    """_jaccard computes set similarity."""

    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        result = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert result == pytest.approx(2 / 4)  # intersection=2, union=4


class TestMmrSelect:
    """mmr_select: Maximal Marginal Relevance document selection."""

    def test_empty_candidates(self):
        assert mmr_select(query="test", candidates=[], k=5, lambda_mult=0.7) == []

    def test_k_greater_than_candidates_returns_all(self):
        docs = [_doc(content="a"), _doc(content="b")]
        result = mmr_select(query="test", candidates=docs, k=10, lambda_mult=0.7)
        assert result == docs

    def test_selects_k_documents(self):
        docs = [_doc(content=f"document {i} with relevant test content") for i in range(5)]
        result = mmr_select(query="relevant test", candidates=docs, k=3, lambda_mult=0.7)
        assert len(result) == 3

    def test_diverse_selection(self):
        """With low lambda (diversity-heavy), duplicates should be penalized."""
        similar_a = _doc(content="machine learning algorithms deep neural")
        similar_b = _doc(content="machine learning algorithms deep neural")
        different = _doc(content="quantum physics photon entanglement")

        result = mmr_select(
            query="machine learning",
            candidates=[similar_a, similar_b, different],
            k=2,
            lambda_mult=0.3,  # Heavy diversity weight
        )
        # With diversity-heavy lambda, should prefer one similar + the different one
        assert different in result

    def test_relevance_selection(self):
        """With high lambda (relevance-heavy), most relevant should come first."""
        relevant = _doc(content="machine learning algorithms research paper")
        irrelevant = _doc(content="cooking recipe pasta garlic bread")

        result = mmr_select(
            query="machine learning",
            candidates=[irrelevant, relevant],
            k=1,
            lambda_mult=1.0,  # Pure relevance
        )
        assert result[0] is relevant


# ═══════════════════════════════════════════════════════════════════════
# formatting/citations.py — _clean_text, _coerce_*, format_citations_to_ui
# ═══════════════════════════════════════════════════════════════════════


class TestCleanText:
    """_clean_text decodes HTML entities and strips markdown artifacts."""

    def test_html_entities(self):
        assert _clean_text("It&#39;s a test &amp; more") == "It's a test & more"

    def test_markdown_headers(self):
        assert _clean_text("Hello ## world ## test") == "Hello world test"

    def test_whitespace_normalization(self):
        assert _clean_text("  hello   world  ") == "hello world"


class TestCoerceTimestampSeconds:
    """_coerce_timestamp_seconds parses various timestamp formats."""

    def test_direct_seconds(self):
        c = _citation(timestamp_seconds=120.0)
        assert _coerce_timestamp_seconds(c) == 120

    def test_negative_returns_none(self):
        c = _citation(timestamp_seconds=-1.0)
        assert _coerce_timestamp_seconds(c) is None

    def test_zero_is_valid(self):
        """0 is a valid timestamp (start of video), not negative."""
        c = _citation(timestamp_seconds=0.0)
        assert _coerce_timestamp_seconds(c) == 0

    @pytest.mark.parametrize(
        ("ts", "expected"),
        [
            ("01:30:00", 5400),  # 1h30m
            ("05:30", 330),  # 5m30s
            ("45", 45),  # 45s
        ],
    )
    def test_string_parsing(self, ts, expected):
        c = _citation(timestamp=ts)
        assert _coerce_timestamp_seconds(c) == expected

    def test_invalid_string(self):
        c = _citation(timestamp="not:a:valid:ts:format")
        assert _coerce_timestamp_seconds(c) is None


class TestCoercePage:
    def test_float_to_int(self):
        assert _coerce_page(3.7) == 3


class TestFormatCitationsToUi:
    """format_citations_to_ui converts Citation models to camelCase UI dicts."""

    def test_video_citation_all_keys(self):
        """Assert every dict key and value in the video output."""
        c = _citation(
            source_type="video",
            title="Lecture 1",
            content="transcript",
            video_id="v123",
            video_url="https://yt.be/v123",
            timestamp="01:30",
            timestamp_seconds=90.0,
            keywords=["ml"],
            summary="A lecture",
            relevance=0.85,
        )
        result = format_citations_to_ui([c])
        assert len(result) == 1
        ui = result[0]
        assert ui["type"] == "video"
        assert ui["title"] == "Lecture 1"
        assert ui["videoId"] == "v123"
        assert ui["videoUrl"] == "https://yt.be/v123"
        assert ui["timestamp"] == "01:30"
        assert ui["timestampSeconds"] == 90
        assert ui["keywords"] == ["ml"]
        assert ui["summary"] == "A lecture"
        assert ui["quote"] == "transcript"
        assert ui["relevance"] == 0.85

    def test_book_citation_all_keys(self):
        """Assert every dict key and value in the book output."""
        c = _citation(
            source_type="book",
            title="Fallback Title",
            content="quote text",
            book_title="Real Book",
            chapter="Ch 3",
            chapter_number=3,
            page_start=42.0,
            page_end=45.0,
            keywords=["philosophy"],
            quote="actual quote",
        )
        result = format_citations_to_ui([c])
        assert len(result) == 1
        ui = result[0]
        assert ui["type"] == "book"
        assert ui["title"] == "Real Book"
        assert ui["chapter"] == "Ch 3"
        assert ui["chapterNumber"] == 3
        assert ui["pageStart"] == 42
        assert ui["pageEnd"] == 45
        assert ui["keywords"] == ["philosophy"]
        assert ui["quote"] == "actual quote"

    def test_qa_citation_all_keys(self):
        c = _citation(
            source_type="qa",
            title="Session Title",
            content="answer text",
            question="What is X?",
            answer="X is Y",
            session_title="Deep Dive",
            keywords=["topic"],
            relevance=0.7,
        )
        result = format_citations_to_ui([c])
        assert len(result) == 1
        ui = result[0]
        assert ui["type"] == "qa"
        assert ui["title"] == "Deep Dive"
        assert ui["question"] == "What is X?"
        assert ui["answer"] == "X is Y"
        assert ui["keywords"] == ["topic"]
        assert ui["relevance"] == 0.7

    def test_web_citation_all_keys(self):
        c = _citation(
            source_type="web",
            title="Article",
            content="body",
            url="https://example.com",
            summary="web summary",
        )
        result = format_citations_to_ui([c])
        assert len(result) == 1
        ui = result[0]
        assert ui["type"] == "web"
        assert ui["title"] == "Article"
        assert ui["summary"] == "web summary"
        assert ui["url"] == "https://example.com"

    def test_allowed_types_filter(self):
        citations = [
            _citation(source_type="video", title="V", content="v"),
            _citation(source_type="book", title="B", content="b"),
            _citation(source_type="qa", title="Q", content="q"),
        ]
        result = format_citations_to_ui(citations, allowed_types=["video", "qa"])
        types = [r["type"] for r in result]
        assert "video" in types
        assert "qa" in types
        assert "book" not in types


# ═══════════════════════════════════════════════════════════════════════
# citations.py — build_citations (with config monkeypatch)
# ═══════════════════════════════════════════════════════════════════════


class TestBuildCitations:
    """build_citations: extraction, deduplication, per-type limits."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        """Provide default RAG settings without hitting config system."""
        from contextunity.router.modules.retrieval.rag.settings import RagRetrievalSettings

        defaults = RagRetrievalSettings()
        monkeypatch.setattr(
            "contextunity.router.modules.retrieval.rag.citations.get_rag_retrieval_settings",
            lambda: defaults,
        )

    def test_book_citation_built(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(
                source_type="book",
                content="quote text",
                book_title="My Book",
                chapter="Ch 1",
                chapter_number=1,
                page_start=10.0,
                page_end=15.0,
                quote="explicit quote",
                keywords=["history"],
            ),
        ]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.source_type == "book"
        assert c.book_title == "My Book"
        assert c.chapter == "Ch 1"
        assert c.chapter_number == 1
        assert c.page_start == 10.0
        assert c.page_end == 15.0
        assert c.quote == "explicit quote"
        assert c.keywords == ["history"]
        assert c.content == "explicit quote"

    def test_video_citation_built(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(
                source_type="video",
                content="transcript",
                video_id="v1",
                video_url="https://yt.be/v1",
                timestamp="01:30",
                timestamp_seconds=90.0,
                keywords=["ml"],
                summary="video summary",
            ),
        ]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.source_type == "video"
        assert c.video_id == "v1"
        assert c.video_url == "https://yt.be/v1"
        assert c.timestamp == "01:30"
        assert c.timestamp_seconds == 90.0
        assert c.keywords == ["ml"]
        assert c.summary == "video summary"
        assert c.content == "transcript"

    def test_qa_citation_built(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(
                source_type="qa",
                content="answer text",
                question="Q?",
                answer="A",
                session_title="Session 1",
            ),
        ]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.question == "Q?"
        assert c.answer == "A"
        assert c.session_title == "Session 1"
        assert c.content == "A"

    def test_web_citation_needs_url(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        # Web citation without URL → skipped
        docs = [_doc(source_type="web", content="text")]
        result = build_citations(docs)
        assert len(result) == 0

    def test_web_citation_with_url(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(
                source_type="web",
                content="body text",
                url="https://example.com",
                title="Article",
                summary="web summary",
            )
        ]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.url == "https://example.com"
        assert c.title == "Article"
        assert c.summary == "web summary"
        assert c.content == "web summary"

    def test_book_deduplication_by_title_and_page(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(source_type="book", content="a", book_title="Book", page_start=10.0),
            _doc(source_type="book", content="b", book_title="Book", page_start=10.0),  # dup
            _doc(source_type="book", content="c", book_title="Book", page_start=20.0),
        ]
        result = build_citations(docs)
        assert len(result) == 2

    def test_video_deduplication_by_id_and_timestamp(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(source_type="video", content="a", video_id="v1", timestamp="01:00"),
            _doc(source_type="video", content="b", video_id="v1", timestamp="01:00"),  # dup
            _doc(source_type="video", content="c", video_id="v1", timestamp="02:00"),
        ]
        result = build_citations(docs)
        assert len(result) == 2

    def test_qa_deduplication_by_question(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(source_type="qa", content="a", question="What?"),
            _doc(source_type="qa", content="b", question="What?"),  # dup
        ]
        result = build_citations(docs)
        assert len(result) == 1

    def test_book_limit_enforced(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content=f"c{i}", book_title=f"B{i}") for i in range(20)]
        result = build_citations(docs, citations_books=3)
        assert len(result) == 3

    def test_video_limit_enforced(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="video", content=f"c{i}", video_id=f"v{i}") for i in range(20)]
        result = build_citations(docs, citations_videos=2)
        assert len(result) == 2

    def test_web_dedup_by_url(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(source_type="web", content="a", url="https://a.com"),
            _doc(source_type="web", content="b", url="https://a.com"),  # dup
        ]
        result = build_citations(docs)
        assert len(result) == 1

    def test_mixed_source_types(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(source_type="book", content="q", book_title="B1"),
            _doc(source_type="video", content="t", video_id="v1"),
            _doc(source_type="qa", content="a", question="Q?"),
            _doc(source_type="web", content="s", url="https://x.com"),
        ]
        result = build_citations(docs)
        types = {c.source_type for c in result}
        assert types == {"book", "video", "qa", "web"}

    def test_custom_builder(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        def custom_builder(doc: RetrievedDoc) -> Citation | None:
            return Citation(source_type="knowledge", title="Custom", content=doc.content)

        docs = [_doc(source_type="knowledge", content="data")]
        result = build_citations(docs, builders={"knowledge": custom_builder})
        assert len(result) == 1
        assert result[0].title == "Custom"

    # ── Config fallback path (survived mutants: ~50) ──────────────

    def test_config_defaults_used_when_no_explicit_limits(self):
        """Calls build_citations WITHOUT explicit limit kwargs.

        This exercises the `if citations_books is None: use cfg.citations_books`
        path that survived mutations because previous tests always passed limits.
        """
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content=f"c{i}", book_title=f"B{i}") for i in range(20)]
        # No explicit citations_books= kwarg → uses cfg.citations_books (default: 10)
        result = build_citations(docs)
        assert len(result) == 10

    def test_config_defaults_for_video_limit(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="video", content=f"c{i}", video_id=f"v{i}") for i in range(20)]
        result = build_citations(docs)
        assert len(result) == 10  # cfg.citations_videos default

    def test_config_defaults_for_qa_limit(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="qa", content=f"a{i}", question=f"Q{i}?") for i in range(20)]
        result = build_citations(docs)
        assert len(result) == 10  # cfg.citations_qa default

    def test_config_defaults_for_web_limit(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [
            _doc(source_type="web", content=f"c{i}", url=f"https://site{i}.com") for i in range(10)
        ]
        result = build_citations(docs)
        assert len(result) == 3  # cfg.citations_web default

    # ── Field fallback paths (survived mutants: ~33) ──────────────

    def test_qa_missing_question_and_answer(self):
        """QA doc with no question/answer fields — fallback paths exercised."""
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="qa", content="just content")]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.question == ""
        assert c.answer == "just content"  # fallback to doc.content
        assert c.title == "Q&A Session"  # fallback title

    def test_qa_empty_answer_falls_back_to_content(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="qa", content="real answer", question="Q?", answer="   ")]
        result = build_citations(docs)
        assert result[0].answer == "real answer"

    def test_video_missing_video_id(self):
        """Video doc with no video_id — fallback paths exercised."""
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="video", content="transcript", title="Lecture")]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.video_id == ""
        assert c.title == "Lecture"  # falls back to doc.title

    def test_video_missing_all_names(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="video", content="transcript")]
        result = build_citations(docs)
        assert result[0].title == "Video"  # final fallback

    def test_book_missing_book_title(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content="quote", title="Doc Title")]
        result = build_citations(docs)
        assert len(result) == 1
        c = result[0]
        assert c.book_title == ""
        assert c.title == "Doc Title"  # fallback to doc.title

    def test_book_no_title_at_all(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content="quote")]
        result = build_citations(docs)
        assert result[0].title == "Unknown Book"  # final fallback

    def test_book_no_quote_falls_back_to_content(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content="the content", book_title="B")]
        result = build_citations(docs)
        assert result[0].quote == "the content"
        assert result[0].content == "the content"

    def test_book_no_keywords_defaults_to_empty_list(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content="c", book_title="B")]
        result = build_citations(docs)
        assert result[0].keywords == []

    def test_video_no_summary_defaults_to_empty(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="video", content="c", video_id="v")]
        result = build_citations(docs)
        assert result[0].summary == ""

    def test_web_uses_snippet_as_fallback(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="web", url="https://x.com", snippet="snippet text")]
        result = build_citations(docs)
        assert result[0].summary == "snippet text"

    def test_web_no_title_falls_back_to_url(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="web", url="https://example.com")]
        result = build_citations(docs)
        assert result[0].title == "https://example.com"

    # ── Zero-limit enforcement ────────────────────────────────────

    def test_zero_book_limit_excludes_books(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content="c", book_title="B")]
        result = build_citations(docs, citations_books=0)
        assert len(result) == 0

    def test_zero_video_limit_excludes_videos(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="video", content="c", video_id="v")]
        result = build_citations(docs, citations_videos=0)
        assert len(result) == 0

    def test_zero_qa_limit_excludes_qa(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="qa", content="c", question="Q")]
        result = build_citations(docs, citations_qa=0)
        assert len(result) == 0

    def test_zero_web_limit_excludes_web(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="web", content="c", url="https://x.com")]
        result = build_citations(docs, citations_web=0)
        assert len(result) == 0

    def test_unknown_source_type_skipped(self):
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [RetrievedDoc(source_type="unknown", content="data", relevance=0.5)]
        result = build_citations(docs)
        assert len(result) == 0

    def test_metadata_defaults_to_empty_dict(self):
        """metadata field defaults to empty dict when not provided."""
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        docs = [_doc(source_type="book", content="c", book_title="B")]
        result = build_citations(docs)
        assert result[0].metadata == {}
