"""Book ingestion plugin using PyMuPDF TOC extraction for accurate chapter detection.

Uses PyMuPDF's built-in table of contents (TOC) extraction to identify chapters,
then converts PDF content to markdown with pymupdf4llm. Falls back to markdown
header parsing if TOC is not available.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

# Try to import and activate pymupdf_layout BEFORE importing pymupdf4llm
# This suppresses the warning and improves PDF parsing quality
try:
    import pymupdf.layout

    pymupdf.layout.activate()
except ImportError:
    # pymupdf_layout is optional; continue without it
    pass

# pymupdf4llm will be imported inside functions when needed

# PyMuPDF (fitz) for TOC extraction - pymupdf4llm depends on it
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz  # Alternative import name
    except ImportError:
        fitz = None  # Fallback to markdown parsing if not available

from contextrouter.core import Config

from ..core.plugins import IngestionPlugin
from ..core.prompts import book_batch_analysis_prompt
from ..core.registry import register_plugin
from ..core.types import (
    BookStructData,
    GraphEnrichmentResult,
    IngestionMetadata,
    RawData,
    ShadowRecord,
)
from ..core.utils import (
    clean_str_list,
    get_graph_enrichment,
    llm_generate_tsv,
    load_taxonomy_safe,
    normalize_clean_text,
    parse_tsv_line,
)
from ..settings import RagIngestionConfig
from ..utils.records import generate_id

logger = logging.getLogger(__name__)

# NOTE: pymupdf4llm does NOT reliably emit <page: N> markers. In practice it emits
# separators like: "--- end of page=12 ---". Some parts of the pipeline still
# scan for "page markers" inside text; this pattern supports both formats.
PAGE_MARKER_PATTERN = re.compile(
    r"(?:<page:\s*(\d+)>|<page:(\d+)>|---\s*end\s+of\s+page\s*=\s*(\d+)\s*---)",
    re.IGNORECASE,
)

# Patterns for cleaning PDF artifacts
_PDF_ARTIFACT_PATTERNS_BASE = [
    # Common watermark / footer artifacts
    # OceanofPDF watermark variants:
    # - _OceanofPDF.com_ / OceanofPDF.com
    # - markdown links like [OceanofPDF.com](https://oceanofpdf.com/)
    re.compile(r"(?im)^\s*_*\s*OceanofPDF\.com\s*_*\s*$"),
    re.compile(r"(?i)\[\s*OceanofPDF\.com\s*\]\([^)]+oceanofpdf[^)]*\)"),
    re.compile(r"(?i)\bOceanofPDF\.com\b"),
    # Images placeholders
    re.compile(
        r"\*\*==>\s*picture\b[^\n]*?\bomitted\b[^\n]*?<==\*\*",
        re.IGNORECASE,
    ),
    # Non-content separators / noise lines
    re.compile(r"(?m)^\s*[-_]{3,}\s*$"),
]
_PDF_ARTIFACT_PATTERNS_PAGE_MARKERS = [
    re.compile(r"<page:\s*\d+>", re.IGNORECASE),
    re.compile(r"---\s*end\s+of\s+page[^\n]*---", re.IGNORECASE),
    re.compile(r"---\s*end\s+of\s+page\s*=\s*\d+\s*---", re.IGNORECASE),
    # Variant seen in some converters: "--- end of page.page_number=123 ---"
    re.compile(r"---\s*end\s+of\s+page\.page_number\s*=\s*\d+\s*---", re.IGNORECASE),
]


def clean_book_content(text: str, *, keep_page_markers: bool = False) -> str:
    """Remove PDF artifacts and formatting noise from book content.

    Args:
        text: Raw book content text

    Returns:
        Cleaned text with artifacts removed
    """
    cleaned = text

    # Normalize ambiguous unicode early (smart quotes, dashes, nbsp, etc.)
    from ..core.utils import normalize_ambiguous_unicode

    cleaned = normalize_ambiguous_unicode(cleaned)

    # Remove artifact patterns
    for pattern in _PDF_ARTIFACT_PATTERNS_BASE:
        cleaned = pattern.sub("", cleaned)
    if not keep_page_markers:
        for pattern in _PDF_ARTIFACT_PATTERNS_PAGE_MARKERS:
            cleaned = pattern.sub("", cleaned)

    # Remove URLs (http://, https://, www.) - centralized in one place
    url_pattern = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE
    )
    cleaned = url_pattern.sub("", cleaned)

    # Remove excessive whitespace (including after URL removal)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # Max 2 consecutive newlines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)  # Normalize spaces/tabs
    # Only trim spaces/tabs around newlines (do NOT collapse blank lines)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)  # trailing spaces before newline
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)  # leading spaces after newline

    return cleaned.strip()


@register_plugin("book")
class BookPlugin(IngestionPlugin):
    """Plugin for processing PDF books using PyMuPDF TOC extraction for accurate chapter detection.

    Uses PyMuPDF's built-in table of contents (TOC) extraction to identify chapters,
    then converts PDF content to markdown with pymupdf4llm. Falls back to markdown
    header parsing if TOC is not available.

    Note: Core processing logic has been refactored into components in book/ directory:
    - PDFExtractor: PDF parsing and TOC extraction
    - ContentNormalizer: Text cleaning and normalization
    - BookAnalyzer: Content analysis and chunking
    - BookTransformer: Main orchestration
    """

    @property
    def source_type(self) -> str:
        return "book"

    def load(self, assets_path: str) -> list[RawData]:
        """Load PDF files using PyMuPDF TOC for chapter extraction and pymupdf4llm for content."""
        source_dir = Path(assets_path)
        if not source_dir.exists():
            logger.warning("Book source directory does not exist: %s", assets_path)
            return []

        raw_data: list[RawData] = []

        for pdf_file in source_dir.glob("*.pdf"):
            try:
                logger.info("Processing PDF: %s", pdf_file.name)

                # Extract book title from filename
                # Normalize separators and collapse repeated whitespace
                book_title_raw = pdf_file.stem.replace("_", " ").replace("-", " ").strip()
                book_title_raw = re.sub(r"\s+", " ", book_title_raw)
                book_title = book_title_raw.title()

                # Extract chapters using PyMuPDF TOC (more accurate than markdown headers)
                if fitz is not None:
                    chapters = self._extract_chapters_from_toc(str(pdf_file), book_title)
                else:
                    logger.warning("PyMuPDF (fitz) not available, falling back to markdown parsing")
                    chapters = self._extract_chapters_from_markdown(str(pdf_file), book_title)

                for chapter_info in chapters:
                    metadata: IngestionMetadata = {
                        "book_title": book_title,
                        "chapter": chapter_info.get("title", ""),
                        "page_number": chapter_info.get("start_page", 1),
                    }

                    raw_data.append(
                        RawData(
                            content=chapter_info["content"],
                            source_type="book",
                            metadata=metadata,
                        )
                    )

            except Exception as e:
                logger.error("Failed to process PDF %s: %s", pdf_file, e, exc_info=True)
                continue

        return raw_data

    def _extract_chapters_from_toc(self, pdf_path: str, book_title: str) -> list[dict[str, Any]]:
        """Extract chapters using PyMuPDF's table of contents (TOC) for accurate structure.

        This method:
        1. Extracts TOC from PDF (includes chapter titles and page numbers)
        2. Filters for real chapters (numbered, level 1-2, etc.)
        3. Extracts content between chapter boundaries
        4. Converts to markdown with page markers

        Args:
            pdf_path: Path to PDF file
            book_title: Title of the book

        Returns:
            List of chapter dicts with title, content, and start_page
        """
        chapters: list[dict[str, Any]] = []

        # Open PDF to extract TOC
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        doc.close()

        if not toc:
            logger.warning("No TOC found in PDF %s, falling back to markdown parsing", pdf_path)
            return self._extract_chapters_from_markdown(pdf_path, book_title)

        logger.info("Found %d TOC entries in PDF", len(toc))

        # Filter TOC for real chapters (numbered, level 1-2, exclude front/back matter)
        chapter_toc = self._filter_chapter_toc(toc)

        if not chapter_toc:
            logger.warning("No valid chapters found in TOC, falling back to markdown parsing")
            return self._extract_chapters_from_markdown(pdf_path, book_title)

        logger.info("Extracted %d chapters from TOC", len(chapter_toc))

        # Log what was included/excluded for debugging
        excluded_count = len(toc) - len(chapter_toc)
        if excluded_count > 0:
            logger.info("Excluded %d TOC entries (cover, title, backcover)", excluded_count)

        # Log all chapter titles for verification
        if chapter_toc:
            logger.info("Extracted chapters:")
            for i, (level, title, page) in enumerate(chapter_toc, 1):
                logger.info("  %d. [Level %d] %s (page %d)", i, level, title, page)

        # Group TOC entries by page to merge subchapters
        # Format: "Chapter Title [Subchapter Title]" when both exist on same page
        grouped_toc = self._group_toc_by_page(chapter_toc)

        # Extract content directly from PDF pages (robust, does not depend on markers)
        if fitz is None:
            logger.warning(
                "PyMuPDF (fitz) not available for page extraction; falling back to markdown parsing"
            )
            return self._extract_chapters_from_markdown(pdf_path, book_title)

        doc2 = fitz.open(pdf_path)
        try:
            for i, (merged_title, start_page) in enumerate(grouped_toc):
                # Determine end page (next chapter or end of book)
                end_page = grouped_toc[i + 1][1] if i + 1 < len(grouped_toc) else None

                # Extract content for this chapter using TOC page boundaries
                content = self._extract_pdf_text_between_pages(
                    doc2, start_page=start_page, end_page=end_page
                )

                # Clean chapter title
                clean_title = self._clean_chapter_title(merged_title)

                if not content:
                    logger.warning(
                        "Empty content for chapter '%s' (TOC pages %d-%s).",
                        clean_title,
                        start_page,
                        end_page or "end",
                    )

                chapters.append(
                    {
                        "title": clean_title,
                        "content": content,
                        "start_page": start_page,
                    }
                )

                logger.debug(
                    "Extracted chapter: %s (pages %d-%s, content length: %d)",
                    clean_title,
                    start_page,
                    end_page or "end",
                    len(content),
                )
        finally:
            doc2.close()

        return chapters

    def _filter_chapter_toc(self, toc: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
        """Filter TOC entries to identify real chapters.

        Includes all entries except: cover, title, backcover

        Args:
            toc: List of (level, title, page) tuples from PDF TOC

        Returns:
            Filtered list of chapter TOC entries
        """
        chapter_toc: list[tuple[int, str, int]] = []

        # Keywords to exclude (only cover, title, backcover)
        exclude_keywords = {"cover", "title", "backcover", "back cover"}

        for level, title, page in toc:
            title_lower = title.lower().strip()

            # Skip only: cover, title, backcover
            if any(keyword in title_lower for keyword in exclude_keywords):
                continue

            # Include everything else
            chapter_toc.append((level, title, page))

        return chapter_toc

    def _clean_chapter_title(self, title: str) -> str:
        """Clean chapter title by removing numbering prefixes if needed.

        Args:
            title: Raw chapter title from TOC

        Returns:
            Cleaned title (may keep numbering if it's part of the title)
        """
        # Remove "CHAPTER N" prefix but keep numbered titles like "1. TITLE"
        title = re.sub(r"^CHAPTER\s+\d+\s*[-:]?\s*", "", title, flags=re.IGNORECASE)
        title = title.strip()

        # Normalize unicode in title
        from ..core.utils import normalize_ambiguous_unicode

        title = normalize_ambiguous_unicode(title)

        return title

    def _group_toc_by_page(self, chapter_toc: list[tuple[int, str, int]]) -> list[tuple[str, int]]:
        """Group TOC entries by page number and merge subchapters.

        When multiple entries share the same page, combines them as:
        "Main Chapter [Subchapter]"

        Args:
            chapter_toc: List of (level, title, page) tuples

        Returns:
            List of (merged_title, page) tuples
        """
        grouped: dict[int, list[tuple[int, str]]] = {}  # page -> [(level, title), ...]

        for level, title, page in chapter_toc:
            if page not in grouped:
                grouped[page] = []
            grouped[page].append((level, title))

        # Merge entries on same page
        merged: list[tuple[str, int]] = []
        for page in sorted(grouped.keys()):
            entries = grouped[page]

            if len(entries) == 1:
                # Single entry, use as-is
                merged.append((entries[0][1], page))
            else:
                # Multiple entries on same page - merge them
                # Sort by level (lower level = main chapter)
                entries_sorted = sorted(entries, key=lambda x: x[0])
                main_chapter = entries_sorted[0][1]  # Lowest level = main
                subchapters = [title for level, title in entries_sorted[1:]]  # Higher levels = subs

                if subchapters:
                    # Format: "Main Chapter [Subchapter1] [Subchapter2]"
                    sub_str = "] [".join(subchapters)
                    merged_title = f"{main_chapter} [{sub_str}]"
                else:
                    merged_title = main_chapter

                merged.append((merged_title, page))

        return merged

    def _extract_pdf_text_between_pages(
        self,
        doc: Any,
        *,
        start_page: int,
        end_page: int | None,
    ) -> str:
        """Extract paragraph-ish text from a PDF document between TOC page boundaries.

        Notes:
        - TOC page numbers are 1-based.
        - `fitz` pages are 0-based indices.
        """
        page_count = getattr(doc, "page_count", 0) or 0
        if page_count <= 0:
            return ""

        start_idx = max(0, start_page - 1)
        end_idx = (end_page - 1) if end_page is not None else page_count
        end_idx = max(start_idx, min(end_idx, page_count))

        parts: list[str] = []
        for page_index in range(start_idx, end_idx):
            try:
                page = doc.load_page(page_index)
                # Use blocks to preserve paragraph-like structure (better for chunking).
                # Each block is (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks") or []
                block_texts: list[str] = []
                for b in sorted(blocks, key=lambda x: (x[1], x[0])):  # top-to-bottom, left-to-right
                    txt = b[4] if len(b) > 4 else ""
                    if not isinstance(txt, str):
                        continue
                    t = txt.strip()
                    if not t:
                        continue
                    # De-hyphenate wrapped words
                    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
                    # Join soft line breaks inside a block
                    t = re.sub(r"[ \t]*\n[ \t]*", " ", t)
                    block_texts.append(t.strip())
                # Keep as paragraphs (double-newline between blocks)
                txt = "\n\n".join(block_texts)
            except Exception:
                txt = ""
            if isinstance(txt, str) and txt.strip():
                # Inject an explicit page marker BEFORE EACH paragraph block so page_end can be
                # computed even when chunks start mid-page. Markers are removed from final chunk text.
                page_num = page_index + 1  # back to 1-based
                page_paras = [p for p in txt.split("\n\n") if p.strip()]
                marked_paras = [f"<page:{page_num}>\n{p.strip()}" for p in page_paras]
                parts.append("\n\n".join(marked_paras))

        content = "\n\n".join(parts).strip()
        if content:
            # Keep page markers for pagination downstream; strip them later when saving chunks.
            content = clean_book_content(content, keep_page_markers=True)
        return content

    def _extract_chapters_from_markdown(
        self, pdf_path: str, book_title: str
    ) -> list[dict[str, Any]]:
        """Fallback: Extract chapters from markdown headers (original method).

        Used when TOC is not available or empty.
        """
        try:
            import pymupdf4llm

            # Convert PDF to Markdown using pymupdf4llm with layout analysis
            markdown_content = pymupdf4llm.to_markdown(
                pdf_path,
                write_images=False,
                page_separators=True,  # Adds <page: N> markers
            )
        except ImportError:
            logger.warning("pymupdf4llm not available, falling back to basic text extraction")
            # Fallback to basic text extraction if pymupdf4llm is not available
            markdown_content = self._extract_text_basic(pdf_path)

        # Clean markdown: normalize headers, filter testimonials, normalize unicode
        from ..core.utils import (
            clean_markdown_headers,
            filter_testimonial_signatures,
            normalize_ambiguous_unicode,
        )

        markdown_content = normalize_ambiguous_unicode(markdown_content)
        markdown_content = clean_markdown_headers(markdown_content)
        markdown_content = filter_testimonial_signatures(markdown_content)

        # Parse markdown to extract chapters (by highest-level headers)
        chapters = self._extract_chapters_with_pages(markdown_content)

        return chapters

    def _extract_chapters_with_pages(self, markdown: str) -> list[dict[str, Any]]:
        """Extract chapters from markdown content, splitting by highest-level headers.

        Also parses page markers (<page: N>) to track page numbers accurately.

        Args:
            markdown: Markdown content with page markers

        Returns:
            List of chapter dicts with title, content, and start_page
        """
        chapters = []
        lines = markdown.split("\n")
        current_chapter: dict[str, Any] = {
            "title": "Introduction",
            "content": "",
            "start_page": 1,
        }
        current_page = 1
        highest_header_level = None  # Track the highest-level header we've seen

        for line in lines:
            # Check for page markers first
            page_match = PAGE_MARKER_PATTERN.search(line)
            if page_match:
                page_num = page_match.group(1) or page_match.group(2) or page_match.group(3)
                current_page = int(page_num) if page_num else current_page
                # Don't include page markers in content
                continue

            # Check for headers (#, ##, ###, etc.)
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                header_level = len(header_match.group(1))
                header_title = header_match.group(2).strip()

                # Filter out testimonial signatures (lines starting with -- or — at start of line or after header)
                # These are not real chapters but testimonial quotes
                if (
                    header_title.startswith("--")
                    or header_title.startswith("—")
                    or header_title.startswith("- ")
                ):
                    # This is likely a testimonial signature, not a real chapter header
                    # Include it in content instead
                    current_chapter["content"] += line + "\n"
                    continue

                # Filter out sentence-like "headers" that are not real chapter titles
                # Criteria: too short, ends with period, starts with lowercase, or no capitalization
                if self._is_sentence_like(header_title):
                    # Treat as content, not a chapter header
                    current_chapter["content"] += line + "\n"
                    continue

                # Determine highest header level on first header
                if highest_header_level is None:
                    highest_header_level = header_level

                # Split on highest-level headers only (usually chapters)
                if header_level == highest_header_level:
                    # Save previous chapter
                    if current_chapter["content"].strip():
                        chapters.append(current_chapter.copy())

                    # Start new chapter - strip markdown from title
                    from ..core.utils import strip_markdown_from_text

                    clean_title = strip_markdown_from_text(header_title)
                    current_chapter = {
                        "title": clean_title,
                        "content": "",
                        "start_page": current_page,
                    }
                else:
                    # Sub-headers: include in content
                    current_chapter["content"] += line + "\n"
            else:
                # Regular content line
                current_chapter["content"] += line + "\n"

        # Add last chapter
        if current_chapter["content"].strip():
            chapters.append(current_chapter)

        logger.info("Extracted %d chapters from markdown", len(chapters))
        return chapters

    @staticmethod
    def _is_sentence_like(text: str) -> bool:
        """Check if text looks like a sentence fragment rather than a chapter title.

        Filters out false positives where PDF conversion creates markdown headers
        from regular sentences or sentence fragments.

        Args:
            text: Header text to check

        Returns:
            True if text appears to be a sentence fragment, False if it's a valid title
        """
        if not text or len(text) < 10:
            return True

        # Ends with period - likely a sentence, not a title
        if text.rstrip().endswith("."):
            return True

        # Starts with lowercase word - not a proper title
        first_word = text.split()[0] if text.split() else ""
        if first_word and first_word[0].islower():
            return True

        # Too long for a title (likely a sentence)
        if len(text) > 100:
            return True

        # No uppercase letters in first 20 chars - probably not a title
        first_part = text[:20]
        if first_part and not any(c.isupper() for c in first_part if c.isalpha()):
            return True

        return False

    def transform(
        self,
        data: list[RawData],
        enrichment_func: Callable[[str], GraphEnrichmentResult],
        taxonomy_path: Path | None = None,
        config: RagIngestionConfig | None = None,
        core_cfg: Config | None = None,
        **kwargs: object,
    ) -> list[ShadowRecord]:
        """Transform book data by chunking within chapters (by paragraphs, avoid mid-sentence breaks)."""
        _ = enrichment_func, kwargs
        # Load taxonomy for keyword extraction
        taxonomy = load_taxonomy_safe(taxonomy_path) if taxonomy_path else None

        # Read LLM topic extraction settings
        if config is None:
            from ..config import load_config

            config = load_config()
        llm_topic_extraction_enabled = config.book.llm_topic_extraction_enabled
        if llm_topic_extraction_enabled and core_cfg is None:
            raise ValueError(
                "BookPlugin.transform requires core_cfg when llm_topic_extraction_enabled=true"
            )

        # Get taxonomy categories for LLM prompt
        taxonomy_categories = None
        if taxonomy and isinstance(taxonomy.get("categories"), dict):
            taxonomy_categories = [
                cat_name.replace("_", " ").title()
                for cat_name in taxonomy.get("categories", {}).keys()
            ]

        shadow_records: list[ShadowRecord] = []

        for raw in data:
            book_title = raw.metadata.get("book_title", "Book")
            chapter = raw.metadata.get("chapter", "")
            page_number = raw.metadata.get("page_number", 1)

            # Clean content before processing but KEEP page markers so we can compute page_end.
            cleaned_content = clean_book_content(raw.content, keep_page_markers=True)

            # Split by paragraphs (double newline)
            paragraphs = [p.strip() for p in cleaned_content.split("\n\n") if p.strip()]

            # Chunk paragraphs, ensuring no mid-sentence breaks
            # Use chapter's start_page as base (page markers already extracted in load phase)
            chunks = self._chunk_paragraphs(
                paragraphs, target_size=1000, min_size=400, base_page=page_number
            )

            # Batch LLM topic extraction if enabled
            chunk_topics: dict[int, dict[str, str]] = {}
            if llm_topic_extraction_enabled and chunks:
                logger.info(
                    "book: analyzing %d chunks from %s - %s with LLM...",
                    len(chunks),
                    book_title,
                    chapter,
                )
                chunk_topics = self._analyze_chunks_batch(
                    chunks, taxonomy_categories, core_cfg=core_cfg
                )

            for chunk_idx, chunk_info in enumerate(chunks):
                chunk_text = chunk_info["text"]
                chunk_start_page = chunk_info["start_page"]
                chunk_end_page = chunk_info.get(
                    "end_page", chunk_start_page
                )  # Fallback to start if not set

                # Clean chunk text again to catch any missed artifacts (drop page markers here).
                chunk_text = clean_book_content(chunk_text, keep_page_markers=False)

                # Get topic from LLM analysis if available
                topic_info = chunk_topics.get(chunk_idx, {})
                topic = topic_info.get("topic", "")

                # Enrichment (graph + taxonomy)
                graph_keywords, summary, parent_categories = get_graph_enrichment(
                    text=chunk_text, enrichment_func=enrichment_func
                )

                # Extract taxonomy keywords
                taxonomy_keywords = self._extract_taxonomy_terms(chunk_text, taxonomy)

                # Combine metadata + graph + taxonomy keywords, deduplicated
                initial_keywords = raw.metadata.get("keywords", [])
                if not isinstance(initial_keywords, list):
                    initial_keywords = []
                keywords = list(
                    dict.fromkeys([*initial_keywords, *graph_keywords, *taxonomy_keywords])
                )[:10]

                # Build input_text with QA-style explicit enrichment format (include topic if available)
                input_text = self._build_input_text(
                    content=chunk_text,
                    keywords=keywords,
                    summary=summary,
                    parent_categories=parent_categories,
                    topic=topic,
                )

                # Clean quote text: replace newlines with spaces for better UI display
                # (unicode already normalized during preprocess)
                clean_quote = normalize_clean_text(chunk_text)
                # Chapter title already normalized during preprocess
                clean_chapter = chapter

                record_id = generate_id(book_title, chapter, str(chunk_start_page), chunk_text[:50])

                struct_data: BookStructData = {
                    "source_type": "book",
                    # snake_case for retriever/citations compatibility
                    "book_title": book_title,
                    "chapter": clean_chapter,
                    "page_start": chunk_start_page,
                    "page_end": (
                        chunk_end_page if chunk_end_page != chunk_start_page else None
                    ),  # Only set if different
                    # optional: small keyword list for UI/debugging
                    "keywords": clean_str_list(keywords, limit=10),
                    "quote": clean_quote,
                }

                shadow_records.append(
                    ShadowRecord(
                        id=record_id,
                        input_text=input_text,
                        struct_data=struct_data,
                        title=f"{book_title} - {chapter}",
                        source_type="book",
                    )
                )

        return shadow_records

    def _build_input_text(
        self,
        content: str,
        keywords: list[str],
        summary: str,
        parent_categories: list[str] | None = None,
        topic: str | None = None,
    ) -> str:
        """Build shadow context input_text with QA-style explicit enrichment format.

        Args:
            content: Book chunk content
            keywords: Graph enrichment keywords
            summary: Graph enrichment summary (relation descriptions)
            parent_categories: Taxonomy categories from graph enrichment
            topic: Optional LLM-extracted topic (5-8 word summary)

        Returns:
            Formatted input_text string with explicit Categories: and Additional Knowledge: headers
        """
        parts = []

        # Add topic if available (QA-style)
        if topic:
            parts.append(f"Topic: {topic}")

        parts.append(content)

        # Add taxonomy categories from graph enrichment (QA-style)
        if parent_categories:
            cats = [c for c in parent_categories if isinstance(c, str) and c.strip()]
            if cats:
                cat_str = ", ".join(cats[:5])
                parts.append(f"Categories: {cat_str}")

        # Add natural language enrichment for keywords (QA-style)
        if keywords:
            top_keywords = keywords[:10]
            if len(top_keywords) == 1:
                parts.append(f"Additional Knowledge: This text is related to {top_keywords[0]}.")
            elif len(top_keywords) <= 3:
                keywords_str = ", ".join(top_keywords[:-1])
                parts.append(
                    f"Additional Knowledge: This text is related to {keywords_str} and {top_keywords[-1]}."
                )
            else:
                keywords_str = ", ".join(top_keywords[:5])
                parts.append(
                    f"Additional Knowledge: This text is related to {keywords_str}, and other concepts."
                )

        # Add summary if available (graph relations)
        if isinstance(summary, str) and summary.strip():
            parts.append(f"Additional Knowledge: {summary.strip()}")

        return "\n".join(parts)

    def _analyze_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
        taxonomy_categories: list[str] | None = None,
        *,
        core_cfg: Config | None,
    ) -> dict[int, dict[str, str]]:
        """Batch LLM processing to identify topic and category for each chunk.

        Args:
            chunks: List of chunk dictionaries with "text" key
            taxonomy_categories: Optional list of taxonomy category names

        Returns dict mapping chunk index to analysis results.
        """
        if not chunks:
            return {}

        # Prepare batch prompt
        batch_items = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            # Truncate to avoid token limits (keep first 2000 chars)
            truncated = chunk_text[:2000] + ("..." if len(chunk_text) > 2000 else "")
            batch_items.append(
                {
                    "index": i,
                    "text": truncated,
                }
            )

        if core_cfg is None:
            raise ValueError("BookPlugin._analyze_chunks_batch requires core_cfg")

        try:
            # Build batch prompt
            items_text = "\n\n---\n\n".join(
                f"CHUNK {item['index']}:\n{item['text']}" for item in batch_items
            )
            prompt = book_batch_analysis_prompt(
                items_text=items_text,
                taxonomy_categories=taxonomy_categories,
            )

            text = llm_generate_tsv(
                core_cfg=core_cfg,
                prompt=prompt,
                model=core_cfg.models.ingestion.taxonomy.model,
                max_tokens=16384,
                temperature=0.1,
                retries=3,
            )

            # Parse TSV: index<TAB>topic<TAB>category
            analyses: dict[int, dict[str, str]] = {}
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = parse_tsv_line(ln)
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0].strip())
                except ValueError:
                    continue
                if 0 <= idx < len(chunks):
                    analyses[idx] = {
                        "topic": parts[1].strip() or "",
                        "category": parts[2].strip() if len(parts) > 2 else "",
                    }

            # Fill in missing analyses with fallback
            for i in range(len(chunks)):
                if i not in analyses:
                    analyses[i] = {
                        "topic": "",
                        "category": "",
                    }

            return analyses

        except Exception as e:
            logger.warning("LLM batch analysis failed: %s, skipping topic extraction", e)
            return {}

    def _extract_taxonomy_terms(
        self,
        content: str,
        taxonomy: dict[str, Any] | None,
    ) -> list[str]:
        """Extract matching taxonomy terms from content.

        Simple case-insensitive keyword matching.
        Returns matched terms sorted by frequency.
        """
        if not taxonomy:
            return []

        all_keywords = taxonomy.get("all_keywords", [])
        if not all_keywords:
            return []

        content_lower = content.lower()
        matches: list[tuple[str, int]] = []

        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            if count > 0:
                matches.append((keyword, count))

        # Sort by frequency, return top 10
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:10]]

    def _chunk_paragraphs(
        self,
        paragraphs: list[str],
        target_size: int = 1000,
        min_size: int = 400,
        base_page: int = 1,
    ) -> list[dict[str, Any]]:
        """Chunk paragraphs together, ensuring no mid-sentence breaks.

        Since we chunk by paragraphs (which are complete units), we naturally avoid
        mid-sentence breaks. Each paragraph is already a complete thought.

        Also tracks page markers (<page: N>) within paragraphs to calculate page_end.

        Args:
            paragraphs: List of paragraph strings
            target_size: Target chunk size in characters
            min_size: Minimum chunk size before breaking
            base_page: Base page number for this chapter (used as starting point)

        Returns:
            List of chunk dicts with text, start_page, and end_page
        """
        chunks = []
        current_chunk_parts: list[str] = []
        current_size = 0
        current_start_page = base_page
        current_end_page = base_page

        for para in paragraphs:
            para_size = len(para)

            # Extract page markers from this paragraph
            page_matches = list(PAGE_MARKER_PATTERN.finditer(para))
            if page_matches:
                # Use the last page marker in this paragraph
                last = (
                    page_matches[-1].group(1)
                    or page_matches[-1].group(2)
                    or page_matches[-1].group(3)
                )
                last_page = int(last) if last else current_end_page
                current_end_page = last_page

            # Check if adding this paragraph would exceed target size
            if current_size + para_size > target_size and current_size >= min_size:
                # Finalize current chunk
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append(
                    {
                        "text": chunk_text,
                        "start_page": current_start_page,
                        "end_page": current_end_page,
                    }
                )

                # Start new chunk - next chunk starts where this one ended
                current_chunk_parts = [para]
                current_size = para_size
                current_start_page = current_end_page
            else:
                # Add to current chunk
                current_chunk_parts.append(para)
                current_size += para_size + 2  # +2 for "\n\n"

        # Add final chunk
        if current_chunk_parts:
            chunk_text = "\n\n".join(current_chunk_parts)
            chunks.append(
                {
                    "text": chunk_text,
                    "start_page": current_start_page,
                    "end_page": current_end_page,
                }
            )

        return chunks
