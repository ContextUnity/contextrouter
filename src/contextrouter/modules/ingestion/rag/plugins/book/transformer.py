"""Main Book transformer that orchestrates all book processing components."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from contextrouter.core import Config
from contextrouter.core.types import StructData

from .analyzer import BookAnalyzer
from .extractor import PDFExtractor
from .normalizer import ContentNormalizer

logger = logging.getLogger(__name__)


class BookTransformer:
    """Main book transformer that orchestrates all book processing."""

    def __init__(self, core_cfg: Config) -> None:
        self.core_cfg = core_cfg
        self.extractor = PDFExtractor()
        self.normalizer = ContentNormalizer()
        self.analyzer = BookAnalyzer(core_cfg)

    def transform_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Transform a single PDF into processed content."""
        try:
            # Extract TOC
            toc = self.extractor.extract_toc(pdf_path)
            logger.info(f"Extracted {len(toc)} chapters from TOC")

            # Extract text
            raw_text = self.extractor.extract_text_with_pymupdf4llm(pdf_path)
            logger.info(f"Extracted {len(raw_text)} characters of text")

            # Normalize content
            cleaned_text = self.normalizer.clean_text(raw_text)
            normalized_text = self.normalizer.normalize_unicode(cleaned_text)

            # Split into chunks
            chunks = self.normalizer.split_into_chunks(normalized_text)

            # Analyze chunks
            analyses = self.analyzer.analyze_batch(chunks)

            # Split by chapters if TOC available
            chapter_chunks = self.analyzer.chunk_by_chapters(normalized_text, toc)

            return {
                "text": normalized_text,
                "chapters": toc,
                "chunks": chunks,
                "analyses": analyses,
                "chapter_chunks": chapter_chunks,
                "metadata": {
                    "page_count": self.extractor.get_page_count(pdf_path),
                    "estimated_reading_time": self.analyzer.estimate_reading_time(normalized_text),
                    "chunk_count": len(chunks),
                },
            }

        except Exception as e:
            logger.error("Book transformation failed for %s: %s", pdf_path, e)
            return {
                "text": "",
                "chapters": [],
                "chunks": [],
                "analyses": [],
                "chapter_chunks": [],
                "metadata": {},
                "error": str(e),
            }

    def transform_multiple_pdfs(self, pdf_paths: list[Path]) -> list[dict[str, Any]]:
        """Transform multiple PDFs."""
        results = []

        for pdf_path in pdf_paths:
            logger.info("Processing book: %s", pdf_path.name)
            result = self.transform_pdf(pdf_path)
            # Keep metadata consistent with ingestion schema (reader expects book_title).
            book_title_raw = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
            book_title_raw = re.sub(r"\s+", " ", book_title_raw)
            result["book_title"] = book_title_raw.title() if book_title_raw else pdf_path.stem
            results.append(result)

        return results

    def create_structured_records(
        self, transformation_result: dict[str, Any], taxonomy: StructData | None = None
    ) -> list[dict[str, Any]]:
        """Create structured records from transformation result."""
        records = []

        chunks = transformation_result.get("chunks", [])
        analyses = transformation_result.get("analyses", [])

        for i, (chunk, analysis) in enumerate(zip(chunks, analyses)):
            record = {
                "content": chunk,
                "chunk_id": i,
                "source_type": "book",
                "metadata": {
                    "themes": analysis.get("themes", []),
                    "topics": analysis.get("topics", []),
                    "book_title": transformation_result.get("book_title", ""),
                    **transformation_result.get("metadata", {}),
                },
            }

            # Add taxonomy if available
            if taxonomy:
                record["taxonomy_categories"] = self._map_to_taxonomy(chunk, taxonomy)

            records.append(record)

        return records

    def _map_to_taxonomy(self, content: str, taxonomy: StructData) -> list[str]:
        """Map content to taxonomy categories."""
        # Simple implementation - can be enhanced
        if isinstance(taxonomy, dict):
            categories = taxonomy.get("categories", [])
            content_lower = content.lower()

            matched = []
            for category in categories:
                if isinstance(category, str):
                    category_lower = category.lower()
                    if any(word in content_lower for word in category_lower.split()):
                        matched.append(category)

            return matched

        return []
