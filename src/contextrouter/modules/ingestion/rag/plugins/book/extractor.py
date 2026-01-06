"""PDF extraction component for Book plugin."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Handles PDF parsing and table of contents extraction."""

    def __init__(self) -> None:
        self._fitz_available = self._check_fitz_availability()

    def _check_fitz_availability(self) -> bool:
        """Check if PyMuPDF (fitz) is available."""
        try:
            import fitz

            return True
        except ImportError:
            try:
                import pymupdf as fitz  # noqa: F401 - dynamic import for availability check

                return True
            except ImportError:
                logger.warning("PyMuPDF not available, using markdown fallback")
                return False

    def extract_toc(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Extract table of contents from PDF."""
        if not self._fitz_available:
            return []

        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            toc = doc.get_toc()

            # Convert to our format
            chapters = []
            for level, title, page in toc:
                chapters.append(
                    {
                        "level": level,
                        "title": title,
                        "page": page,
                        "start_page": page,
                    }
                )

            doc.close()
            return chapters

        except Exception as e:
            logger.warning("TOC extraction failed for %s: %s", pdf_path, e)
            return []

    def extract_text_with_pymupdf4llm(self, pdf_path: Path) -> str:
        """Extract text using pymupdf4llm for better formatting."""
        try:
            import pymupdf4llm

            # Extract with layout preservation
            text = pymupdf4llm.to_markdown(str(pdf_path))
            return text

        except ImportError:
            logger.warning("pymupdf4llm not available, using basic extraction")
            return self._extract_text_basic(pdf_path)
        except Exception as e:
            logger.error("PDF extraction failed for %s: %s", pdf_path, e)
            return ""

    def _extract_text_basic(self, pdf_path: Path) -> str:
        """Basic text extraction fallback."""
        if not self._fitz_available:
            return ""

        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            text = ""

            for page in doc:
                text += page.get_text() + "\n"

            doc.close()
            return text

        except Exception as e:
            logger.error("Basic PDF extraction failed: %s", e)
            return ""

    def get_page_count(self, pdf_path: Path) -> int:
        """Get total page count of PDF."""
        if not self._fitz_available:
            return 0

        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0
