"""Content normalization component for Book plugin."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class ContentNormalizer:
    """Handles text cleaning and normalization for book content."""

    def __init__(self) -> None:
        # Common patterns to clean
        self._header_patterns = [
            r"^Chapter\s+\d+.*$",  # Chapter headers
            r"^Page\s+\d+.*$",  # Page numbers
            r"^\d+\s*$",  # Standalone numbers
        ]

        self._footer_patterns = [
            r".*Page\s+\d+$",  # Page footers
            r".*\d{4}.*",  # Copyright years at end
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Split into lines for processing
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip headers/footers
            if self._is_header_footer(line):
                continue

            # Clean the line
            line = self._clean_line(line)

            if line:  # Only add non-empty lines
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _is_header_footer(self, line: str) -> bool:
        """Check if line is a header or footer to be removed."""
        # Check header patterns
        for pattern in self._header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        # Check footer patterns
        for pattern in self._footer_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    def _clean_line(self, line: str) -> str:
        """Clean individual line."""
        # Remove extra whitespace
        line = re.sub(r"\s+", " ", line)

        # Remove page numbers in brackets [123]
        line = re.sub(r"\[\d+\]", "", line)

        # Remove standalone numbers at start of line
        line = re.sub(r"^\d+\s+", "", line)

        # Clean up punctuation
        line = line.strip(" .,-")

        return line.strip()

    def normalize_unicode(self, text: str) -> str:
        """Normalize problematic Unicode characters."""
        # Common PDF extraction issues
        replacements = {
            "ﬁ": "fi",  # fi ligature
            "ﬂ": "fl",  # fl ligature
            "ﬀ": "ff",  # ff ligature
            "ﬃ": "ffi",  # ffi ligature
            "ﬄ": "ffl",  # ffl ligature
            """: "'",  # smart quotes
            """: "'",
            '"': '"',  # smart double quotes
            "…": "...",  # ellipsis
            "–": "-",  # en dash
            "—": "-",  # em dash
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def split_into_chunks(self, text: str, max_chunk_size: int = 2000) -> list[str]:
        """Split text into manageable chunks."""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
