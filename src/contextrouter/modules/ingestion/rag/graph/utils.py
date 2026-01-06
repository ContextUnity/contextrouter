"""Graph building utilities."""

from __future__ import annotations

import re

# Constants
EXAMPLE_ENTITIES = {"entity1", "entity2", "entity3", "entity4", "entity5"}
INVALID_LABELS = {
    "RELATION_TYPE",
    "RELATION",
    "RELATIONSHIP",
    "CONNECTED",
    "CONNECTS",
    "LINKED",
    "LINKED_TO",
}
LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
GENERIC_NODES = {
    "implementation",
    "action",
    "process",
    "method",
    "system",
    "approach",
    "way",
    "means",
    "thing",
    "element",
    "factor",
}

# Label normalization map
LABEL_NORMALIZATIONS = {
    "IS_AN": "IS_A",
    "HAS_A": "IS_A",
    "CONTAINS": "INCLUDES",
}


def clean_entity(entity: str) -> str | None:
    """Clean entity name, removing OCR artifacts, whitespace, and special chars."""
    if not entity:
        return None

    s = str(entity)
    # Remove ALL control characters (including single \r, \n, \t)
    s = re.sub(r"[\r\n\t]", "", s)
    # Remove other control chars (0x00-0x1F, 0x7F-0x9F)
    s = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", s)
    s = s.strip()
    s = s.lstrip("-•").strip()
    s = s.strip("`\"'''")
    s = re.sub(r"\s+", " ", s).strip()

    if not s or s.lower() in EXAMPLE_ENTITIES:
        return None
    if len(s) < 2 or len(s) > 120:
        return None

    return s


def clean_label(label: str, allowed_labels: set[str]) -> str | None:
    """Clean and validate relation label. Returns None if invalid."""
    if not label:
        return None

    s = str(label).strip().upper()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Z0-9_]", "", s)

    if not s or s in INVALID_LABELS:
        return None
    if not LABEL_RE.match(s):
        return None
    if s.count("_") > 8 or len(s) > 64:
        return None

    # Normalize common variations
    s = LABEL_NORMALIZATIONS.get(s, s)

    return s if s in allowed_labels else None


def normalize_article(text: str) -> str:
    """Remove leading articles from text."""
    return re.sub(r"^(the|a|an)\s+", "", text.lower(), flags=re.IGNORECASE).strip()


def extract_name_key(name: str) -> str | None:
    """Extract name key for grouping (first name + initial or first name)."""
    words = name.lower().split()
    if len(words) < 2:
        return None

    first_name = words[0]
    if len(words) >= 2 and len(words[1]) <= 2:  # Initial like "B" or "B."
        return f"{first_name} {words[1]}"
    return first_name


def is_name_like(text: str) -> bool:
    """Check if text looks like a name (multi-word)."""
    words = text.split()
    return len(words) >= 2


def find_best_canonical(
    entity: str,
    candidates: list[tuple[int, str]],
    max_diff: int = 10,
) -> str | None:
    """Find best canonical form from candidates (prefer longest, within max_diff)."""
    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))  # Longest first, then alphabetical
    best_len, best_canon = candidates[0]

    if abs(best_len - len(entity.lower())) <= max_diff:
        return best_canon
    return None
