"""PIIUnmasker — unmask PII tokens back to real values.
Server-side only: real values never leave the server.
The LLM response contains tokens like DOC_7f3a which are replaced
with actual values before showing to the user.
"""

from __future__ import annotations

import re
from typing import final

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list

from contextunity.router.cortex.privacy.masking.store import TOKEN_PATTERN, MappingStore

logger = get_contextunit_logger(__name__)


def _unmask_object(store: MappingStore, obj: object) -> object:
    """Recursively unmask strings inside dict/list containers."""
    if isinstance(obj, str):
        return store.resolve_all_tokens(obj)
    if is_object_dict(obj):
        return {str(key): _unmask_object(store, value) for key, value in obj.items()}
    if is_object_list(obj):
        return [_unmask_object(store, item) for item in obj]
    return obj


@final
class PIIUnmasker:
    """Unmask PII tokens in LLM responses.

    Args:
        store: MappingStore containing the token→value mappings.
    """

    def __init__(self, store: MappingStore) -> None:
        """Create a new PII unmasker.

        Pre-compiles the ``TOKEN_PATTERN`` regex for efficient scanning.

        Args:
            store: The mapping store holding token ↔ real-value mappings
                for the current session.
        """
        self._store = store
        self._token_re = re.compile(TOKEN_PATTERN)

    def unmask_text(self, text: str) -> str:
        """Find and replace all PII tokens in text with real values.

        Tokens not found in the store are left as-is.

        Args:
            text: Text containing PII tokens.

        Returns:
            Text with tokens replaced by real values.
        """
        if not text:
            return text
        return self._store.resolve_all_tokens(text)

    def unmask_dict(self, data: dict[str, object]) -> dict[str, object]:
        """Recursively unmask PII tokens in a dict.

        Handles nested dicts and lists.

        Args:
            data: Dict potentially containing PII tokens in string values.

        Returns:
            New dict with tokens replaced by real values.
        """
        result = _unmask_object(self._store, data)
        return result if is_object_dict(result) else data

    def unmask_list(self, data: list[object]) -> list[object]:
        """Recursively unmask PII tokens in a list structure.

        Handles nested dicts, lists, and plain strings at any depth.

        Args:
            data: List potentially containing token strings or nested
                structures with token strings.

        Returns:
            New list with all recognized tokens replaced by real values.
        """
        result = _unmask_object(self._store, data)
        return list(result) if is_object_list(result) else data

    def has_tokens(self, text: str) -> bool:
        """Check whether text contains any PII masking tokens.

        Args:
            text: The text to scan.

        Returns:
            ``True`` if at least one ``TOKEN_PATTERN`` match is found.
        """
        return bool(self._token_re.search(text))
