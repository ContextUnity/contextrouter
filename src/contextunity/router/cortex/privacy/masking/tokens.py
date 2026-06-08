"""Token generation strategies.
Tokens are designed to be:
1. Consistent: same input + same session → same token
2. Unique: different inputs → different tokens
3. Minimal metadata leakage: LLM can't determine count or order
4. LLM-readable: prefix indicates entity type (DOC vs PAT)
Format: {PREFIX}_{suffix}
Regex:  ^[A-Z]{2,5}_[a-f0-9]{4,12}$
"""

from __future__ import annotations

import secrets
from typing import Literal, final


@final
class TokenGenerator:
    """Generate PII replacement tokens with configurable style."""

    def __init__(self, style: Literal["random_hex", "uuid", "sequential"] = "random_hex") -> None:
        """Create a token generator with the specified suffix strategy.

        Args:
            style: Suffix generation strategy. ``"random_hex"`` produces
                4-char cryptographic hex (65 536 combinations); ``"uuid"``
                produces 12-char hex; ``"sequential"`` uses monotonic
                per-prefix counters (deterministic, testing only).
        """
        self._style = style
        self._counters: dict[str, int] = {}

    def generate(self, prefix: str) -> str:
        """Generate a new token with the given prefix.

        Args:
            prefix: Entity type prefix (e.g. "DOC", "PAT").

        Returns:
            Token string like "DOC_7f3a" or "PAT_b2c1e4f8".
        """
        match self._style:
            case "random_hex":
                suffix = secrets.token_hex(2)  # 4 chars, 65536 combos
                return f"{prefix}_{suffix}"
            case "uuid":
                suffix = secrets.token_hex(6)  # 12 chars
                return f"{prefix}_{suffix}"
            case "sequential":
                self._counters[prefix] = self._counters.get(prefix, 0) + 1
                return f"{prefix}_{self._counters[prefix]:04d}"
            case _:
                suffix = secrets.token_hex(2)
                return f"{prefix}_{suffix}"
