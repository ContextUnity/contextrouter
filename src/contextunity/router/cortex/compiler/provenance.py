"""Provenance HMAC for compiled graphs.

- Each compiled graph gets an HMAC signature computed over its
structural identity (graph_name, node_names, tenant_id). Verified
at execution time to detect graph tampering in the registry.

Per-graph granularity — one HMAC per compile, one verify per execution.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence


def compute_provenance_hmac(
    *,
    graph_name: str,
    node_names: Sequence[str],
    tenant_id: str,
    signing_key: str,
) -> str:
    """Compute HMAC over graph structural identity.

    Node names are sorted before hashing — order doesn't matter.

    Returns:
        Hex-encoded HMAC string.
    """
    # Canonical message: sorted node names joined with |
    sorted_names = sorted(node_names)
    message = f"{graph_name}:{','.join(sorted_names)}:{tenant_id}"

    return hmac.new(
        signing_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_provenance_hmac(
    expected_hmac: str,
    *,
    graph_name: str,
    node_names: Sequence[str],
    tenant_id: str,
    signing_key: str,
) -> bool:
    """Verify HMAC against graph structural identity.

    Uses constant-time comparison to prevent timing attacks.
    """
    computed = compute_provenance_hmac(
        graph_name=graph_name,
        node_names=node_names,
        tenant_id=tenant_id,
        signing_key=signing_key,
    )
    return hmac.compare_digest(computed, expected_hmac)


__all__ = ["compute_provenance_hmac", "verify_provenance_hmac"]
