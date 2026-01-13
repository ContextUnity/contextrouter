"""Stage 6: JSONL -> Cloud -> Search Index.

Uses modular upload providers configured in settings.toml [upload] section.
Default provider: gcloud (GCS + Vertex AI Search).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..settings import RagIngestionConfig
from ..upload_providers import UploadResult, get_provider

logger = logging.getLogger(__name__)


def deploy_jsonl_files(
    *,
    jsonl_paths_by_type: dict[str, str],
    config: RagIngestionConfig,
    wait: bool,
) -> dict[str, UploadResult]:
    """Deploy JSONL files using configured upload provider.

    Args:
        jsonl_paths_by_type: Mapping of type -> JSONL file path
        config: Full ingestion config (from load_config())
        wait: If True, wait for indexing to complete

    Returns:
        Mapping of type -> UploadResult
    """
    provider = get_provider(config)
    summary = provider.get_config_summary()
    logger.info("Using upload provider: %s", summary)

    results: dict[str, UploadResult] = {}

    for t, p in jsonl_paths_by_type.items():
        local_path = Path(p)
        logger.info("deploy: type=%s path=%s wait=%s", t, local_path, wait)
        result = provider.upload_and_index(local_path, wait=wait)

        if result.success:
            logger.info("deploy: type=%s succeeded - %s", t, result.details)
        else:
            logger.error("deploy: type=%s failed - %s", t, result.error)

        results[t] = result

    return results
