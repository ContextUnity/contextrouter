"""OpenAI Batch API provider for async batch processing.
This module provides 50% cost savings for large-scale async operations.
Batches complete within 24 hours (often faster during low-load periods).
Use cases:
- Commerce harvester enrichment (thousands of products)
- Bulk content generation
- Large-scale data labeling/extraction
NOT for real-time operations - use regular OpenAI LLM for that.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps, json_loads
from contextunity.core.sdk.payload import get_json_dict_list, get_str
from contextunity.core.types import JsonDict, is_json_dict

from contextunity.router.core import RouterConfig
from contextunity.router.core.exceptions import RouterLLMError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = get_contextunit_logger(__name__)


@dataclass
class BatchRequest:
    """Single request in a batch job."""

    custom_id: str
    messages: list[JsonDict]
    model: str = "gpt-4o-mini"
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, str] | None = None


@dataclass
class BatchResult:
    """Result from a completed batch job."""

    custom_id: str
    content: str
    usage: dict[str, int] | None = None
    error: str | None = None


@dataclass
class BatchJob:
    """Batch job status."""

    id: str
    status: str  # validating, in_progress, completed, failed, expired, cancelled
    created_at: int
    completed_at: int | None = None
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    request_counts: dict[str, int] = field(default_factory=dict)


class OpenAIBatchClient:
    """Client for OpenAI Batch API.

    Usage:
        client = OpenAIBatchClient(config)

        # Create batch
        requests = [
            BatchRequest(custom_id="item-1", messages=[...]),
            BatchRequest(custom_id="item-2", messages=[...]),
        ]
        job = await client.create_batch(requests)

        # Poll for completion (or use webhook)
        while job.status not in ("completed", "failed"):
            job = await client.get_batch(job.id)
            await asyncio.sleep(60)

        # Get results
        results = await client.get_batch_results(job)
    """

    def __init__(self, config: RouterConfig) -> None:
        """Create an ``AsyncOpenAI`` client configured for the batch file-upload API."""
        self._config: RouterConfig = config
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazily initialize and return the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI Batch API requires `openai` package. Install with: pip install openai"
                ) from e

            self._client = AsyncOpenAI(
                api_key=self._config.openai.api_key,
                organization=self._config.openai.organization,
            )
        return self._client

    async def create_batch(
        self,
        requests: list[BatchRequest],
        *,
        description: str | None = None,
    ) -> BatchJob:
        """Create a new batch job."""
        client = self._get_client()

        # Create JSONL content
        jsonl_lines: list[str] = []
        for req in requests:
            body: dict[str, object] = {
                "model": req.model,
                "messages": req.messages,
            }
            if req.temperature is not None:
                body["temperature"] = req.temperature
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens
            if req.response_format is not None:
                body["response_format"] = req.response_format

            line: dict[str, object] = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            jsonl_lines.append(json_dumps(line))

        jsonl_content = "\n".join(jsonl_lines)

        # Upload file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _ = f.write(jsonl_content)
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                file_response = await client.files.create(
                    file=f,
                    purpose="batch",
                )
            input_file_id = file_response.id
        finally:
            temp_path.unlink()

        logger.info("Uploaded batch input file: %s", input_file_id)

        # Create batch job
        batch_response = await client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description or "contextunity.router batch job"},
        )

        logger.info("Created batch job: %s", batch_response.id)

        return BatchJob(
            id=batch_response.id,
            status=batch_response.status,
            created_at=batch_response.created_at,
            input_file_id=input_file_id,
            request_counts=dict(batch_response.request_counts or {}),
        )

    async def get_batch(self, batch_id: str) -> BatchJob:
        """Get batch job status."""
        client = self._get_client()
        batch = await client.batches.retrieve(batch_id)

        return BatchJob(
            id=batch.id,
            status=batch.status,
            created_at=batch.created_at,
            completed_at=batch.completed_at,
            input_file_id=batch.input_file_id,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            request_counts=dict(batch.request_counts or {}),
        )

    async def get_batch_results(self, job: BatchJob) -> list[BatchResult]:
        """Get results from a completed batch job.

        Args:
            job: Completed BatchJob (must have output_file_id)

        Returns:
            List of BatchResult with responses

        Raises:
            ValueError: If job is not completed or has no output
        """
        if job.status != "completed":
            raise RouterLLMError(f"Batch job not completed: {job.status}")

        if not job.output_file_id:
            raise RouterLLMError("Batch job has no output file")

        client = self._get_client()

        # Download output file
        content = await client.files.content(job.output_file_id)
        text = content.text

        results: list[BatchResult] = []
        for line in text.strip().split("\n"):
            if not line:
                continue

            data_obj: object = json_loads(line)
            if not is_json_dict(data_obj):
                continue
            row: JsonDict = data_obj
            custom_id = get_str(row, "custom_id", "unknown")

            # Check for error
            error = row.get("error")
            if error is not None:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        content="",
                        error=str(error),
                    )
                )
                continue

            # Extract response
            response_obj = row.get("response")
            response = response_obj if isinstance(response_obj, dict) else {}
            body_obj = response.get("body")
            body = body_obj if isinstance(body_obj, dict) else {}
            choices_obj = body.get("choices")
            choices = choices_obj if isinstance(choices_obj, list) else []

            content_text = ""
            if choices and isinstance(choices[0], dict):
                message_obj = choices[0].get("message")
                message = message_obj if isinstance(message_obj, dict) else {}
                raw_content = message.get("content")
                content_text = (
                    raw_content if isinstance(raw_content, str) else str(raw_content or "")
                )

            usage_obj = body.get("usage")
            usage: dict[str, int] | None = None
            if isinstance(usage_obj, dict):
                usage = {
                    str(key): int(value)
                    for key, value in usage_obj.items()
                    if isinstance(value, int)
                }

            results.append(
                BatchResult(
                    custom_id=custom_id,
                    content=content_text,
                    usage=usage,
                )
            )

        return results

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a batch job."""
        client = self._get_client()
        batch = await client.batches.cancel(batch_id)

        return BatchJob(
            id=batch.id,
            status=batch.status,
            created_at=batch.created_at,
        )

    async def list_batches(self, limit: int = 20) -> list[BatchJob]:
        """List recent batch jobs."""
        client = self._get_client()
        batches = await client.batches.list(limit=limit)

        return [
            BatchJob(
                id=b.id,
                status=b.status,
                created_at=b.created_at,
                completed_at=b.completed_at,
                output_file_id=b.output_file_id,
                request_counts=dict(b.request_counts or {}),
            )
            for b in batches.data
        ]

    async def wait_for_completion(
        self,
        batch_id: str,
        *,
        poll_interval: int = 60,
        max_wait: int = 86400,  # 24 hours
    ) -> BatchJob:
        """Wait for batch job to complete.

        Args:
            batch_id: Batch job ID
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            Completed BatchJob

        Raises:
            TimeoutError: If max_wait exceeded
            RuntimeError: If batch failed
        """
        import asyncio

        start = time.time()
        while True:
            job = await self.get_batch(batch_id)

            if job.status == "completed":
                logger.info("Batch %s completed", batch_id)
                return job

            if job.status in ("failed", "expired", "cancelled"):
                raise RouterLLMError(f"Batch job {job.status}: {batch_id}")

            elapsed = time.time() - start
            if elapsed > max_wait:
                raise TimeoutError(f"Batch job timeout after {elapsed:.0f}s")

            logger.debug("Batch %s status: %s, waiting...", batch_id, job.status)
            await asyncio.sleep(poll_interval)


# Helper function for simple batch operations
async def run_batch_completions(
    config: RouterConfig,
    requests: list[JsonDict],
    *,
    model: str = "gpt-4o-mini",
    description: str | None = None,
    wait: bool = True,
) -> list[BatchResult]:
    """Run batch completions with simple interface.

    Args:
        config: Router config
        requests: List of dicts with 'id' and 'messages' keys
        model: Model to use
        description: Optional batch description
        wait: Whether to wait for completion

    Returns:
        List of BatchResult if wait=True, else empty list

    Example:
        results = await run_batch_completions(
            config,
            [
                {"id": "prod-1", "messages": [{"role": "user", "content": "Describe..."}]},
                {"id": "prod-2", "messages": [{"role": "user", "content": "Categorize..."}]},
            ],
            model="gpt-4o-mini",
        )
    """
    client = OpenAIBatchClient(config)

    batch_requests: list[BatchRequest] = []
    for req in requests:
        messages = get_json_dict_list(req, "messages")
        temperature_raw = req.get("temperature")
        temperature = temperature_raw if isinstance(temperature_raw, (int, float)) else None
        max_tokens_raw = req.get("max_tokens")
        max_tokens = max_tokens_raw if isinstance(max_tokens_raw, int) else None
        batch_requests.append(
            BatchRequest(
                custom_id=get_str(req, "id"),
                messages=messages,
                model=model,
                temperature=float(temperature) if temperature is not None else None,
                max_tokens=max_tokens,
            )
        )

    job = await client.create_batch(batch_requests, description=description)

    if not wait:
        logger.info("Batch job created: %s (not waiting)", job.id)
        return []

    completed = await client.wait_for_completion(job.id)
    return await client.get_batch_results(completed)


__all__ = [
    "BatchRequest",
    "BatchResult",
    "BatchJob",
    "OpenAIBatchClient",
    "run_batch_completions",
]
