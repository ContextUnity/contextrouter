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

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextrouter.core import Config

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Single request in a batch job."""

    custom_id: str
    messages: list[dict[str, Any]]
    model: str = "gpt-4o-mini"
    temperature: float | None = None
    max_tokens: int | None = None


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

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy init OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI Batch API requires `openai` package. "
                    "Install with: pip install openai"
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
        """Create a new batch job.

        Args:
            requests: List of batch requests
            description: Optional description for the batch

        Returns:
            BatchJob with ID to track progress
        """
        client = self._get_client()

        # Create JSONL content
        jsonl_lines = []
        for req in requests:
            body: dict[str, Any] = {
                "model": req.model,
                "messages": req.messages,
            }
            if req.temperature is not None:
                body["temperature"] = req.temperature
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens

            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            jsonl_lines.append(json.dumps(line))

        jsonl_content = "\n".join(jsonl_lines)

        # Upload file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write(jsonl_content)
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

        logger.info(f"Uploaded batch input file: {input_file_id}")

        # Create batch job
        batch_response = await client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description or "ContextRouter batch job"},
        )

        logger.info(f"Created batch job: {batch_response.id}")

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
            raise ValueError(f"Batch job not completed: {job.status}")

        if not job.output_file_id:
            raise ValueError("Batch job has no output file")

        client = self._get_client()

        # Download output file
        content = await client.files.content(job.output_file_id)
        text = content.text

        results = []
        for line in text.strip().split("\n"):
            if not line:
                continue

            data = json.loads(line)
            custom_id = data.get("custom_id", "unknown")

            # Check for error
            error = data.get("error")
            if error:
                results.append(BatchResult(
                    custom_id=custom_id,
                    content="",
                    error=str(error),
                ))
                continue

            # Extract response
            response = data.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])

            content_text = ""
            if choices:
                message = choices[0].get("message", {})
                content_text = message.get("content", "")

            usage = body.get("usage")

            results.append(BatchResult(
                custom_id=custom_id,
                content=content_text,
                usage=usage,
            ))

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
                logger.info(f"Batch {batch_id} completed")
                return job

            if job.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch job {job.status}: {batch_id}")

            elapsed = time.time() - start
            if elapsed > max_wait:
                raise TimeoutError(f"Batch job timeout after {elapsed:.0f}s")

            logger.debug(f"Batch {batch_id} status: {job.status}, waiting...")
            await asyncio.sleep(poll_interval)


# Helper function for simple batch operations
async def run_batch_completions(
    config: Config,
    requests: list[dict[str, Any]],
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

    batch_requests = [
        BatchRequest(
            custom_id=req["id"],
            messages=req["messages"],
            model=model,
            temperature=req.get("temperature"),
            max_tokens=req.get("max_tokens"),
        )
        for req in requests
    ]

    job = await client.create_batch(batch_requests, description=description)

    if not wait:
        logger.info(f"Batch job created: {job.id} (not waiting)")
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
